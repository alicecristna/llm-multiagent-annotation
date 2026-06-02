"""
Multi-Agent 异步标注脚本：Actor ↔ Critic 双智能体协作
- Actor Round 1: 初始标注 + 逐字段置信度
- Critic:       审查 Actor 输出，纠正已知错误模式（遵循 prompt4critic.md）
- Actor Round 2: 根据 Critic 反馈修正标注，输出最终结果

架构：Actor(R1) → Critic → Actor(R2)
多行并发处理（Semaphore 控制），单行内三步串行。
"""
import pandas as pd
import json
import time
import os
import re
import asyncio
import argparse
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════
MAX_CONCURRENT = 50          # 并发行数
MAX_RETRIES = 3
CHECKPOINT_FILE = "_checkpoint_multiagent.json"
ANNOTATION_COLS = [
    "ann_related", "ann_year",
    "ann_fin_flag", "ann_fin_info",
    "third_party_flag", "third_party_list",
]
CONFIDENCE_COLS = [f"{c}_confidence" for c in ANNOTATION_COLS]
CRITIC_COLS = ["critic_errors_found", "critic_corrections"]
REVISION_COLS = ["revision_notes"]

client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=httpx.Timeout(300.0, connect=60.0),
)

# ═══════════════════════════════════════════════════════
# 1. 加载 prompt 模板
# ═══════════════════════════════════════════════════════

# --- Actor Round 1 prompt（标注 + 置信度）---
with open("prompts/annotation_prompt.md", "r", encoding="utf-8") as f:
    base_prompt = f.read()

# 修改输出格式：在原有字段基础上，每个字段增加 _confidence
confidence_schema = """
### 置信度说明（必填）
对每个判断字段给出 0.0-1.0 的置信度：
- 1.0 = 完全确定，文本明确无误
- 0.8-0.9 = 高度确定，文本支撑充分
- 0.5-0.7 = 不确定，文本有歧义或信息不足
- <0.5 = 纯推测，不建议作为最终结果

### 修正后的输出格式
```json
{
  "ann_related": 0 或 1,
  "ann_related_confidence": 0.0-1.0,
  "ann_year": [年份列表] 或 null,
  "ann_year_confidence": 0.0-1.0,
  "ann_fin_flag": 0 或 1 或 null,
  "ann_fin_flag_confidence": 0.0-1.0,
  "ann_fin_info": [{"year": 年份, "elements": ["要素1", ...]}] 或 null,
  "ann_fin_info_confidence": 0.0-1.0,
  "third_party_flag": 0 或 1 或 null,
  "third_party_flag_confidence": 0.0-1.0,
  "third_party_list": [{"name": "名称", "type": "类型", "role": "角色"}] 或 null,
  "third_party_list_confidence": 0.0-1.0
}
```
"""
actor_r1_system = base_prompt.split("## 待标注文本")[0].strip() + "\n\n" + confidence_schema
actor_r1_user_template = base_prompt.split("## 待标注文本")[1].strip()

# --- Critic prompt ---
with open("prompts/critic_prompt.md", "r", encoding="utf-8") as f:
    critic_full = f.read()
critic_parts = critic_full.split("## 输入")
critic_system = critic_parts[0].strip()
critic_user_template = critic_parts[1].strip() if len(critic_parts) > 1 else ""

# --- Actor Round 2 prompt（修订）---
actor_r2_system = """你是一个专业的标注修正专家。你现在会收到：
1. 原始违规文本
2. 你第一轮的标注结果（含置信度）
3. Critic Agent 的审查反馈（含错误类型和修改建议）

任务：根据 Critic 的反馈，修正你的标注，输出最终结果。

修正原则：
- 如果 Critic 指出的错误确实合理，采纳并修正
- 如果 Critic 的反馈与原文事实不符，坚持你的原始判断
- 修改任何字段后，检查依赖链是否仍然正确
- 保留原始的置信度，但额外输出一条修正说明

输出格式（严格 JSON，不要任何其他文字）：
```json
{
  "ann_related": 0 或 1,
  "ann_year": [年份列表] 或 null,
  "ann_fin_flag": 0 或 1 或 null,
  "ann_fin_info": [{"year": 年份, "elements": ["要素1", ...]}] 或 null,
  "third_party_flag": 0 或 1 或 null,
  "third_party_list": [{"name": "名称", "type": "类型", "role": "角色"}] 或 null,
  "revision_notes": "说明本次修改了哪些字段，以及修改原因；若未修改则写'无修正，原始标注已通过审查'"
}
```"""

# ═══════════════════════════════════════════════════════
# 2. 读取数据（支持 JSON 和 Excel）
# ═══════════════════════════════════════════════════════
def load_data(filepath: str) -> pd.DataFrame:
    """根据文件扩展名自动选择读取方式"""
    if filepath.endswith('.json'):
        df = pd.read_json(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {filepath}，请使用 .json 或 .xlsx")
    if 'Activity' not in df.columns:
        raise ValueError("输入文件必须包含 'Activity' 列")
    return df

# ═══════════════════════════════════════════════════════
# 3. 断点恢复（在 main() 中初始化）
# ═══════════════════════════════════════════════════════
df = None
results = []
done_flags = []
sem = None
checkpoint_lock = asyncio.Lock()


def advance_completed():
    """返回第一个未完成的行索引"""
    for i, d in enumerate(done_flags):
        if not d:
            return i
    return len(df)


async def save_checkpoint():
    async with checkpoint_lock:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False)


async def call_deepseek(system_prompt: str, user_msg: str, label: str = "") -> dict | None:
    """通用 DeepSeek 调用，返回 parsed JSON 或 None"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            resp = await client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=4096,
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
            elapsed = time.time() - t0
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            print(f"  {label} ✅ {elapsed:.0f}s")
            return parsed
        except json.JSONDecodeError as e:
            print(f"  {label} ❌ JSON解析失败 (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2)
        except Exception as e:
            wait = 5 * attempt
            print(f"  {label} ❌ API失败 (attempt {attempt}, ⏳ {wait}s): {type(e).__name__}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(wait)
    return None


async def process_one_row(idx: int):
    """处理单行的完整 Actor → Critic → Actor 流水线"""
    text = df.iloc[idx]["Activity"]
    if not isinstance(text, str) or text.strip() == "":
        done_flags[idx] = True
        return

    row_result = {}

    async with sem:
        print(f"\n[{idx + 1}/{len(df)}] ── 开始处理 ──")

        # ── Actor Round 1 ──
        r1_user = actor_r1_user_template.replace("{ACTIVITY_TEXT}", text)
        r1 = await call_deepseek(actor_r1_system, r1_user, f"[{idx+1}] Actor-R1")
        if r1 is None:
            return
        row_result["r1"] = r1

        # ── Critic ──
        # 构造 Critic 输入：原始文本 + Actor R1 输出（仅标注字段，不含置信度）
        agent_output_for_critic = {k: r1[k] for k in ANNOTATION_COLS if k in r1}
        critic_user = critic_user_template.replace("{ACTIVITY_TEXT}", text)
        critic_user = critic_user.replace(
            "{AGENT_OUTPUT}",
            json.dumps(agent_output_for_critic, ensure_ascii=False),
        )
        critic = await call_deepseek(critic_system, critic_user, f"[{idx+1}] Critic")
        if critic is None:
            return
        # 提取 Critic 的 corrected_output
        critic_corrected = critic.get("corrected_output", agent_output_for_critic)
        row_result["critic"] = {
            "errors_found": critic.get("errors_found", []),
            "corrections": critic.get("corrections", ""),
            "corrected_output": critic_corrected,
        }

        # ── Actor Round 2 ──
        r2_user = f"""原始违规文本：
{text}

我第一轮的标注结果（含置信度）：
{json.dumps(r1, ensure_ascii=False, indent=2)}

Critic Agent 的审查反馈：
- 发现的错误类型：{critic.get("errors_found", [])}
- 修改建议：{critic.get("corrections", "")}
- Critic 修正后的输出：{json.dumps(critic_corrected, ensure_ascii=False, indent=2)}

请根据以上信息，输出你的最终标注结果。"""
        r2 = await call_deepseek(actor_r2_system, r2_user, f"[{idx+1}] Actor-R2")
        if r2 is None:
            return
        row_result["r2_final"] = r2

        results[idx] = row_result
        done_flags[idx] = True

    # checkpoint（每完成一行就存，避免并发阻塞时丢失）
    if sum(done_flags) % 10 == 0:
        await save_checkpoint()
        completed = sum(done_flags)
        print(f"  ── checkpoint ({completed}/{len(df)}) ──")


async def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent 金融文本标注 (Actor → Critic → Actor)"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/sample_input.json",
        help="输入文件路径 (.json 或 .xlsx)，需包含 'Activity' 列"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/result_multiagent.xlsx",
        help="输出 Excel 文件路径"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=MAX_CONCURRENT,
        help=f"并发数 (默认 {MAX_CONCURRENT})"
    )
    args = parser.parse_args()

    global df, sem, results, done_flags
    df = load_data(args.input)
    print(f"读取完成: {len(df)} 条数据")
    sem = asyncio.Semaphore(args.concurrency)

    # 初始化结果数组和断点恢复
    results = [None] * len(df)
    done_flags = [False] * len(df)

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            ck = json.load(f)
        for i, r in enumerate(ck.get("results", [])):
            if i < len(results) and r is not None:
                results[i] = r
                done_flags[i] = True
        print(f"从断点恢复: 已完成 {sum(done_flags)} 条")

    pending = [i for i in range(len(df)) if not done_flags[i]]
    print(f"开始 Multi-Agent 标注: {len(pending)} 条待处理")
    print(f"并发数: {args.concurrency}, 流水线: Actor(R1) → Critic → Actor(R2)")
    print(f"每行 3 次 API 调用, 总调用约: {len(pending) * 3} 次")
    print("─" * 50)

    # 分批提交
    BATCH_SIZE = MAX_CONCURRENT * 2
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        tasks = [process_one_row(i) for i in batch]
        await asyncio.gather(*tasks)
        await save_checkpoint()

    # ═══════════════════════════════════════════════════════
    # 展平结果到 DataFrame
    # ═══════════════════════════════════════════════════════
    df_out = df.copy()
    # 最终标注列
    for col in ANNOTATION_COLS:
        df_out[col] = None
    # 置信度列
    for col in CONFIDENCE_COLS:
        df_out[col] = None
    # Critic 列
    for col in CRITIC_COLS:
        df_out[col] = None
    # 修订说明列
    for col in REVISION_COLS:
        df_out[col] = None

    for i, r in enumerate(results):
        if r is None:
            continue

        # Actor R1 置信度
        r1 = r.get("r1", {})
        for col in ANNOTATION_COLS:
            df_out.at[i, col + "_confidence"] = r1.get(col + "_confidence")

        # Critic 反馈
        critic = r.get("critic", {})
        df_out.at[i, "critic_errors_found"] = json.dumps(
            critic.get("errors_found", []), ensure_ascii=False
        )
        df_out.at[i, "critic_corrections"] = critic.get("corrections", "")

        # Actor R2 最终结果
        final = r.get("r2_final", r1)  # 若 R2 失败，退回到 R1
        for col in ANNOTATION_COLS:
            val = final.get(col)
            if isinstance(val, (list, dict)):
                df_out.at[i, col] = json.dumps(val, ensure_ascii=False)
            else:
                df_out.at[i, col] = val

        df_out.at[i, "revision_notes"] = final.get("revision_notes", "")

    df_out.to_excel(args.output, index=False)

    success = sum(done_flags)
    failed = [i for i, d in enumerate(done_flags) if not d]
    print(f"\n{'=' * 50}")
    print(f"Multi-Agent 标注完成! {success}/{len(df)} 条成功")
    if failed:
        print(f"失败行号: {failed}")
    print(f"输出: {args.output}")

    if os.path.exists(CHECKPOINT_FILE) and not failed:
        os.remove(CHECKPOINT_FILE)
        print("checkpoint 已清理")


if __name__ == "__main__":
    asyncio.run(main())
