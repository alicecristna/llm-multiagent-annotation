"""
单 Agent 异步标注脚本：使用 AsyncOpenAI + asyncio 并发调用 LLM，
支持断点续传，输出标注结果到 Excel。

用法:
    python batch_annotate_single.py --input data/sample_input.json --output output/result_single.xlsx
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
MAX_CONCURRENT = 50           #
MAX_RETRIES = 3
CHECKPOINT_FILE = "_checkpoint_single.json"
ANNOTATION_COLS = [
    "ann_related", "ann_year",
    "ann_fin_flag", "ann_fin_info",
    "third_party_flag", "third_party_list",
]

# ═══════════════════════════════════════════════════════
# 1. Async DeepSeek 客户端
# ═══════════════════════════════════════════════════════
client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=httpx.Timeout(300.0, connect=60.0),
)

# ═══════════════════════════════════════════════════════
# 2. 读取 prompt
# ═══════════════════════════════════════════════════════
with open("prompts/annotation_prompt.md", "r", encoding="utf-8") as f:
    prompt_md = f.read()
parts = prompt_md.split("## 待标注文本")
system_prompt = parts[0].strip()
user_template = parts[1].strip() if len(parts) > 1 else ""

# ═══════════════════════════════════════════════════════
# 3. 读取数据（在 main() 中初始化）
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
# 4. 断点恢复（在 main() 中初始化）
# ═══════════════════════════════════════════════════════
df = None
results = []
semaphore = None
completed_count = 0
checkpoint_lock = asyncio.Lock()


def advance_completed():
    """推进 completed_count 到第一个 None 的位置，保证连续性"""
    global completed_count
    while completed_count < len(results) and results[completed_count] is not None:
        completed_count += 1


# 从 checkpoint 恢复时初始化 completed_count
advance_completed()


async def save_checkpoint():
    async with checkpoint_lock:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "next_idx": completed_count,
                "results": [r for r in results[:completed_count]]
            }, f, ensure_ascii=False)


async def annotate_one(idx: int) -> dict | None:
    """标注单条，返回 parsed dict 或 None"""
    text = df.iloc[idx]["Activity"]
    if not isinstance(text, str) or text.strip() == "":
        return None

    user_msg = user_template.replace("{ACTIVITY_TEXT}", text)

    async with semaphore:
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
                tokens = resp.usage.total_tokens if resp.usage else "?"
                print(f"[{idx + 1}/{len(df)}] ✅ {elapsed:.0f}s  tokens={tokens}")
                return parsed

            except json.JSONDecodeError as e:
                print(f"[{idx + 1}/{len(df)}] ❌ JSON解析失败 (attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2)
            except Exception as e:
                wait = 5 * attempt
                print(f"[{idx + 1}/{len(df)}] ❌ API失败 (attempt {attempt}, ⏳ {wait}s): {type(e).__name__}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(wait)

    return None


async def main():
    global completed_count, df, semaphore, results

    parser = argparse.ArgumentParser(
        description="Single Agent 金融文本标注（异步并发版）"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/sample_input.json",
        help="输入文件路径 (.json 或 .xlsx)，需包含 'Activity' 列"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/result_single.xlsx",
        help="输出 Excel 文件路径"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=MAX_CONCURRENT,
        help=f"并发数 (默认 {MAX_CONCURRENT})"
    )
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"读取完成: {len(df)} 条数据")
    semaphore = asyncio.Semaphore(args.concurrency)

    # 初始化结果数组和断点恢复
    results = [None] * len(df)
    start_idx = 0

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            ck = json.load(f)
        start_idx = ck.get("next_idx", 0)
        saved = ck.get("results", [])
        for i, r in enumerate(saved):
            if i < len(results):
                results[i] = r
        print(f"从断点恢复: start_idx={start_idx}, 已完成 {len(saved)} 条")

    completed_count = start_idx
    advance_completed()

    pending = list(range(start_idx, len(df)))
    print(f"开始并发标注: {start_idx} → {len(df)}")
    print(f"并发数: {args.concurrency}, 预估剩余: {((len(pending) * 12) / args.concurrency) / 3600:.1f} 小时")
    print("─" * 50)

    # 分批提交，每批 MAX_CONCURRENT * 2 个任务
    BATCH_SIZE = MAX_CONCURRENT * 2
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        tasks = [annotate_one(i) for i in batch]
        batch_results = await asyncio.gather(*tasks)

        for i, r in zip(batch, batch_results):
            results[i] = r

        # 推进连续完成计数（跳过 None 间隙）
        advance_completed()

        # 每批结束存一次 checkpoint
        await save_checkpoint()
        pct = (batch_start + len(batch)) / len(pending) * 100
        print(f"  ── checkpoint 已保存 (批次进度 {min(pct, 100):.0f}%) ──")

    # ═══════════════════════════════════════════════════════
    # 6. 展平 JSON → DataFrame
    # ═══════════════════════════════════════════════════════
    df_out = df.copy()
    for col in ANNOTATION_COLS:
        df_out[col] = None

    for i, r in enumerate(results):
        if r is None:
            continue
        df_out.at[i, "ann_related"] = r.get("ann_related")
        df_out.at[i, "ann_year"] = (
            json.dumps(r.get("ann_year"), ensure_ascii=False)
            if r.get("ann_year") is not None else None
        )
        df_out.at[i, "ann_fin_flag"] = r.get("ann_fin_flag")
        df_out.at[i, "ann_fin_info"] = (
            json.dumps(r.get("ann_fin_info"), ensure_ascii=False)
            if r.get("ann_fin_info") is not None else None
        )
        df_out.at[i, "third_party_flag"] = r.get("third_party_flag")
        df_out.at[i, "third_party_list"] = (
            json.dumps(r.get("third_party_list"), ensure_ascii=False)
            if r.get("third_party_list") is not None else None
        )

    df_out.to_excel(args.output, index=False)

    success = sum(1 for r in results if r is not None)
    errors_idx = [i for i, r in enumerate(results) if r is None]
    print(f"\n{'=' * 50}")
    print(f"完成! {success}/{len(df)} 条成功标注")
    if errors_idx:
        print(f"失败行号: {errors_idx[:20]}{'...' if len(errors_idx) > 20 else ''}")
    print(f"输出: {args.output}")

    if os.path.exists(CHECKPOINT_FILE) and not errors_idx:
        os.remove(CHECKPOINT_FILE)
        print("checkpoint 已清理")


if __name__ == "__main__":
    asyncio.run(main())
