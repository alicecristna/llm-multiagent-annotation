# LLM Multi-Agent Financial Text Annotation

基于 **Reflexion 架构**的 Multi-Agent 金融文本智能标注系统。通过 Actor ↔ Critic 双智能体协作，提升上市公司违规行为文本的分类准确率。

## Architecture

```
                    ┌─────────────┐
   Input Text ────▶ │ Actor (R1)  │ ──▶ 初始标注 + 置信度
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Critic    │ ──▶ 审查 + 指出错误 + 修正建议
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Actor (R2)  │ ──▶ 根据反馈修正 → 最终输出
                    └─────────────┘
```

- **Actor (Round 1)**: 接收违规文本，输出三层级标注（年报相关性 → 财务信息识别 → 第三方配合舞弊识别），每个字段附带置信度
- **Critic**: 基于预先研究的 5 种高频错误模式（推理疲劳、浅层关键词匹配、追溯调整边界、资金占用误判、依赖链泄漏），逐字段审查 Actor 输出
- **Actor (Round 2)**: 根据 Critic 反馈修正标注——采纳合理建议，对与原文事实不符的反馈坚持原始判断

## Single Agent vs Multi Agent

| 维度 | Single Agent | Multi Agent (Reflexion) |
|------|-------------|------------------------|
| 架构 | 单次 LLM 调用 | Actor(R1) → Critic → Actor(R2)，3 次调用 |
| 错误自纠正 | 无 | Critic 审查 → Actor 修正 |
| 置信度信息 | 无 | 每个字段附带 0-1 置信度 |
| 适用场景 | 低复杂度、高吞吐 | 高精度要求、边界 case 多的任务 |

> 在 178 条验证集上，Multi Agent 的 F1 Score 相比 Single Agent 有显著提升。详细对比数据见结课报告。

## Key Design Decisions

**1. Critic 只审查，不写最终结果。**
Actor Round 2 保有最终决定权——Critic 也会犯错（尤其在置信度边界样本上），Actor 需要独立判断是否接受反馈。Disagreement 本身也是有价值的诊断信息。

**2. 至多 1 轮修正。**
分类任务中 Actor→Critic→Actor 的一轮循环已捕获大部分错误。多轮修正对简单标签修改的边际收益递减，但 token 消耗线性增长。

**3. Critic 的错误模式是数据驱动的。**
Critic 的审查维度不是凭空设计，而是从 186 条样本的实际标注错误中归纳出的 5 种高频错误模式——这保证了 Critic 的反馈是靶向的，而非泛泛的"再检查一遍"。

**4. 依赖链不可逆。**
标注任务有严格的层级依赖（`ann_related=0` → 后续字段全部 `null`）。Critic 在逐字段审查时，任何上游字段的修改会自动级联清空下游字段，确保输出一致性。

## Tech Stack

- **Language**: Python 3.10+
- **LLM**: DeepSeek V4 Pro (via OpenAI-compatible SDK)
- **Async**: `asyncio` + `AsyncOpenAI`，支持 50 并发
- **Data**: Pandas, openpyxl
- **Prompt Engineering**: TCREI 框架 (Task-Context-Reference-Evaluate-Iterate)

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas openpyxl openai python-dotenv httpx
```

### 2. Set API Key

```bash
export DEEPSEEK_API_KEY="your-api-key"
# 或在项目根目录创建 .env 文件:
# DEEPSEEK_API_KEY=your-api-key
```

### 3. Run Single Agent

```bash
python batch_annotate_single.py --input data/sample_input.json --output output/result_single.xlsx
```

### 4. Run Multi Agent

```bash
python batch_annotate_multiagent.py --input data/sample_input.json --output output/result_multiagent.xlsx
```

**命令行参数：**

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `data/sample_input.json` | 输入文件（.json 或 .xlsx），需含 `Activity` 列 |
| `--output` | `-o` | `output/result_*.xlsx` | 输出 Excel 路径 |
| `--concurrency` | `-c` | Single: 5, Multi: 50 | 并发数 |

### 5. Evaluate Results

计算 Single Agent vs Multi Agent 的 F1 Score：

```bash
python evaluate.py --ground data/sample_ground_truth.json --predict output/result_single.xlsx
python evaluate.py --ground data/sample_ground_truth.json --predict output/result_multiagent.xlsx
```

**评估字段：** `ann_related` / `ann_year` / `ann_fin_flag` / `ann_fin_info` / `third_party_flag` / `third_party_list`

输出每个字段的 Precision / Recall / F1 / Accuracy，以及错误详情。

### 6. Input Format

输入文件为 JSON 数组，每条记录需包含 `Activity` 字段：

```json
[
  {
    "id": 1,
    "Activity": "公司在2019年、2020年年度报告中，通过虚构销售合同..."
  }
]
```

参考 `data/sample_input.json` 中的 12 条示例数据。

## File Structure

```
├── README.md
├── .gitignore
├── data/
│   ├── sample_input.json          # 12 条 mock 示例数据
│   └── sample_ground_truth.json   # Ground truth 标注（用于 F1 计算）
├── prompts/
│   ├── annotation_prompt.md       # Actor 标注 Prompt（TCREI 框架）
│   └── critic_prompt.md           # Critic 审查 Prompt（含 5 种错误模式）
├── batch_annotate_single.py       # 单 Agent 异步标注脚本
├── batch_annotate_multiagent.py   # 多 Agent Actor→Critic→Actor 脚本
├── evaluate.py                    # F1/Precision/Recall 评估脚本
└── output/
    └── .gitkeep
```

## Data Note

`data/sample_input.json` 中的 12 条示例数据均为基于公开公司信息的虚构违规描述，仅用于展示 pipeline 的输入输出格式。项目实际训练和验证使用的数据来自课程提供，未包含在本仓库中。

## References

- Shinn, N. et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
- Yao, S. et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023.
- Wu, E. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." 2023.
