"""
评估脚本：计算 Single Agent 与 Multi Agent 标注结果的 F1 Score。

用法:
    python evaluate.py --ground data/sample_ground_truth.json --predict output/result_multiagent.xlsx
    python evaluate.py --ground data/sample_ground_truth.json --predict output/result_single.xlsx

输出每个标注字段的 Precision / Recall / F1，以及总计。
"""

import json
import argparse
import pandas as pd
import numpy as np
from typing import Any


# ═══════════════════════════════════════════════════════
# 字段级比较函数
# ═══════════════════════════════════════════════════════

def compare_binary(y_true, y_pred) -> tuple[bool, str]:
    """比较二值字段 (0/1)，返回 (正确?, 详情)"""
    if y_true == y_pred:
        return True, "match"
    return False, f"true={y_true}, pred={y_pred}"


def compare_years(y_true, y_pred) -> tuple[bool, str]:
    """比较年份列表 - 精确集合匹配"""
    t = set(y_true or [])
    p = set(y_pred or [])
    if t == p:
        return True, "match"
    missing = t - p
    extra = p - t
    msg = []
    if missing:
        msg.append(f"missing={sorted(missing)}")
    if extra:
        msg.append(f"extra={sorted(extra)}")
    return False, ", ".join(msg)


def compare_fin_info(y_true, y_pred) -> tuple[bool, str]:
    """比较 ann_fin_info: [{year, elements}, ...]

    按年匹配，每年比较 elements 集合。
    年份缺失或多余算部分错误。
    """
    t = y_true or []
    p = y_pred or []

    # 转为 {year: set(elements)}
    t_map = {item["year"]: set(item.get("elements", [])) for item in t}
    p_map = {item["year"]: set(item.get("elements", [])) for item in p}

    errors = []
    all_years = set(t_map.keys()) | set(p_map.keys())

    for yr in sorted(all_years):
        t_elems = t_map.get(yr, set())
        p_elems = p_map.get(yr, set())
        if t_elems != p_elems:
            missing = sorted(t_elems - p_elems)
            extra = sorted(p_elems - t_elems)
            parts = []
            if missing:
                parts.append(f"{yr}:missing={missing}")
            if extra:
                parts.append(f"{yr}:extra={extra}")
            if not t_elems:
                parts.append(f"{yr}:unexpected")
            if not p_elems:
                parts.append(f"{yr}:missed")
            errors.extend(parts)

    if not errors:
        return True, "match"
    return False, "; ".join(errors)


def compare_third_party_list(y_true, y_pred) -> tuple[bool, str]:
    """比较 third_party_list: [{name, type, role}, ...]

    按 name 做模糊匹配（name 包含关系的视为同一实体）。
    """
    t = y_true or []
    p = y_pred or []

    def _fuzzy_key(name: str) -> str:
        return name.strip().lower()

    t_names = {_fuzzy_key(item["name"]): item for item in t}
    p_names = {_fuzzy_key(item["name"]): item for item in p}

    # 先做完全 name 匹配，再做包含匹配
    matched_t = set()
    matched_p = set()
    errors = []

    for t_key, t_item in t_names.items():
        found = None
        for p_key, p_item in p_names.items():
            if p_key in matched_p:
                continue
            # 双向包含匹配
            if t_key in p_key or p_key in t_key:
                found = p_key
                break
        if found:
            matched_t.add(t_key)
            matched_p.add(found)
            # 即使 name 匹配了，也校验 type 和 role
            p_item = p_names[found]
            if t_item.get("type", "").strip() != p_item.get("type", "").strip():
                errors.append(f"{t_item['name']}:type mismatch")
            if t_item.get("role", "").strip() != p_item.get("role", "").strip():
                errors.append(f"{t_item['name']}:role mismatch")
        else:
            errors.append(f"{t_item['name']}:not found in predictions")

    unmatched_p = set(p_names.keys()) - matched_p
    for p_key in unmatched_p:
        errors.append(f"{p_names[p_key]['name']}:unexpected in predictions")

    if not errors:
        return True, "match"
    return False, "; ".join(errors)


# ═══════════════════════════════════════════════════════
# 逐条评估
# ═══════════════════════════════════════════════════════

def evaluate_all(ground: list[dict], predictions: list[dict]) -> dict:
    """逐条评估所有字段，累计 TP/FP/FN"""
    fields = [
        "ann_related",
        "ann_year",
        "ann_fin_flag",
        "ann_fin_info",
        "third_party_flag",
        "third_party_list",
    ]

    metrics = {f: {"tp": 0, "fp": 0, "fn": 0, "errors": []} for f in fields}

    # 按 id 对齐
    pred_map = {p.get("id") if isinstance(p, dict) else i: p for i, p in enumerate(predictions)}
    if not isinstance(predictions[0], dict) or "id" not in predictions[0]:
        # 无 id 列，按行号对齐
        pred_map = {i: p for i, p in enumerate(predictions)}
        gt_map = {i: g for i, g in enumerate(ground)}
    else:
        gt_map = {g["id"]: g for g in ground}

    common_ids = set(gt_map.keys()) & set(pred_map.keys())
    if not common_ids:
        raise ValueError(
            "Ground truth 和 predictions 没有可对齐的 id。"
            "请确保两者有相同的 id 列，或行数相同。"
        )

    for idx in sorted(common_ids):
        gt = gt_map[idx]
        pred = pred_map[idx]

        # --- ann_related ---
        correct, detail = compare_binary(gt.get("ann_related"), pred.get("ann_related"))
        if correct:
            metrics["ann_related"]["tp"] += 1
        else:
            # 判断是 FP 还是 FN
            t_val, p_val = gt.get("ann_related"), pred.get("ann_related")
            if p_val == 1 and t_val != 1:
                metrics["ann_related"]["fp"] += 1
            elif p_val != 1 and t_val == 1:
                metrics["ann_related"]["fn"] += 1
            else:
                metrics["ann_related"]["fp"] += 1  # 都错算 FP
            metrics["ann_related"]["errors"].append(f"id={idx}: {detail}")

        # --- ann_year (仅在 ann_related=1 且双方一致时评估) ---
        if gt.get("ann_related") == 1 and pred.get("ann_related") == 1:
            correct, detail = compare_years(gt.get("ann_year"), pred.get("ann_year"))
        elif gt.get("ann_related") == 0 and pred.get("ann_related") == 0:
            correct, detail = True, "both null (correct)"  # 都为空，正确
        else:
            correct, detail = False, "ann_related mismatch, skip"

        if correct:
            metrics["ann_year"]["tp"] += 1
        else:
            metrics["ann_year"]["errors"].append(f"id={idx}: {detail}")
            # 区分 FP/FN
            t_years = set(gt.get("ann_year") or [])
            p_years = set(pred.get("ann_year") or [])
            if p_years - t_years:
                metrics["ann_year"]["fp"] += 1
            if t_years - p_years:
                metrics["ann_year"]["fn"] += 1

        # --- ann_fin_flag ---
        if gt.get("ann_related") == 1 and pred.get("ann_related") == 1:
            correct, detail = compare_binary(gt.get("ann_fin_flag"), pred.get("ann_fin_flag"))
        elif gt.get("ann_related") == 0 and pred.get("ann_related") == 0:
            correct, detail = True, "both null (correct)"
        else:
            correct, detail = False, "ann_related mismatch, skip"

        if correct:
            metrics["ann_fin_flag"]["tp"] += 1
        else:
            t_val, p_val = gt.get("ann_fin_flag"), pred.get("ann_fin_flag")
            if p_val == 1 and t_val != 1:
                metrics["ann_fin_flag"]["fp"] += 1
            elif p_val != 1 and t_val == 1:
                metrics["ann_fin_flag"]["fn"] += 1
            else:
                metrics["ann_fin_flag"]["fp"] += 1
            metrics["ann_fin_flag"]["errors"].append(f"id={idx}: {detail}")

        # --- ann_fin_info (仅在 ann_fin_flag=1 且双方一致时评估) ---
        both_fin1 = (gt.get("ann_fin_flag") == 1 and pred.get("ann_fin_flag") == 1)
        if both_fin1:
            correct, detail = compare_fin_info(gt.get("ann_fin_info"), pred.get("ann_fin_info"))
        elif gt.get("ann_fin_flag") == pred.get("ann_fin_flag"):
            correct, detail = True, "both null/0 (correct)"
        else:
            correct, detail = False, "ann_fin_flag mismatch, skip"

        if correct:
            metrics["ann_fin_info"]["tp"] += 1
        else:
            metrics["ann_fin_info"]["errors"].append(f"id={idx}: {detail}")
            metrics["ann_fin_info"]["fp"] += 1  # fin_info 主要看预测多了什么

        # --- third_party_flag ---
        both_rel1 = (gt.get("ann_related") == 1 and pred.get("ann_related") == 1)
        if both_rel1:
            correct, detail = compare_binary(gt.get("third_party_flag"), pred.get("third_party_flag"))
        elif gt.get("ann_related") == 0 and pred.get("ann_related") == 0:
            correct, detail = True, "both null (correct)"
        else:
            correct, detail = False, "ann_related mismatch, skip"

        if correct:
            metrics["third_party_flag"]["tp"] += 1
        else:
            t_val, p_val = gt.get("third_party_flag"), pred.get("third_party_flag")
            if p_val == 1 and t_val != 1:
                metrics["third_party_flag"]["fp"] += 1
            elif p_val != 1 and t_val == 1:
                metrics["third_party_flag"]["fn"] += 1
            else:
                metrics["third_party_flag"]["fp"] += 1
            metrics["third_party_flag"]["errors"].append(f"id={idx}: {detail}")

        # --- third_party_list (仅在 third_party_flag=1 且双方一致时评估) ---
        both_tp1 = (gt.get("third_party_flag") == 1 and pred.get("third_party_flag") == 1)
        if both_tp1:
            correct, detail = compare_third_party_list(gt.get("third_party_list"), pred.get("third_party_list"))
        elif gt.get("third_party_flag") == pred.get("third_party_flag"):
            correct, detail = True, "both null/0 (correct)"
        else:
            correct, detail = False, "third_party_flag mismatch, skip"

        if correct:
            metrics["third_party_list"]["tp"] += 1
        else:
            metrics["third_party_list"]["errors"].append(f"id={idx}: {detail}")
            metrics["third_party_list"]["fp"] += 1

    # 计算 P/R/F1
    for f in fields:
        tp = metrics[f]["tp"]
        fp = metrics[f]["fp"]
        fn = metrics[f]["fn"]
        n = tp + fp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = tp / n if n > 0 else 0.0

        metrics[f]["precision"] = round(prec, 4)
        metrics[f]["recall"] = round(rec, 4)
        metrics[f]["f1"] = round(f1, 4)
        metrics[f]["accuracy"] = round(acc, 4)
        metrics[f]["n_samples"] = n

    return metrics


# ═══════════════════════════════════════════════════════
# 加载函数
# ═══════════════════════════════════════════════════════

def load_ground_truth(filepath: str) -> list[dict]:
    """加载 ground truth JSON 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    return data


def load_predictions(filepath: str) -> list[dict]:
    """加载预测结果（JSON 或 Excel）

    JSON: 每个元素是包含标注字段的 dict
    Excel: 包含 ann_related, ann_year, ... 列的表格
    """
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
        return data

    elif filepath.endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
        preds = []
        for _, row in df.iterrows():
            pred = {}
            for col in ["id", "ann_related", "ann_year", "ann_fin_flag",
                        "ann_fin_info", "third_party_flag", "third_party_list"]:
                if col in df.columns:
                    val = row[col]
                    # Excel 中 JSON 字段以字符串形式存储，需要反序列化
                    if col in ("ann_year", "ann_fin_info", "third_party_list"):
                        if isinstance(val, str) and val.strip():
                            try:
                                val = json.loads(val)
                            except json.JSONDecodeError:
                                val = None
                        elif pd.isna(val):
                            val = None
                    elif pd.isna(val):
                        val = None
                    pred[col] = val
            preds.append(pred)
        return preds

    else:
        raise ValueError(f"不支持的文件格式: {filepath}")


# ═══════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════

def print_report(metrics: dict, label: str = "Evaluation"):
    """打印评估报告"""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}\n")

    # 表格头
    header = f"{'Field':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10} {'N':>6}"
    print(header)
    print("-" * 70)

    fields_order = [
        "ann_related",
        "ann_year",
        "ann_fin_flag",
        "ann_fin_info",
        "third_party_flag",
        "third_party_list",
    ]

    for f in fields_order:
        m = metrics[f]
        print(
            f"  {f:<20}"
            f" {m['precision']:>10.4f}"
            f" {m['recall']:>10.4f}"
            f" {m['f1']:>10.4f}"
            f" {m['accuracy']:>10.4f}"
            f" {m['n_samples']:>6}"
        )

    # 宏平均
    macro_f1 = np.mean([metrics[f]["f1"] for f in fields_order])
    macro_prec = np.mean([metrics[f]["precision"] for f in fields_order])
    macro_rec = np.mean([metrics[f]["recall"] for f in fields_order])

    print("-" * 70)
    print(
        f"  {'MACRO AVG':<20}"
        f" {macro_prec:>10.4f}"
        f" {macro_rec:>10.4f}"
        f" {macro_f1:>10.4f}"
        f" {'—':>10}"
        f" {'—':>6}"
    )
    print()

    # 错误详情（最多显示 10 条）
    for f in fields_order:
        errs = metrics[f].get("errors", [])
        if errs:
            print(f"  ⚠ {f} 错误 ({len(errs)} 条):")
            for e in errs[:5]:
                print(f"      {e}")
            if len(errs) > 5:
                print(f"      ... and {len(errs) - 5} more")
            print()

    print(f"{'=' * 70}\n")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="评估金融文本标注结果 — 计算各字段 Precision / Recall / F1"
    )
    parser.add_argument(
        "--ground", "-g",
        required=True,
        help="Ground truth 文件路径 (.json)"
    )
    parser.add_argument(
        "--predict", "-p",
        required=True,
        help="预测结果文件路径 (.json 或 .xlsx)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出评估报告到 JSON 文件（可选）"
    )
    args = parser.parse_args()

    gt = load_ground_truth(args.ground)
    preds = load_predictions(args.predict)

    print(f"Ground Truth: {len(gt)} 条")
    print(f"Predictions:  {len(preds)} 条")

    metrics = evaluate_all(gt, preds)
    print_report(metrics, label=f"Predictions: {args.predict}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"评估报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
