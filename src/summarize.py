"""
汇总多组验证结果：按"后期照片日期"排序，输出时间线表并检查进度是否单调递增。

用法：python src/summarize.py [output 目录]
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stages import compute_progress

import glob
import json
import os
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT = {"done": "已完成", "in_progress": "进行中",
         "not_started": "未开始", "unclear": "看不清"}
STAGE_ORDER = ["earthwork", "foundation", "structure", "masonry", "roof",
               "facade", "mep", "finishing", "landscape"]


def collect(out_root: str) -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(out_root, "*", "vlm_raw.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:                               # noqa: BLE001
            continue
        meta, vlm = d.get("meta", {}), d.get("vlm", {})
        prog = compute_progress(vlm.get("stages", [])) or d.get("progress", {})
        rows.append({
            "dir": os.path.basename(os.path.dirname(f)),
            "before_date": meta.get("before_date", ""),
            "after_date": meta.get("after_date", ""),
            "progress": prog.get("progress_pct"),
            "floor": prog.get("progress_floor"),
            "counted": prog.get("counted_weight"),
            "floors": next((x["count_done"] for x in prog.get("stages", [])
                            if x.get("factor_source") == "count"), None),
            "floors_txt": next((f"{x['count_done']}/{x['count_total']}"
                                for x in prog.get("stages", [])
                                if x.get("factor_source") == "count"), "-"),
            "inc": next((x.get("count_inc") for x in prog.get("stages", [])
                         if x.get("factor_source") == "count"), None),
            "days": meta.get("days"),
            "struct_pct": next((round(x["count_done"] / x["count_total"] * 100, 1)
                                for x in prog.get("stages", [])
                                if x.get("factor_source") == "count"
                                and x.get("count_total")), None),
            "statuses": {s.get("id"): s.get("status") for s in vlm.get("stages", [])},
            "conf": {s.get("id"): s.get("confidence") for s in vlm.get("stages", [])},
            "changes": len(vlm.get("changes", [])),
            "risks": len(vlm.get("risks", [])),
            "summary": vlm.get("summary", ""),
        })
    # 只保留有日期的、且能解析的
    rows = [r for r in rows if r["after_date"]]
    rows.sort(key=lambda r: r["after_date"])
    return rows


def main() -> None:
    out_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "output")
    rows = collect(out_root)
    if not rows:
        print("没找到结果。先跑 batch_validate.sh。")
        return

    print("=" * 78)
    print(" 时间线验证（按后期照片日期排序）")
    print("=" * 78)
    print(f"{'后期日期':12s}{'区间':24s}{'封顶层数':>10s}{'增量':>7s}"
          f"{'层/月':>7s}{'结构完成':>9s}{'进度(保守)':>11s}{'权重':>6s}")
    for r in rows:
        span = f"{r['before_date']}~{r['after_date']}"
        p = f"{r['progress']:.1f}%" if r["progress"] is not None else "-"
        fl = f"{r['floor']:.1f}%" if r["floor"] is not None else "-"
        inc = r.get("inc")
        days = r.get("days")
        inc_txt = f"+{inc:g}" if isinstance(inc, (int, float)) else "-"
        rate_txt = (f"{inc / days * 30:.1f}" if isinstance(inc, (int, float)) and days
                    else "-")
        print(f"{r['after_date']:12s}{span:24s}{str(r['floors_txt']):>10s}"
              f"{inc_txt:>7s}{rate_txt:>7s}"
              f"{(f'{r[chr(39)+chr(115)+chr(116)+chr(114)+chr(117)+chr(99)+chr(116)+chr(95)+chr(112)+chr(99)+chr(116)]}%' if False else '')}"
              f"{str(r['struct_pct']) + '%' if r.get('struct_pct') is not None else '-':>9s}"
              f"{fl:>11s}{str(r['counted']):>6s}")

    print("\n单调性检查（越晚的照片，数值应不减）:")
    vals = [(r["after_date"], r["floor"]) for r in rows
            if r["floor"] is not None]
    print("  [保守口径进度]")
    ok = True
    for (d1, v1), (d2, v2) in zip(vals, vals[1:]):
        flag = "OK " if v2 >= v1 else "下降"
        if v2 < v1:
            ok = False
        print(f"  {d1} {v1:.1f}%  ->  {d2} {v2:.1f}%   {flag} "
              f"({v2 - v1:+.1f})")
    print(f"  结论：{'单调递增，通过' if ok else '存在倒退，需检查'}")

    fv = [(r["after_date"], r["floors"]) for r in rows
          if isinstance(r["floors"], (int, float))]
    if len(fv) >= 2:
        print("  [封顶层数]")
        ok2 = True
        for (d1, v1), (d2, v2) in zip(fv, fv[1:]):
            flag = "OK " if v2 >= v1 else "下降"
            if v2 < v1:
                ok2 = False
            print(f"  {d1} {v1:g} 层  ->  {d2} {v2:g} 层   {flag} ({v2 - v1:+g})")
        print(f"  结论：{'单调递增，通过' if ok2 else '存在倒退，需检查'}")

    rates = [r["inc"] / r["days"] * 30 for r in rows
             if isinstance(r.get("inc"), (int, float)) and r.get("days")]
    if len(rates) >= 2:
        mean = sum(rates) / len(rates)
        sd = (sum((x - mean) ** 2 for x in rates) / len(rates)) ** 0.5
        cv = sd / mean if mean else 0
        print(f"\n  [施工速率一致性] 各区间约 {mean:.1f} 层/月，"
              f"标准差 {sd:.2f}，变异系数 {cv:.0%}")
        print(f"  {'速率稳定，说明增量计数可信' if cv < 0.5 else '速率波动大，增量计数仍不稳'}")

    print("\n工序状态矩阵:")
    hdr = f"{'工序':12s}" + "".join(f"{r['after_date'][5:]:>12s}" for r in rows)
    print(hdr)
    for sid in STAGE_ORDER:
        line = f"{sid:12s}"
        for r in rows:
            st = SHORT.get(r["statuses"].get(sid), "-")
            line += f"{st:>12s}"
        print(line)

    md = ["# 多组样本验证汇总\n",
          "| 后期日期 | 区间 | 进度 | 保守下限 | 可判权重 | 新增条目 |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['after_date']} | {r['before_date']}~{r['after_date']} "
                  f"| {r['progress']}% | {r['floor']}% | {r['counted']} "
                  f"| {r['changes']} |")
    md.append(f"\n**单调性**：{'通过' if ok else '存在倒退'}\n")
    md.append("## 各组摘要\n")
    for r in rows:
        md.append(f"- **{r['after_date']}**：{r['summary']}")
    p = os.path.join(out_root, "summary.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n汇总已写入 {p}")


if __name__ == "__main__":
    main()
