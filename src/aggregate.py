"""
多次采样的聚合。

为什么需要：视觉模型对同一张图重复回答会抖。实测同一对照片两次调用，
主体结构封顶层数差 2~5 层（16 对 13、20 对 15），单次结果不可采信。

聚合规则：
- 工序状态：多数票；平票时取更保守的那个（unclear > not_started > in_progress > done）
- 楼层数：各次明细里"最高塔封顶层数"的中位数
- changes / risks / summary：取与最终共识最接近的那一次（而不是硬拼凑）
"""

from __future__ import annotations

from collections import Counter
from statistics import median

# 平票时的保守程度：数字越小越保守
_CONSERVATIVE = {"unclear": 0, "not_started": 1, "in_progress": 2, "done": 3}


def _max_floors(entry: dict) -> float | None:
    """单次样本里某工序的封顶层数（取各塔最高）。"""
    detail = [d for d in (entry.get("count_detail") or []) if isinstance(d, dict)]
    vals = [float(d["done"]) for d in detail
            if isinstance(d.get("done"), (int, float))]
    if vals:
        return max(vals)
    cnt = entry.get("count") or {}
    d = cnt.get("done")
    return float(d) if isinstance(d, (int, float)) else None


def _max_inc(entry: dict) -> float | None:
    """单次样本里最高塔的新增层数（图2 相对图1）。"""
    detail = [d for d in (entry.get("count_detail") or []) if isinstance(d, dict)]
    vals = [float(d["inc"]) for d in detail
            if isinstance(d.get("inc"), (int, float))]
    if vals:
        return max(vals)
    cnt = entry.get("count") or {}
    v = cnt.get("inc") if isinstance(cnt, dict) else None
    return float(v) if isinstance(v, (int, float)) else None


def _total_floors(entry: dict) -> float | None:
    """单次样本里估计的总层数。"""
    cnt = entry.get("count") or {}
    t = cnt.get("total") if isinstance(cnt, dict) else None
    if isinstance(t, (int, float)) and t > 0:
        return float(t)
    detail = [d for d in (entry.get("count_detail") or []) if isinstance(d, dict)]
    vals = [float(d["total"]) for d in detail
            if isinstance(d.get("total"), (int, float)) and d["total"] > 0]
    return float(median(vals)) if vals else None


def aggregate(vlms: list[dict]) -> tuple[dict, dict]:
    """把 N 次结果合成一份，返回 (合并结果, 统计信息)。"""
    if not vlms:
        raise ValueError("没有可聚合的样本")
    if len(vlms) == 1:
        return vlms[0], {"samples": 1, "agreement": {}}

    ids: list[str] = []
    for v in vlms:
        for s in v.get("stages", []):
            sid = s.get("id")
            if sid and sid not in ids:
                ids.append(sid)

    merged_stages: list[dict] = []
    agreement: dict[str, str] = {}
    for sid in ids:
        entries = [s for v in vlms for s in v.get("stages", [])
                   if s.get("id") == sid]
        if not entries:
            continue

        statuses = [str(e.get("status", "unclear")).lower() for e in entries]
        cnt = Counter(statuses)
        top_n = cnt.most_common(1)[0][1]
        cands = [s for s, n in cnt.items() if n == top_n]
        if len(cands) > 1:
            cands.sort(key=lambda s: _CONSERVATIVE.get(s, 0))
        status = cands[0]
        agreement[sid] = f"{cnt[status]}/{len(entries)}"

        rep = next((e for e in entries
                    if str(e.get("status", "")).lower() == status), entries[0])
        merged = dict(rep)
        merged["status"] = status

        floors = [f for f in (_max_floors(e) for e in entries) if f is not None]
        totals = [t for t in (_total_floors(e) for e in entries) if t is not None]
        incs = [i for i in (_max_inc(e) for e in entries) if i is not None]
        if floors or incs:
            done_med = median(floors) if floors else None
            total_med = median(totals) if totals else (max(floors) if floors else None)
            inc_med = median(incs) if incs else None
            merged["count"] = {"done": done_med, "total": total_med, "inc": inc_med}
            merged["count_detail"] = None
            if floors:
                agreement[f"{sid}:floors"] = (
                    f"中位数 {done_med:g}，各次 {sorted(floors)}，"
                    f"极差 {int(max(floors) - min(floors))}")
            if incs:
                agreement[f"{sid}:inc"] = (
                    f"中位数 {inc_med:g}，各次 {sorted(incs)}")
        merged_stages.append(merged)

    # 选与共识最接近的那次作为文字部分（changes/risks/summary）的来源
    def match_score(v: dict) -> int:
        by_id = {s.get("id"): str(s.get("status", "")).lower()
                 for s in v.get("stages", [])}
        return sum(1 for s in merged_stages
                   if by_id.get(s.get("id")) == s.get("status"))

    best = max(vlms, key=match_score)
    merged_out = {
        "stages": merged_stages,
        "changes": best.get("changes", []),
        "risks": best.get("risks", []),
        "summary": best.get("summary", ""),
    }
    return merged_out, {"samples": len(vlms), "agreement": agreement}
