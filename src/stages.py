"""
工序清单与权重配置。

这是整个方案里唯一需要"按项目定制"的东西：
不同工程（住宅/厂房/市政）的工序和权重完全不同，改这里就行，不用动代码。

weight 是该项在总进度里的占比权重，全部加起来建议 = 100。
"""

from __future__ import annotations

# 状态 -> 折算系数（用于算进度）
STATUS_FACTOR = {
    "done": 1.0,
    "in_progress": 0.5,
    "not_started": 0.0,
    # unclear 不参与分母，避免模型看不清就把进度拉低
}

STAGES: list[dict] = [
    {"id": "earthwork", "name": "土方开挖与基坑支护", "weight": 5,
     "visibility": "exterior",
     "hint": "看基坑轮廓、边坡支护、降水井、土方堆放"},
    {"id": "foundation", "name": "基础与地下室结构", "weight": 12,
     "visibility": "exterior",
     "hint": "看筏板/承台、地下室墙体顶板、地下层数"},
    {"id": "structure", "name": "主体结构（楼板/柱/剪力墙）", "weight": 35,
     "visibility": "exterior", "metric": "count", "unit": "层",
     # count_total 填该建筑主体结构的总层数（从图纸或合同工期表拿）；
     # 留空则用模型估计值，但模型估的总层数会在 32~35 之间飘，只能做参考。
     "count_total": 32,   # TNB 项目：按首期观测估计，实际应用请填图纸总层数
     "hint": "数各塔楼已封顶的楼层数；看模板支撑、钢筋绑扎、混凝土浇筑面"},
    {"id": "masonry", "name": "砌体与二次结构", "weight": 10,
     "visibility": "interior",
     "hint": "看填充墙、构造柱、过梁、圈梁"},
    {"id": "roof", "name": "屋面与防水", "weight": 5,
     "visibility": "exterior",
     "hint": "看屋面找坡层、防水卷材、女儿墙、出屋面结构"},
    {"id": "facade", "name": "外立面（幕墙/门窗/涂料）", "weight": 12,
     "visibility": "exterior",
     "hint": "看玻璃幕墙龙骨与板块、窗框安装、外墙涂料/面砖"},
    {"id": "mep", "name": "机电安装（水/电/暖通）", "weight": 13,
     "visibility": "interior",
     "hint": "看立管、桥架、风管、屋面设备基础、机房"},
    {"id": "finishing", "name": "室内装修", "weight": 6,
     "visibility": "interior",
     "hint": "看隔墙、吊顶、地面找平、公共区域装修"},
    {"id": "landscape", "name": "室外总平与景观", "weight": 2,
     "visibility": "exterior",
     "hint": "看道路硬化、绿化、围挡拆除、场地清理"},
]


def total_weight() -> int:
    return sum(s["weight"] for s in STAGES)


def _count_factor(item: dict, cfg: dict) -> tuple[float | None, dict]:
    """
    计数型工序的折算：优先用"已封顶层数 / 总层数"，没有计数再退回状态四档。

    关键取舍：不用模型给的 count.done（它取的是各塔最小值，而最慢那栋的
    计数往往不稳定——实测 11→11→9，不升反降），改为从 count_detail
    自己算，并取【最高塔】的封顶层数作为进度指示。
    理由：最高塔的楼层计数在各期照片间最稳定（实测 16→18→20，每月 +2 层）。

    同时输出最高/最低/平均三个口径，便于发现"各塔进度不均"。
    """
    detail = [d for d in (item.get("count_detail") or []) if isinstance(d, dict)]
    per_tower = [float(d["done"]) for d in detail
                 if isinstance(d.get("done"), (int, float))]
    cnt = item.get("count") or {}
    model_done = cnt.get("done")
    model_total = cnt.get("total")
    model_inc = cnt.get("inc")

    if per_tower:
        done_max = max(per_tower)
        done_min = min(per_tower)
        done_avg = sum(per_tower) / len(per_tower)
        total = cfg.get("count_total") or model_total or done_max
        factor = max(0.0, min(1.0, done_max / float(total))) if total else None
        return factor, {
            "factor_source": "count",
            "count_done": int(done_max),
            "count_min": int(done_min),
            "count_avg": round(done_avg, 1),
            "count_total": total,
            "count_inc": model_inc,
            "count_is_estimated": cfg.get("count_total") is None,
        }

    if isinstance(model_done, (int, float)) and isinstance(model_total, (int, float)) \
            and model_total > 0:
        # 配置里的总层数优先于模型估计值（模型估的总层数误差很大）
        total = cfg.get("count_total") or model_total
        factor = max(0.0, min(1.0, model_done / float(total)))
        return factor, {"factor_source": "count", "count_done": model_done,
                        "count_min": model_done, "count_avg": model_done,
                        "count_total": total, "count_inc": model_inc,
                        "count_is_estimated": cfg.get("count_total") is None}

    status = str(item.get("status", "unclear")).lower()
    factor = STATUS_FACTOR.get(status)
    return factor, ({"factor_source": "status"} if factor is not None else {})


def compute_progress(vlm_stages: list[dict]) -> dict:
    """
    把模型给出的工序判定折算成加权进度。
    模型只负责"看到什么"，算术由代码做——不让模型直接报百分比。

    同时给出两个数，避免"分母变了进度就跳"的错觉：
      progress_pct   — 只在可判断的工序里算（乐观口径）
      progress_floor — 分母用全部权重，看不清按 0 计（保守口径）

    计数型工序（如主体结构）按"已封顶层数/总层数"折算，比四档状态有分辨率。
    """
    by_id = {s["id"]: s for s in STAGES}
    counted = 0.0
    total_w = 0.0
    detail = []
    for item in vlm_stages:
        sid = item.get("id")
        cfg = by_id.get(sid)
        if not cfg:
            continue
        status = str(item.get("status", "unclear")).lower()
        if cfg.get("metric") == "count":
            factor, extra = _count_factor(item, cfg)
        else:
            factor = STATUS_FACTOR.get(status)
            extra = {"factor_source": "status"} if factor is not None else {}

        base = {**item, "name": cfg["name"], "weight": cfg["weight"],
                "visibility": cfg["visibility"], **extra}
        if factor is None:
            detail.append({**base, "counted": False})
            continue
        counted += cfg["weight"] * factor
        total_w += cfg["weight"]
        detail.append({**base, "factor": round(factor, 4), "counted": True})

    tw = total_weight()
    return {
        "progress_pct": round(counted / total_w * 100.0, 1) if total_w else 0.0,
        "progress_floor": round(counted / tw * 100.0, 1) if tw else 0.0,
        "counted_weight": round(total_w, 1),
        "total_weight": tw,
        "stages": detail,
    }
