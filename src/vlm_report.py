"""
调用视觉大模型（DeepSeek V4.1 Flash / deepseek-flash）解读两期照片，输出进度报告。

用法：
    # 首次：把 key 填进 .env
    python src/vlm_report.py --before 前期.jpg --after 后期.jpg --out output/A

    # 可选：把 CV 差分结果一起喂进去（推荐，能明显提升定位准确率）
    python src/vlm_report.py --before a.jpg --after b.jpg --out output/A \
        --diff output/A/xxx_07_change_grid.png

    # 只检查提示词和图片负载，不真正调用（不花钱）
    python src/vlm_report.py --before a.jpg --after b.jpg --out output/A --dry-run

设计原则：
    模型只回答"有没有看到某道工序"以及"证据在哪"，
    百分比由 stages.py 用权重算，避免模型随口瞎报数字。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time

import cv2
from openai import OpenAI, APIConnectionError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stages import STAGES, compute_progress, total_weight  # noqa: E402
from aggregate import aggregate  # noqa: E402
from face_crop import make_face_crops  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path: str = os.path.join(ROOT, ".env")) -> None:
    """极简 .env 解析，不引入额外依赖。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def encode_image(path: str, max_edge: int = 1536, quality: int = 88) -> str:
    """读图 -> 等比缩放 -> JPEG -> base64 data URL。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不到图片: {path}")
    h, w = img.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _days_between(d1: str, d2: str) -> int | None:
    """两个日期字符串相差的天数；解析不了返回 None。"""
    from datetime import datetime
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return (datetime.strptime(d2, fmt) - datetime.strptime(d1, fmt)).days
        except (ValueError, TypeError):
            continue
    return None


def build_prompt(before_date: str, after_date: str, has_diff: bool,
                 aligned: bool = True, has_face: bool = False) -> str:
    ext = [s for s in STAGES if s.get("visibility") == "exterior"]
    hid = [s for s in STAGES if s.get("visibility") != "exterior"]

    def fmt(items):
        lines = []
        for s in items:
            tag = ("  【计数型：必须数出层数】" if s.get("metric") == "count" else "")
            lines.append(f'- id="{s["id"]}"  {s["name"]}（权重{s["weight"]}）  '
                         f'观察要点：{s["hint"]}{tag}')
        return "\n".join(lines)

    stage_lines = (
        "【A. 外露工序 —— 从外部/高位视角通常可以直接看到】\n"
        + fmt(ext)
        + "\n\n【B. 隐蔽工序 —— 多被楼板、脚手架、防护网遮挡，外部视角往往看不到】\n"
        + fmt(hid))

    after_desc = (
        f"【图2】后期现场照片（{after_date}），已由算法透视变换对齐到图1的视角，"
        f"可直接逐区域对比"
        if aligned else
        f"【图2】后期现场照片（{after_date}），拍摄视角可能与图1存在差异，"
        f"请先自行判断两张照片的对应关系再作比较")

    diff_desc = (
        "\n第三张图是算法生成的差分热力图：红色/橙色区块表示两个时期之间像素变化最大的位置。"
        "它只用来帮你定位“哪里变了”，不代表那里一定完成了某道工序，判断仍要以照片本身为准。\n"
        if has_diff else "")

    face_desc = (
        "\n【图4】【图5】是把图1、图2 中\"主体结构作业面\"区域按同一坐标裁剪并放大后的对比图。\n"
        "  远景里楼层线只有几个像素，放大后可到十几像素，**数楼层请以图4/图5 为准**。\n"
        "  绝对层数仍结合全图判断（因为裁剪范围不含地面基准），但 inc（新增层数）请用图4/图5 数。\n"
        if has_face else "")

    return f"""你是建筑工程施工进度评估专家。现在给你同一建筑项目在不同日期的现场照片。

【图1】前期现场照片（{before_date}）
{after_desc}
{diff_desc}
请完成两件事：
1. 判断【后期照片】中下列每道工序的完成状态；
2. 指出从前期到后期，画面里具体新增了什么。

工序清单：
{stage_lines}

status 只能取这四个值之一：
  done         — 该工序已明显完成（例如整栋楼外立面已完工）
  in_progress  — 正在施工中，能看到该工序的作业面/材料/半成品
  not_started  — 画面中明确看不到该工序的痕迹
  unclear      — 视线被遮挡、分辨率不足或不在画面内，无法判断

硬性要求（务必遵守）：
- 只依据画面可见证据，绝对不要推测或编造。
- 判断不了就写 unclear，这是允许的，不要为了凑数硬猜。
- 不要输出任何百分比数字，进度由程序计算。
- evidence 里必须写明画面位置和可见特征，例如"左侧塔楼第12层顶板已完成浇筑，其上方有钢管脚手架和未拆模板"。
- 判断 A 类外露工序时：如果画面里确实看不到该工序应有的完成痕迹（例如外墙通体还是裸露混凝土、
  屋面还是结构层），可以判 not_started，并在 evidence 里说明"看不到什么"作为依据。
- 判断 B 类隐蔽工序时：严禁写 not_started。被遮挡、看不见内部，一律写 unclear。
  不要因为"整体还在结构施工阶段，所以装修应该没开始"这类推测而下结论，那是推测不是观察。
- 如果两张图不是同一建筑或视角差异过大导致无法对应，在 risks 里明确指出。
- done 的判定门槛要严，必须"完成面完整可见"才算：
  · 土方开挖：只有基坑回填到位、场地平整、无明显开挖面时才可判 done；
    只要还能看到裸露土方、碎石垫层、边坡支护未拆，一律 in_progress。
  · 基础与地下室：只有所有塔楼出地面且地下结构无裸露施工面时才可判 done；
    只要还能看到筏板钢筋、柱头外露、模板未拆，一律 in_progress。
  · 判不准就写 unclear，不要往 done 上靠。done 是这四个状态里最容易犯错的。

关于计数型工序（主体结构）的额外要求，这一项最重要：
- 必须给出 count 字段，数出【后期照片】中各塔楼已封顶（即已浇筑楼板）的层数。
- 先按塔楼分别数，填进 count_detail；count.done 取各塔楼中的最大值（最高塔），
  count.total 是你估计的该建筑最终总层数。total 是估计值，必须在 evidence 里说明依据。
- 数楼层的方法：从地面以上第一层楼板开始，逐层往上数到最新浇筑的那一层。
  如果某一层判断不了是楼板还是模板，按"未封顶"处理，宁可少算一层。
- 同时必须给出【增量】，这一项比绝对层数更重要：
  · count_detail 里每栋塔楼都要填 inc = 图2 该塔已封顶层数 − 图1 该塔已封顶层数；
  · count.inc = 各塔 inc 中的最大值；
  · 只数"图1里没有、图2里新增"的楼层。两期基准不一致时以"同一位置对比"为准；
  · 如果图1里该塔顶部被遮挡、看不到基准层，inc 填 null，不要用绝对层数相减来硬凑。
- 如果楼层被遮挡到无法可靠计数，count 填 null，并把 status 写成 unclear。不许估个大概数字充数。
- status 仍然要填（这一点和 count 并存）：结构还在往上长就是 in_progress。

只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块，格式如下：
{{
  "stages": [
    {{"id": "工序id", "status": "done|in_progress|not_started|unclear",
      "evidence": "画面位置 + 可见特征", "confidence": 0.0,
      "count": {{"done": 0, "total": 0, "inc": 0}},
      "count_detail": [{{"part": "塔楼位置描述", "done": 0, "total": 0, "inc": 0}}]}}
  ],
  "changes": [
    {{"where": "画面位置", "what": "新增/变化的具体内容", "stage_id": "对应工序id"}}
  ],
  "risks": ["值得注意的问题，没有就给空数组"],
  "summary": "一句话概括本期进展"
}}

注意：只有计数型工序需要填 count 和 count_detail，其余工序这两个字段填 null 或省略即可。"""


def parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def render_report(vlm: dict, prog: dict, meta: dict,
                  agg: dict | None = None) -> str:
    rows = ["| 工序 | 状态 | 折算依据 | 权重 | 证据 | 置信度 |",
            "|---|---|---|---|---|---|"]
    label = {"done": "已完成", "in_progress": "进行中",
             "not_started": "未开始", "unclear": "看不清"}
    for s in prog["stages"]:
        raw = next((x for x in vlm.get("stages", [])
                    if x.get("id") == s.get("id")), {})
        if s.get("factor_source") == "count":
            basis = f"**{s['count_done']}/{s['count_total']} 层**"
        elif s.get("counted"):
            basis = f"状态折算 ×{s.get('factor')}"
        else:
            basis = "不计入"
        rows.append(
            f"| {s['name']} | {label.get(raw.get('status', 'unclear'), '—')} "
            f"| {basis} | {s['weight']} "
            f"| {str(raw.get('evidence', '')).replace('|', '/')} "
            f"| {raw.get('confidence', '')} |")

    cnt = next((s for s in prog["stages"] if s.get("factor_source") == "count"), None)
    count_line = ""
    if cnt:
        inc = cnt.get("count_inc")
        days = meta.get("days")
        rate_txt = ""
        if isinstance(inc, (int, float)) and days:
            rate_txt = (f"　本期新增 **+{inc:g} 层 / {days} 天**"
                        f"（约 {inc / days * 30:.1f} 层/月）")
        count_line = (f"\n**主体结构（最高塔）：{cnt['count_done']}/{cnt['count_total']} 层**"
                      f"{rate_txt}\n")

    changes = "\n".join(
        f"- **{c.get('where', '')}**：{c.get('what', '')}"
        for c in vlm.get("changes", [])) or "- 未识别出明确的新增内容"
    risks = "\n".join(f"- {r}" for r in vlm.get("risks", [])) or "- 无"

    agg = agg or {}
    samples = meta.get("samples", 1)
    agree = agg.get("agreement", {})
    if samples > 1:
        floors_key = next((k for k in agree if k.endswith(":floors")), None)
        sample_block = (f"**采样**：{samples} 次，状态多数投票、楼层取中位数\n"
                        + (f"**楼层数一致性**：{agree[floors_key]}\n"
                           if floors_key else ""))
    else:
        sample_block = "**采样**：1 次（未投票，结果可能不稳定，建议 --samples 3）\n"

    return f"""# 施工进度影像评估报告

**前期照片**：{meta['before']}（{meta['before_date']}）
**后期照片**：{meta['after']}（{meta['after_date']}）
**模型**：{meta['model']}　**生成时间**：{meta['time']}
{sample_block}
## 结论

**进度（保守口径）：{prog['progress_floor']}%**
分母固定为全部权重，看不清的工序按未完成计。这是可以做时间序列比较的口径。

进度（乐观口径）：{prog['progress_pct']}%（只算可判断的工序，分母会随可见性变化，仅作参考）

本期可判断权重 {prog['counted_weight']} / {prog['total_weight']}，
差额反映有多少工序被遮挡看不出来，差距越大说明这个机位的信息量越少。
{count_line}
{vlm.get('summary', '')}

## 逐项判断

{chr(10).join(rows)}

## 本期新增内容

{changes}

## 风险与提示

{risks}

---
> 本报告由视觉模型依据照片判断，进度为加权估算值，不能替代现场实测与监理签认。
"""


def call_model(client, model: str, content: list[dict], think: bool):
    """
    调用模型，返回 (response, 实际生效的参数说明)。

    关思考模式实测：网关支持 reasoning_effort="none"，也支持
    thinking={"type":"disabled"}；enable_thinking 会被静默忽略。
    这里做逐级降级，保证换网关/换模型也能跑。
    """
    base = dict(model=model, messages=[{"role": "user", "content": content}])
    opts: list[tuple[str, dict]] = []
    if not think:
        opts.append(('reasoning_effort="none"', {"reasoning_effort": "none"}))
        opts.append(('thinking=disabled', {"extra_body": {"thinking": {"type": "disabled"}}}))
    opts.append(("默认(带思考)", {}))

    last_err: Exception | None = None
    for label, opt in opts:
        for use_rf in (True, False):
            kw = {**base, **opt}
            if use_rf:
                kw["response_format"] = {"type": "json_object"}
            try:
                resp = client.chat.completions.create(**kw)
                note = label + ("+json_mode" if use_rf else "")
                return resp, note
            except Exception as e:                      # noqa: BLE001
                last_err = e
                if isinstance(e, APIConnectionError):
                    raise
                # 该参数组不被支持 -> 换下一组
                break
    raise last_err if last_err else RuntimeError("调用失败")


def main() -> None:
    ap = argparse.ArgumentParser(description="视觉模型解读施工照片")
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)
    ap.add_argument("--out", default="output/vlm")
    ap.add_argument("--diff", default=None, help="CV 差分图（可选，推荐）")
    ap.add_argument("--after-aligned", default=None,
                    help="对齐后的后期图（推荐传，来自 run_compare 的 02_after_aligned.png）")
    ap.add_argument("--before-date", default="前期")
    ap.add_argument("--after-date", default="后期")
    ap.add_argument("--max-edge", type=int, default=1536)
    ap.add_argument("--fast", action="store_true",
                    help="关闭思考模式（快约4倍、省约12倍，但判定稳定性明显下降）")
    ap.add_argument("--change-mask", default=None,
                    help="CV 差分掩膜（run_compare 的 09_change_mask.png），配合 --face-crop")
    ap.add_argument("--face-crop", action="store_true",
                    help="裁出作业面放大后一起送模型，提升楼层计数精度")
    ap.add_argument("--samples", type=int, default=1,
                    help="采样次数，>1 时对状态多数投票、楼层取中位数（推荐 3）")
    ap.add_argument("--from-json", default=None,
                    help="从已保存的 vlm_raw.json 重新渲染报告，不调用 API")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            d = json.load(f)
        prog = compute_progress(d["vlm"].get("stages", []))
        md = render_report(d["vlm"], prog, d["meta"],
                           {"agreement": d.get("agreement", {})})
        out = args.out or os.path.dirname(os.path.abspath(args.from_json))
        os.makedirs(out, exist_ok=True)
        rp = os.path.join(out, "report.md")
        with open(rp, "w", encoding="utf-8") as f:
            f.write(md)
        cnt = next((s for s in prog["stages"]
                    if s.get("factor_source") == "count"), None)
        cnt_txt = (f"  封顶 {cnt['count_done']}/{cnt['count_total']} 层"
                   if cnt else "")
        print(f"[重新渲染] 进度 {prog['progress_pct']}%"
              f" (下限 {prog['progress_floor']}%){cnt_txt}  ->  {rp}")
        return

    load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

    prompt = build_prompt(args.before_date, args.after_date, bool(args.diff),
                          aligned=bool(args.after_aligned),
                          has_face=bool(args.face_crop and args.change_mask))
    img2_path = args.after_aligned or args.after

    content: list[dict] = [{"type": "text", "text": prompt},
                           {"type": "text", "text": "【图1】前期照片："},
                           {"type": "image_url",
                            "image_url": {"url": encode_image(args.before, args.max_edge),
                                          "detail": "high"}},
                           {"type": "text", "text": "【图2】后期照片："},
                           {"type": "image_url",
                            "image_url": {"url": encode_image(img2_path, args.max_edge),
                                          "detail": "high"}}]
    if args.diff:
        content.append({"type": "text", "text": "【图3】差分热力图（红色=变化区域）："})
        content.append({"type": "image_url",
                        "image_url": {"url": encode_image(args.diff, args.max_edge),
                                      "detail": "high"}})

    face_paths = None
    if args.face_crop:
        mask = cv2.imread(args.change_mask, cv2.IMREAD_GRAYSCALE) if args.change_mask else None
        if mask is None:
            print("[提示] --face-crop 需要 --change-mask，已跳过裁剪")
        else:
            b_img = cv2.cvtColor(cv2.imread(args.before, cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2RGB)
            a_img = cv2.cvtColor(cv2.imread(img2_path, cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (b_img.shape[1], b_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            face_paths = make_face_crops(b_img, a_img, mask, args.out,
                                         prefix="", target_edge=1600)
            if face_paths:
                fb, fa, fbox = face_paths
                print(f"[裁剪] 作业面区域 {fbox}，已放大到 1600px")
                content.append({"type": "text", "text": "【图4】作业面放大（前期）："})
                content.append({"type": "image_url",
                                "image_url": {"url": encode_image(fb, 1600),
                                              "detail": "high"}})
                content.append({"type": "text", "text": "【图5】作业面放大（后期）："})
                content.append({"type": "image_url",
                                "image_url": {"url": encode_image(fa, 1600),
                                              "detail": "high"}})
            else:
                print("[提示] 没找到可裁的作业面，已跳过")

    payload_mb = sum(len(c["image_url"]["url"]) for c in content
                     if c["type"] == "image_url") / 1024 / 1024
    n_img = (3 if args.diff else 2) + (2 if args.face_crop and args.change_mask else 0)
    print(f"[准备] 模型={model}  图片={n_img} 张  "
          f"base64 负载≈{payload_mb:.1f}MB")

    if args.dry_run:
        print("[dry-run] 提示词预览：\n" + "-" * 60)
        print(prompt[:1200] + "\n...(略)")
        print("-" * 60)
        print("[dry-run] 未调用 API。")
        return

    if not api_key:
        print("\n[错误] .env 里的 DEEPSEEK_API_KEY 还是空的。")
        print(f"       请编辑 {os.path.join(ROOT, '.env')} 填入你的 key 后重试。")
        sys.exit(2)

    client = OpenAI(api_key=api_key, base_url=base_url)
    t0 = time.time()

    raw_samples: list[dict] = []
    total_in = total_out = 0
    used_note = ""
    for i in range(max(1, args.samples)):
        resp, used_note = call_model(client, model, content, think=not args.fast)
        text = resp.choices[0].message.content or ""
        raw_samples.append(parse_json(text))
        u = resp.usage
        total_in += getattr(u, "prompt_tokens", 0) or 0
        total_out += getattr(u, "completion_tokens", 0) or 0
        if args.samples > 1:
            print(f"[采样] 第 {i + 1}/{args.samples} 次完成  "
                  f"{time.time() - t0:.0f}s")
    print(f"[调用] 实际生效参数：{used_note}  采样 {len(raw_samples)} 次")

    vlm, agg = aggregate(raw_samples)
    prog = compute_progress(vlm.get("stages", []))
    os.makedirs(args.out, exist_ok=True)
    meta = {
        "before": os.path.basename(args.before),
        "after": os.path.basename(img2_path),
        "before_date": args.before_date, "after_date": args.after_date,
        "model": model,
        "samples": len(raw_samples),
        "days": _days_between(args.before_date, args.after_date),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(args.out, "vlm_raw.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "vlm": vlm, "progress": prog,
                   "agreement": agg.get("agreement", {})},
                  f, ensure_ascii=False, indent=2)
    if len(raw_samples) > 1:
        with open(os.path.join(args.out, "vlm_samples.json"), "w",
                  encoding="utf-8") as f:
            json.dump(raw_samples, f, ensure_ascii=False, indent=2)

    md = render_report(vlm, prog, meta, agg)
    rp = os.path.join(args.out, "report.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n[完成] 耗时 {time.time() - t0:.1f}s  "
          f"tokens 入/出 = {total_in}/{total_out}")
    cnt = next((s for s in prog["stages"] if s.get("factor_source") == "count"), None)
    cnt_txt = ""
    if cnt:
        inc = cnt.get("count_inc")
        days = meta.get("days")
        cnt_txt = f"  封顶 {cnt['count_done']}/{cnt['count_total']} 层"
        if isinstance(inc, (int, float)) and days:
            cnt_txt += f" (+{inc:g} 层/{days}天)"
    print(f"[结果] 进度 {prog['progress_pct']}% (保守下限 {prog['progress_floor']}%)"
          f"{cnt_txt}  (可判断权重 {prog['counted_weight']}/{prog['total_weight']})")
    print(f"[报告] {rp}")
    print(f"[原始] {os.path.join(args.out, 'vlm_raw.json')}")
    print("\n" + "-" * 60)
    print(vlm.get("summary", ""))


if __name__ == "__main__":
    main()
