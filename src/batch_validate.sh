#!/usr/bin/env bash
# 批量验证：同一项目不同时间段的照片对，逐组跑 CV 差分 + 视觉模型
set -u
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe"
R="baseline-terrsoshi/Resources"

# 组号:前期日期:后期日期
SETS="
3:2020-07-25:2020-08-14
4:2020-08-17:2020-09-07
5:2020-09-09:2020-10-01
"

for item in $SETS; do
  n="${item%%:*}"; rest="${item#*:}"
  d1="${rest%%:*}"; d2="${rest#*:}"
  echo ""
  echo "==================== TNB_$n  ($d1 -> $d2) ===================="

  $PY src/run_compare.py --before "$R/TNB_${n}_1.jpg" --after "$R/TNB_${n}_2.jpg" \
      --out "output/TNB_$n" --tag "tnb$n" 2>&1 | grep -E "内点率|重叠区|平均 SSIM|差异区域|耗时|失败"

  $PY src/vlm_report.py --before "$R/TNB_${n}_1.jpg" --after "$R/TNB_${n}_2.jpg" \
      --after-aligned "output/TNB_$n/tnb${n}_02_after_aligned.png" \
      --diff "output/TNB_$n/tnb${n}_07_change_grid.png" \
      --out "output/TNB_${n}_vlm" --before-date "$d1" --after-date "$d2" \
      --samples 3 2>&1 \
      | grep -E "实际生效|完成\]|结果\]|失败|Error"
done

echo ""
echo "全部完成"
