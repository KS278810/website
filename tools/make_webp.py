"""LP の重い PNG を WebP に変換する。

showcase 画像は 890KB / 433KB あり、Hobbies を開いた時点で両方読み込む。
WebP なら見た目を保ったまま 1/3 前後になる。PNG も残し、<picture> で
フォールバックさせるので、WebP 非対応環境でも表示は壊れない。
"""
import sys
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

FIG = Path(r"D:\_claude\03_HP作成\260801_MyWeb_Motion\figures")
TARGETS = ["surrobot-showcase.png", "explorebot-showcase.png", "ogp.png",
           "showcase1.png", "trag-logo.png"]

total_before = total_after = 0
for name in TARGETS:
    src = FIG / name
    if not src.exists():
        print(f"  skip (無し): {name}")
        continue
    im = Image.open(src)
    dst = src.with_suffix(".webp")
    # quality=82 は写真・スクリーンショットで劣化がほぼ見えない実用域
    im.save(dst, "WEBP", quality=82, method=6)
    b, a = src.stat().st_size, dst.stat().st_size
    total_before += b
    total_after += a
    print(f"  {name:<28} {b/1024:7.0f} KB -> {a/1024:7.0f} KB  ({a/b:.0%})")

print(f"\n  合計 {total_before/1024:.0f} KB -> {total_after/1024:.0f} KB "
      f"({(total_before-total_after)/1024:.0f} KB 削減)")
