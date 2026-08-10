"""ポートフォリオ(LP)の OGP 画像を作る。

既存の本人写真は 400x400 しかなく、og:image の推奨(1200x630)に足りない。
LP の配色(--bg #0A0B10 / --amber #E8944A / --azure #5AA8FF)に合わせた
カードを組み、写真を円形に切り抜いて配置する。
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.stdout.reconfigure(encoding="utf-8")

W, H = 1200, 630
BG = (10, 11, 16)
INK = (236, 237, 243)
MUTED = (151, 153, 176)
AMBER = (232, 148, 74)
AZURE = (90, 168, 255)

FIG = Path(r"D:\_claude\03_HP作成\260801_MyWeb_Motion\figures")
OUT = FIG / "ogp.png"


def font(size, bold=False):
    for name in (("YuGothB.ttc", "YuGothM.ttc") if bold else ("YuGothM.ttc", "YuGothR.ttc")):
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                pass
    return ImageFont.load_default()


canvas = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(canvas)

# 右側に amber→azure のごく淡い光。LP の Hero と同じ雰囲気にする
glow = Image.new("RGB", (W, H), BG)
gd = ImageDraw.Draw(glow)
gd.ellipse([W - 560, -160, W + 160, 420], fill=(38, 26, 18))
gd.ellipse([W - 360, 240, W + 260, 820], fill=(14, 26, 46))
canvas = Image.blend(canvas, glow.filter(ImageFilter.GaussianBlur(120)), 0.9)
draw = ImageDraw.Draw(canvas)

# 本人写真を円形に
photo = Image.open(FIG / "KoheiShintani.jpg").convert("RGB").resize((300, 300), Image.LANCZOS)
mask = Image.new("L", (300, 300), 0)
ImageDraw.Draw(mask).ellipse([0, 0, 299, 299], fill=255)
px, py = 96, (H - 300) // 2
ring = Image.new("RGB", (W, H), BG)
ImageDraw.Draw(ring).ellipse([px - 6, py - 6, px + 305, py + 305], fill=(60, 66, 86))
canvas.paste(ring.crop((px - 6, py - 6, px + 306, py + 306)), (px - 6, py - 6),
             Image.new("L", (312, 312), 0).point(lambda _: 0))
draw.ellipse([px - 5, py - 5, px + 304, py + 304], outline=(70, 78, 100), width=3)
canvas.paste(photo, (px, py), mask)

tx = px + 300 + 70
draw.text((tx, 214), "Kohei Shintani, Ph.D.", font=font(56, True), fill=INK)
draw.text((tx, 292), "AI Researcher & Vehicle Engineer", font=font(30), fill=AZURE)
draw.text((tx, 344), "機械学習と自動車工学をつなぐ", font=font(26), fill=MUTED)

draw.line([(tx, 400), (tx + 300, 400)], fill=AMBER, width=3)
draw.text((tx, 424), "60 publications · 11 awards · 9 patents",
          font=font(24), fill=MUTED)

canvas.save(OUT, "PNG", optimize=True)
print(f"{OUT.name}  {W}x{H}  {OUT.stat().st_size/1024:.0f} KB")
