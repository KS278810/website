"""LP の favicon を作る。

写真をそのまま縮めると 16px では何も判別できないので、イニシャルの
モノグラムにする。配色は LP の --bg / --azure / --amber に合わせる。
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

FIG = Path(r"D:\_claude\03_HP作成\260801_MyWeb_Motion\figures")
BG = (10, 11, 16)
AZURE = (90, 168, 255)
AMBER = (232, 148, 74)


def font(size):
    for name in ("YuGothB.ttc", "arialbd.ttf", "seguisb.ttf"):
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                pass
    return ImageFont.load_default()


S = 512
img = Image.new("RGBA", (S, S), BG + (255,))
d = ImageDraw.Draw(img)

# 角丸の下地 + amber のアクセントライン
d.rounded_rectangle([28, 28, S - 28, S - 28], radius=96, fill=(18, 19, 28))
d.rounded_rectangle([28, 28, S - 28, S - 28], radius=96, outline=(56, 62, 82), width=6)

f = font(230)
text = "KS"
bbox = d.textbbox((0, 0), text, font=f)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
d.text(((S - tw) / 2 - bbox[0], (S - th) / 2 - bbox[1] - 14), text, font=f, fill=AZURE + (255,))
d.line([(S / 2 - 92, S - 132), (S / 2 + 92, S - 132)], fill=AMBER + (255,), width=14)

img.resize((256, 256), Image.LANCZOS).save(FIG / "favicon-256.png", "PNG", optimize=True)
img.save(FIG / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.resize((180, 180), Image.LANCZOS).save(FIG / "apple-touch-icon.png", "PNG", optimize=True)

for n in ("favicon.ico", "favicon-256.png", "apple-touch-icon.png"):
    print(f"  {n:<24} {(FIG / n).stat().st_size/1024:6.1f} KB")
