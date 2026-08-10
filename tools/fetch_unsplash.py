"""hobbies.html が外部から読んでいた Unsplash 画像をローカルへ取り込む。

CSS の background-image で参照していたため、リンク監査(<a href>)から漏れていた。
フォントを自己ホスト化しても、ここが残ると外部依存は消えない。
Unsplash License はダウンロードと再配布を許可している。
"""
import sys
import urllib.request
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

FIG = Path(r"D:\_claude\03_HP作成\260801_MyWeb_Motion\figures")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"

IMAGES = {
    "kaggle-thumb": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=800",
    "sandbox-thumb": "https://images.unsplash.com/photo-1455390582262-044cdead277a?auto=format&fit=crop&q=80&w=800",
}

for name, url in IMAGES.items():
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    jpg = FIG / f"{name}.jpg"
    jpg.write_bytes(data)
    im = Image.open(jpg).convert("RGB")
    webp = FIG / f"{name}.webp"
    im.save(webp, "WEBP", quality=82, method=6)
    print(f"  {name:<16} {im.width}x{im.height}  "
          f"jpg {jpg.stat().st_size/1024:5.0f} KB / webp {webp.stat().st_size/1024:5.0f} KB")
