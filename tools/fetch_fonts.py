"""Google Fonts を自己ホスト用にダウンロードする。

LP だけが外部(fonts.googleapis.com / fonts.gstatic.com)に依存していた。
ツール側は外部参照ゼロなので、そこだけ非対称だった。

やること:
  1. 実際に使っている @import URL の CSS を woff2 前提の UA で取得
  2. CSS 内の gstatic URL をすべて落とす
  3. URL をローカルパスに書き換えた CSS を書き出す

Noto Sans JP は unicode-range で細かく分割配信されている。全部落としても
ブラウザは「そのページに出てくる文字が含まれるファイル」しか取りに行かない
ので、転送量は増えない(リポジトリ容量だけ増える)。
"""
import re
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(r"D:\_claude\03_HP作成\260801_MyWeb_Motion\fonts")
CSS_OUT = OUT_DIR / "fonts.css"

# LP が読み込んでいる 2 本の CSS
SOURCES = [
    ("main", "https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;0,600;1,400;1,500"
             "&family=Noto+Sans+JP:wght@400;500;700&family=Space+Grotesk:wght@400;500;700"
             "&family=JetBrains+Mono:wght@400;500;700&display=swap"),
    ("inter", "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"),
]

# woff2 を返させるための UA(古い UA だと ttf になる)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read() if binary else r.read().decode("utf-8")


OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "files").mkdir(exist_ok=True)

merged = ["/* Google Fonts をローカルへ取り込んだもの。fetch_fonts.py が生成。\n"
          "   直接編集せず、スクリプトを回し直すこと。 */\n"]
seen = {}
total = 0

for label, url in SOURCES:
    css = get(url)
    print(f"[{label}] CSS {len(css):,} bytes")
    for m in re.finditer(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css):
        src = m.group(1)
        if src in seen:
            continue
        name = src.rsplit("/", 1)[-1].split("?")[0]
        # 同名衝突を避けるため family 名を前置
        fam = re.search(r"/s/([a-z0-9]+)/", src)
        fname = f"{fam.group(1) if fam else 'font'}-{name}"
        data = get(src, binary=True)
        (OUT_DIR / "files" / fname).write_bytes(data)
        seen[src] = f"./files/{fname}"
        total += len(data)
    for src, local in seen.items():
        css = css.replace(src, local)
    merged.append(css)

CSS_OUT.write_text("\n".join(merged), encoding="utf-8")
print(f"\nフォントファイル {len(seen)} 個 / {total/1024/1024:.2f} MB")
print(f"CSS: {CSS_OUT}  ({CSS_OUT.stat().st_size/1024:.0f} KB)")
