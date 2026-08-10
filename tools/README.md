# tools/ — このサイトの資産を作り直すスクリプト

`figures/` と `fonts/` の中身は、ここのスクリプトが生成したものです。
手で置き換えず、対応するスクリプトを回し直してください。

実行には Python と Pillow が要ります。

```powershell
# 例（任意の Python 環境で）
python tools\make_webp.py
```

| スクリプト | 作るもの | 回すとき |
|---|---|---|
| `fetch_fonts.py` | `fonts/`（Google Fonts 5書体を自己ホスト用に取得） | 書体やウェイトを変えたとき |
| `fetch_unsplash.py` | `figures/kaggle-thumb.*` `sandbox-thumb.*` | 写真を差し替えるとき |
| `make_webp.py` | `figures/*.webp` | **PNG を追加・差し替えたら必ず** |
| `make_ogp_lp.py` | `figures/ogp.png`（SNS共有カード 1200x630） | 肩書や実績の数字を変えたとき |
| `make_favicon_lp.py` | `figures/favicon.ico` `favicon-256.png` `apple-touch-icon.png` | ブランドの見た目を変えたとき |

## 気をつけること

- **外部からリソースを読まない**方針です。フォントも写真もこのリポジトリに
  取り込んであります。CDN の URL を直接書かないでください
  （`<a href>` で外部サイトへリンクするのは問題ありません）。
- 画像を差し替えたら `make_webp.py` を回し、HTML 側の `<picture>` /
  `image-set()` の両方が新しいファイルを指しているか確認してください。
- `<img>` の `width` / `height` は実寸を書いてください。以前 `trag-logo` で
  実寸と違う値が入っており、確保される領域の縦横比がずれていました。
