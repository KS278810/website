# T-regressor ブラウザ版

CSVをアップロードして回帰モデルを自動学習・予測する T-regressor を、**インストール不要・ブラウザだけ**で動かす版です。
学習も予測も **Pyodide (WebAssembly) 上で利用者の端末で実行** され、データやモデルはサーバーへ送信されません。

デスクトップ版(Tauri/exe)と同じ Python ロジック（`train_bridge.py` / `_light.py` /
`predict_template.py`）をベースにしています。`.treg` 書き出し部分のみ、Web版JS予測エンジンが
対応する全モデル種別(linear/poly-Ridge/lgbm/rf/xt/gp/mlp/blend)向けに拡張されており
（デスクトップ版のネイティブC++予測エンジンは現状4種別のみ対応）、学習ロジック本体は
デスクトップ版と同一です。詳細は[プロジェクトルートのREADME](../README.md#プロジェクト構成)参照。

## GUIはexe版と共通（2026-07-05〜）

以前は `web/index.html`・`web/offline.html` を exe版(`frontend/index.html`)とは別にHTML/CSS/JSを
手書きしており、見た目・機能がexe版より簡素だった。現在は **`frontend/index.html` が唯一のフロントエンド
ソース**で、exeもWebもここから作られる:

- `frontend/index.html` に `IS_TAURI` 分岐の `Platform` 抽象化層を実装。バックエンド呼び出し
  (Tauri invoke/event ↔ Pyodide直接呼び出し)だけが分岐し、UI・見た目・演出ロジックは完全に共通。
- `web/build_frontend.mjs` が `frontend/index.html` を読み込み、`reference/` → `assets/` の
  パス置換など配布形態ごとに機械的に決まる差分だけを適用して `web/index.html`・`web/offline.html` を生成する。

**`frontend/index.html` を変更したら、必ず以下を実行して Web版を最新化すること:**

```bash
cd web
node build_frontend.mjs
```

Web版とexe版でUIの見た目・挙動が食い違ってきたら、大抵は「Web版側を個別に直してしまった」のが原因。
直すべきは常に `frontend/index.html` 側 → `build_frontend.mjs` を再実行、が正しい順序。

Web版だけに存在する機能差(Platform capabilities で吸収):
- CPU並列数設定: ブラウザは単一スレッド実行のため無効化表示
- CANCEL(学習中断): ブラウザ版は非対応(WASM実行中の中断不可)のためボタン非表示
- サンプルCSV: exe版は保存ダイアログ→再ドロップの2手順、Web版はワンクリックで直接読み込み
- 画面サイズ: exe版のネイティブウィンドウ(1100x720)と同じ縦横比を保ったまま中央表示する。
  DOM自体は常に実寸1100x720で描画し、ウィンドウに収まる倍率をJSで1つだけ計算して
  CSSの`zoom`で一括拡縮する(`html.web-shell` / `body.boxed`、IS_TAURI 判定で分岐)。
  widthとheightを別々に`min()`で決める方式は縦横比が崩れて内部の円形要素(起動時の
  リングや読み込みアニメーションの縁)が楕円に見える不具合があったため、単一倍率の
  `zoom`方式に変更した(2026-07-05)。ウィンドウが十分広ければ最大1.25倍まで拡大表示する。
  Chromium系(Edge/Chrome)専用のCSSプロパティのため、Firefox/Safariでは等倍表示になる
  (未対応ブラウザでも歪みは発生しない。単に拡縮されないだけ)。
- ロボット表示(`.char-box`、220x220の正円)が楕円に潰れて見える別バグも修正(2026-07-05)。
  原因はズーム方式とは無関係で、`.char-panel`(縦方向flex)内でチャット文が長くなると
  `.chat-bubble`が伸び、`flex-shrink:0`の無かった`.char-box`が高さだけ圧縮されていた。
  `.char-box`に`flex-shrink:0`を付け、`.chat-bubble`側にも`flex-shrink:0`と
  `max-height:150px; overflow-y:auto`を付けて、どちらも潰れず長文はスクロールにした。

## アクセス解析(オプトイン・デフォルト無効)

「ゼロ外部通信」というWeb版の売り(学習・予測データは一切送信されない)を崩さないよう、
ページ閲覧数だけを見る軽量なアクセス解析を**オプトインかつデフォルト無効**で用意した。
2種類を用意しており、どちらか一方でも両方でも(あるいはどちらも使わなくても)よい。

- **選択肢1: [GoatCounter](https://www.goatcounter.com/)** — 非商用利用は無料でページビュー数
  無制限。Cookie不使用、IPは日次ソルト付きハッシュ化後に破棄、スクリプトは約3.5KBと軽量。
  有効化するには `frontend/index.html`(および `web/index.html` / `web/offline.html`)の
  JS内 `const GOATCOUNTER_CODE = '';` に、GoatCounterで作成した無料アカウントの
  サブドメインコード(例: `tregressor`)を入れるだけ。
- **選択肢2: [Cloudflare Web Analytics](https://www.cloudflare.com/web-analytics/)** —
  完全無料・件数上限なし。サイトのDNSをCloudflareに移管する必要はなく、Cloudflareの
  無料アカウントでサイトを追加すると発行される Beacon トークンを`const CF_BEACON_TOKEN = '';`
  に設定するだけで有効になる。Cookie不使用でダッシュボードでページビュー・参照元・
  国・ブラウザ内訳などが見られる。
  Google Analytics は無料だがGoogleにデータが渡る点が「サーバー送信なし」の訴求と
  矛盾するため両者とも見送り、Plausibleは自己ホストにコストがかかるため見送った。
- **`offline.html`(ダブルクリック版)ではどちらの値も空のままにしておくこと**を推奨。
  オフライン配布・LAN内配布・エアギャップ環境での利用を想定した配布物であり、
  外部スクリプトを読みに行く処理を仕込むのはこの用途に矛盾するため。
- 計測対象はページ閲覧数のみで、CSV・モデル・学習結果などユーザーデータは一切含まれない
  (そもそも両サービスへ送るのはURL/リファラ/画面サイズ程度)。
- 空文字のままならどちらのスクリプトも一切読み込まれない(exe版でも常に無効)。

「学習済モデルのDL」は **単体HTML** (`predict_template.html` の `__TREG_BASE64__` プレースホルダを
学習済み `.treg` のBase64で置き換えたもの)を出力する(2026-07〜。旧: `predict_native.exe` に
`.treg` を追記する方式だったが、Windows MOTW/SmartScreenで未署名exeがブロックされる問題が
あったためHTML方式に置換した)。HTTP版・オフライン版とも `predict_template.html` を
テンプレートとして使う(HTTP版は `fetch("./predict_template.html")`、オフライン版は
`offline-embed.js` に文字列同梱)。書き出されるHTMLはダブルクリックで開くだけで使え、
Windows以外(Mac/Linuxのブラウザ)でも動作する。予測ロジックの詳細・全モデル種別対応状況は
[`js_predict_poc/README.md`](js_predict_poc/README.md) を参照。

**2つの配布形態**があります。用途に応じて使い分けてください。

| | `index.html`（HTTP版） | `offline.html`（ダブルクリック版） |
|---|---|---|
| 開き方 | `node serve.mjs` 等でHTTP配信して開く | フォルダごと渡してファイルを**直接ダブルクリック** |
| サーバー | 静的ホスティング/ローカルサーバーが必要 | 一切不要 |
| 外部通信 | なし（vendor同梱・実機確認済み） | なし（実機確認済み） |
| 用途 | GitHub Pages / HF Space 等での公開共有 | USB/LAN共有・オフラインPCへの配布 |

どちらも計算ロジックは同じ `py/` を使い、結果はビット単位で同一です。

## 構成

```
frontend/index.html    ★唯一のフロントエンドソース(exe版と共通)。直接編集するのはここだけ。

web/
├─ build_frontend.mjs   frontend/index.html → index.html・offline.html を生成するビルドスクリプト
├─ index.html          UI（HTTP版・単一ページ・依存なし）※自動生成・直接編集しない
├─ treg-engine.js       HTTP版の Pyodide アダプタ
├─ offline.html         UI（ダブルクリック版）※自動生成・直接編集しない
├─ offline-engine.js     ダブルクリック版の Pyodide アダプタ(Import Maps+fetchオーバーライドで通信ゼロ化)
├─ offline-embed.js      ダブルクリック版が使う埋め込みデータ(約59MB・自動生成・要ビルド)
├─ build_offline.mjs     offline-embed.js を生成するビルドスクリプト
├─ predict_template.html 「学習済モデルのDL」が組み立てる単体HTML配布物のベーステンプレート
├─ py/                 デスクトップ版から流用する Python（無改変）
│  ├─ train_bridge.py
│  ├─ _light.py
│  └─ predict_template.py
├─ js_predict_poc/     予測ロジックのJS実装＋数値パリティテスト一式(詳細は同ディレクトリのREADME参照)
├─ vendor/pyodide/     Pyodide 本体 + 依存ライブラリを実体同梱（約43MB・外部CDN不使用）
├─ assets/             ロゴ・マスコット（frontend/reference/ から build_frontend.mjs が自動補完）
├─ sample_data.csv     お試し用サンプル
└─ serve.mjs           ローカル確認用の最小サーバー（HTTP版用）
```

`vendor/pyodide/` の中身（バージョン更新など）を変更したら、必ず次を実行して埋め込みデータを再生成すること:

```bash
cd web
node build_offline.mjs
```

## ローカルで確認する

**HTTP版:**
```bash
cd web
node serve.mjs           # → http://localhost:8000
```
`file://` で直接開くと ES モジュールの読込がブロックされるため、必ず HTTP 経由で開いてください。

**ダブルクリック版:**
```
web/offline.html をエクスプローラーで直接ダブルクリックするだけ。
```
`web/` フォルダ一式（`offline.html` / `offline-embed.js` / `offline-engine.js` / `vendor/pyodide/` / `py/` / `assets/`）を
まとめてコピーすれば、サーバーなし・ネットワークなしのPCでもそのまま動きます。

### ダブルクリック版の技術的背景（file:// で動く理由）

`file://` は本来 ES モジュールの `import` も `fetch()` も CORS でブロックしますが、以下の回避策を実機検証の上で採用しています:
- Pyodide本体・依存ライブラリ・Pythonソース・サンプルCSV は全て `offline-embed.js` に base64/テキストとして埋め込み、`fetch` は使わない
- `window.fetch` を「実ネットワークに出ず埋め込みデータから `Response` を合成する」版に差し替え、wasm/依存wheelの読込を解決
- Pyodide内部が `pyodide.asm.js` だけを動的 `import()` で読みに行くため、**Import Maps** でその指定子(実行時に解決される絶対URL)を `data:` URL にリマップして回避
- UI本体もESモジュールを使わない通常の `<script>` に統一

実機（Edge）で「`file://`／`data:`／`blob:` 以外への通信が0件」であることを確認済みです。

## 公開（ホスティング・HTTP版）

計算ライブラリ(numpy/pandas/scipy/lightgbm 等)と Pyodide 本体は `vendor/pyodide/` に**実体として同梱**しており、
**外部 CDN には一切アクセスしません**（実機で外部ドメインへの通信ゼロを確認済み）。
社内プロキシ等で外部 CDN が遅い/塞がれている環境でも動作します。オフライン環境（LAN内配布、USB配布等）でも動作可能です。
本アプリは Python スレッドを使わないため **COOP/COEP ヘッダも不要**で、静的ホスティングにそのまま置けます。

### GitHub Pages
1. `web/` の中身（`vendor/` を含む）をリポジトリ（例 `docs/` フォルダ）に置く
2. Settings → Pages で公開ブランチ/フォルダを指定
3. 発行された URL を共有

### Hugging Face Space（Static）
1. SDK = **Static** で Space を作成
2. `index.html` がルートに来るよう `web/` の中身（`vendor/` を含む）をアップロード
3. Space の URL を共有

いずれも `py/` `vendor/` `assets/` `sample_data.csv` `predict_template.html` を同じ階層に含めること
(`predict_template.html` が無いと「学習済モデルのDL」が失敗する)。
`vendor/` は約43MBあるため、リポジトリサイズやアップロード時間に注意（Git LFS推奨のケースもある）。

## 動作特性（利用者に伝えるべき点）

- 初回アクセス時にページと同梱の計算ライブラリ(約43〜56MB)を読み込む。外部通信は発生しないため、
  回線が遅い環境でも「配布元サーバー/ディスクの速度」だけに依存する
  （実測: HTTP版のエンジン初期化 約5秒、ダブルクリック版はページの読込込みで約6秒）。
- 一度ブラウザにキャッシュされれば2回目以降はさらに高速（ダブルクリック版はキャッシュの効き方がブラウザ設定に依存する点に注意）。
- 計算は WebAssembly で行うためデスクトップ版より遅い。目安（数百行データ）:
  - お急ぎモード … 数十秒
  - じっくりモード … 数分（探索＋6モデル合成のため）
- 数十〜数万行の表形式データ向け。巨大データには不向き。
- ダブルクリック版は動的 import() のリマップ(Import Maps)という比較的新しいブラウザ機能を使っている。
  Chromium系(Edge/Chrome)での動作は実機確認済み。Firefox/Safari等での動作は未検証。

## モデルの保存と再利用

UIがexe版と共通化されたため、保存はダッシュボードの **「学習済モデルのDL」** ボタン1つのみ
(exe版と同じ導線)。クリックすると `predict_template.html` に学習済み `.treg` をBase64埋め込みした
**単体HTML**(`model_xxx.html`)がダウンロードされる。ダブルクリックで開くだけで使え、
Windows/Mac/Linuxいずれのブラウザでも動作する(Python不要)。

以前(〜2026-06)は `predict_native.exe` に `.treg` を追記する単体exe方式だったが、ブラウザ経由で
ダウンロードしたファイルに Windows が付与する **Mark of the Web** により、未署名exeの実行が
SmartScreen等でブロックされる問題があった。HTMLファイルは実行ファイルではないためこの制約を
受けず、2026-07にHTML方式へ全面的に置き換えた。対応モデル種別の詳細・数値検証結果は
[`js_predict_poc/README.md`](js_predict_poc/README.md) を参照。

## 検証（js_predict_poc/）

予測ロジック(JS版 `predict-core.js` / `predict_template.html` / `.treg` 書き出し)の数値検証一式。
C++参照実装(`native_predictor/predict_native_v2.cpp`)・Python版(`predict_template.py`)との
数値一致をテストする。詳細・実行方法は [`js_predict_poc/README.md`](js_predict_poc/README.md) を参照。

```bash
cd web/js_predict_poc
npm install
npm test
```
