# JS推論エンジン PoC(全候補モデル種別対応)

「学習済みモデルのDL」をexe(署名・MOTW問題あり)ではなく、Pyodideも使わない**単体JSファイル**
で提供できないかを検証するPoC。`.treg`(配布用の独自コンパクトバイナリ形式)を直接JSで
パースし、Python版(`predict_template.py`)と数値的に同一のロジックで予測する。

対応モデル種別(`.treg` type 0〜5、2026-07拡張): **linear(Ridge) / lgbm(LightGBM) /
gp(ARD-RBF) / mlp / linear_poly(poly-Ridge, type 4) / blend(アンサンブル, type 5)**の
6種別すべて。学習候補として存在する rf/xt は LightGBM の `boosting_type='rf'` モデルであり、
type 1 (lgbm) にエイリアスして書き出す(`average_output`ヘッダフラグを検出し、leaf値を
木の本数で事前割り算することでJS側の「全木の和」ロジックと数値的に一致させている)。
これにより **UIで表示されるベストモデルがどの種別であっても、そのまま`.treg`として書き出して
単体HTML配布できる**(以前はpoly-Ridge/rf/xt/blendが選ばれた場合、書き出し時に別の
モデルへ無言ですり替わっていた)。

C++参照実装(`native_predictor/predict_native_v2.cpp`)は2026-07にtype 4(linear_poly)/
type 5(blend)の読込・整合性検査(n_feat一致検査・LGBMノード範囲検査・blend再帰深さ制限)にも
対応済みで、`matrix/_manifest.json`にlinear_poly 3設定・blend 3設定(既存分含め計46設定)が
統合されており、全型をC++参照実装(`matrix_cpp_out/`)との機械的な突合せで検証している。
`matrix/_manifest_pyref.json`(Python独立実装`matrix_python_out/`との突合せ経路)は現状空で、
将来C++がまだ対応しない新しい`.treg`型が追加された場合のための予備の仕組みとして残してある。

## 結果

### 初回パリティテスト(run_parity.js、7行 x 4種別)

C++参照実装(`predict_native_v2.cpp`をLinux向けにg++でビルドしたもの)と同じ`.treg`・同じCSVで
予測し、4モデル種別すべてで出力を突き合わせた。

| モデル種別 | 最大絶対誤差 | 判定 |
|---|---|---|
| linear (Ridge) | 1.2×10⁻⁶ | PASS |
| lgbm (LightGBM) | 4.5×10⁻⁷ | PASS |
| gp (ARD-RBF) | 9.4×10⁻⁷ | PASS |
| mlp | 9.9×10⁻⁷ | PASS |

- **全体最大絶対誤差: 1.2×10⁻⁶**(float32ベースのモデル形式そのものの精度限界内。実用上は完全一致)
- **コードサイズ: 約10KB**(4種別すべて込み・依存ライブラリなし)
- **モデル読み込み: 約1〜2ms**
- **予測速度(lgbm、最も重い部類): 1行あたり約6μs**(10,000行で約60ms)

参考: 現行のPyodide埋め込み案(約57MB、初期化に数秒)と比べて、サイズ・起動速度とも
桁違いに軽い。

### 網羅的マトリクステスト(run_matrix_test.js、302行 x 46設定)

4モデル種別(linear/lgbm/gp/mlp) × y_transform 5パターン(none/log1p/yeo_johnson×3λ値、
λ=0.0とλ=2.0は`yeo_johnson_inv`の特殊分岐を踏む境界値)× round_output(true/false) = 40通りに
加え、2026-07-16にlinear_poly(type4)3設定・blend(type5)3設定を追加した**計46通り**の`.treg`を、
欠損値・極端値(±1e6, ±1e-6)・範囲外値・全NaN行・全ゼロ行を含む302行の`stress_test.csv`で検証。

```
検証設定数: 46 / 46
PASS: 46  FAIL: 0
総検証行数: 13,892件
✅ 全設定・全行で一致を確認
```

round_output=trueの設定・round_output=falseの設定とも、期待値に応じた相対混合閾値
(絶対1e-6下限、期待値に対し相対1e-4)で判定している。以前はround_output=trueを
「整数化されるため誤差ゼロのはず」として絶対1e-9固定にしていたが、linear_poly(多項式)に
極端な入力値(x=1e6等)を通すと標準化後の値を二乗する項で絶対値~1e11に達し、float32の
ULPレベルの丸め順序差が丸め後も絶対差~1e3のオーダーで残ることが判明したため、
round_output有無を問わず相対閾値に統一した。NaN/非NaNの不一致も0件。
`predict_template.html`(本番配布用テンプレート)に対しても同一の46設定×302行のテストを
`run_html_matrix_test.js`で再実行し、同じく全PASSを確認済み
(実際のFile/FileReaderドロップ経路のシミュレーションも含む)。

### linear_poly(type4)/blend(type5)専用フィクスチャ

C++参照実装は2026-07-15にtype4/5の読込に対応し、2026-07-16に整合性検査・blend再帰深さ制限を
実装したが、上記46設定のうち直近追加分は以下の観点を専用に検証する:

- **linear_poly_univariate_none_roundFalse**: 特徴量1本のみ(交差項が存在しない最小境界:
  項は`[x1, x1^2]`の2つだけ)。
- **linear_poly_mixed_log1p_roundFalse**: 特徴量4本、y_transform=log1p。単項+全ペア積+
  全二乗が混在する標準的なpoly展開。
- **linear_poly_derived_none_roundTrue**: 自動特徴量エンジニアリングの派生特徴(`x1*x2`、
  v4フォーマットの派生特徴ブロック)を明示的に投入した上でpoly展開(v4+type4の組み合わせ)。
  **このフィクスチャの追加で、`predict_linear_poly`の標準化ステップ`(x-center)/scale`が
  float単精度で減算→除算の2回丸めを行っており、JS版(倍精度で計算してから1回だけ
  float32に丸める)と極端な入力値で有意に不一致になるバグを発見・修正した**
  (積・二乗項`s[a]*s[b]`も同様にfloatではなくdoubleで計算するよう修正)。
- **blend_all_types_mixed_none_roundFalse**: 1回の学習で得られるlinear(この行数/特徴数
  条件では自動的にlinear_polyになる)/LightGBM/GP/MLPの4種類をすべてメンバーとするblend
  (blend内にtype4がネストするケースも同時に検証)。
- **blend_log1p_smear_round**: 2メンバー(linear+lgbm)、y_transform=log1p、smear!=1.0、
  有限y_clip、round_output=trueを組み合わせた後処理フルパス。
- **blend_lgbm_linear_log1p_roundFalse**(既存): 2メンバー(lgbm+linear)、log1p。

フィクスチャは`gen_type45_fixtures.py`で`train_bridge.py`の内部関数を直接呼び出して実際に
モデルを学習させて生成しており(手書きバイナリではない)、再生成は
`python gen_type45_fixtures.py`(要 lightgbm。sklearn/scipyは不要)。

- **rf / xt → lgbm**: `average_output`ヘッダを検出しleaf値を`1/木の本数`で事前スケール
  することで、JS側の「全木の和」ロジックのままLightGBMの「平均」出力と一致
  (修正前は誤差~2700〜2950と全く不一致だった)。既存のlgbm系フィクスチャ経由で検証済み。

## 残差の原因

微小な誤差(1e-7〜1e-6オーダー)は、木の分岐しきい値ぎりぎりの入力やfloat32演算の
丸め方向がC++とJSでわずかに異なることに起因すると推測される(モデルの`.treg`バイナリ
自体がfloat32精度で保存されているため、これ以上の一致は原理的に無意味)。
実用上の予測精度に影響するレベルではない。

## ファイル

- `predict-core.js` — JS推論エンジン本体(treg読み込み + 4モデル種別の予測 + 前処理/後処理)。
  **本番用の `web/predict_template.html` にも同一ロジックがインライン化されている**
  (変更する場合は両方に反映し、両方のテストを再実行すること)。
- `run_parity.js` — C++参照実装との数値比較テスト、4種別まとめて実行(`node run_parity.js`)
- `run_matrix_test.js` — 46設定 × 302行の網羅的マトリクステスト(`node run_matrix_test.js`)。
  加えて `matrix/_manifest_pyref.json` が空でなければ、そこに載った(C++がまだ対応しない
  将来の型のための予備の)フィクスチャをPython独立実装(`matrix_python_out/`)と突き合わせる
  追加検証も同じスクリプト内で実行する(現状は空で未使用)。フィクスチャが1件でも欠損している
  場合はFAIL扱いになる(低-M17。以前は[SKIP]するだけで集計対象外になり、フィクスチャ全損でも
  「0/40 FAIL:0」でCIが緑になっていた)。
- `gen_type45_fixtures.py` — linear_poly(type4)/blend(type5)専用フィクスチャの生成スクリプト。
  `train_bridge.py`の内部関数(`_try_linear`/`_try_lgbm`/`_try_gp`/`_try_mlp`/`_export_treg`/
  `_export_treg_blend`)を直接呼び出して実データを学習させ、`matrix/`に`.treg`を書き込み
  `_manifest.json`に統合する。要lightgbm(sklearn/scipyは不要。python-embedでも実行可)。
- `run_encoding_matrix_test.js` — E14: CSVエンコーディング境界ケース(大きなスケールの値・
  cp932・BOM・半角カナ混在ヘッダ)のパリティテスト。`encoding_matrix/`のフィクスチャを
  predict_template.html に実際にFile/FileReader経由(jsdom)で読ませて検証する。
- `matrix/` — 46通りの`.treg`フィクスチャ(`_manifest.json`に一覧、type0〜5すべて)+
  将来のC++非対応型のための予備フィクスチャ一覧(`_manifest_pyref.json`。現状空)
- `encoding_matrix/` — E14用の4フィクスチャ(`_manifest.json`に期待値込みで一覧)+
  生成スクリプト`gen_encoding_fixtures.py`
- `matrix_cpp_out/` — 対応するC++参照実装の予測結果CSV(46ファイル、type0〜5すべて)
- `matrix_python_out/` — `_manifest_pyref.json`用のPython独立実装の予測結果CSV(現状未使用)
- `stress_test.csv` — 302行の網羅的ストレステスト入力(欠損・極端値・範囲外値・全NaN行を含む)
- `sample_linear_model.treg` / `sample_lgbm_model_noround.treg` / `sample_gp_model.treg` /
  `sample_mlp_model.treg` — 初期4種別パリティテスト用モデル(合成データ)
- `sample_strict.csv` — 初期テスト入力(欠損値・境界値・極端な値を含む、7行)
- `sample2_pred_linear.csv` / `sample_strict_pred_lgbm.csv` / `sample2_pred_gp.csv` /
  `sample2_pred_mlp.csv` — 各モデル種別についてC++参照実装が出した正解出力
- `sample_lgbm_model.treg` / `sample.csv` / `sample_pred_reference_cpp.csv` — 初回PoC
  (lgbmのみ、round_output=trueの整数丸めあり)時の検証データ。参考として残置。
- `predict_native_ref` — 検証用にビルドしたLinux向けC++参照実装バイナリ(配布物には含めない)。

### E14: CSVエンコーディング境界ケース(run_encoding_matrix_test.js)

数値ロジックの網羅テストとは別に、「CSVの読み方」自体を検証する: 大きなスケールの値
(~1e6)・日本語Excel既定のcp932(Shift-JIS)・UTF-8 BOM付き・半角カナ+全角混在ヘッダ、
の4パターンを`encoding_matrix/`に用意し、本番配布物 predict_template.html に実際に
File/FileReader経由(jsdom)で読ませ、係数を単純な整数にした最小`.treg`から手計算した
期待値と突き合わせる。

```
PASS  large_scale          [JS]   予測=[8000999.92,-968134.24,-998999.99]
PASS  large_scale          [native] 予測=[8001000,-968134.25,-999000]
PASS  cp932_japanese       [JS]   予測=[17.0,24.0]
  [SKIP native] cp932_japanese: native実装はこのエンコーディング(cp932)に非対応
PASS  bom_utf8             [JS]   予測=[12.0,34.0]
PASS  bom_utf8             [native] 予測=[12,34]
PASS  halfwidth_kana       [JS]   予測=[17.0,7.0]
PASS  halfwidth_kana       [native] 予測=[17,7]
```

C++参照実装(`predict_native_v2.cpp`)はUTF-8 BOM除去のみでcp932デコードには非対応
(`_read_csv_with_encoding_fallback`相当の仕組みが無い)ため、cp932フィクスチャは
native比較をスキップしJS/Pythonの2系統のみで検証する(既知の制約)。
フィクスチャは`encoding_matrix/gen_encoding_fixtures.py`で再生成できる
(`train_bridge._write_treg_stream`を直接呼び出し、係数center=0/scale=1の恒等スケーラ
+単純な整数係数で組み立てるため、期待値は手計算そのまま)。

## 再現方法

```bash
npm install     # jsdomのみ(run_html_matrix_test.js/run_encoding_matrix_test.js用)。他は依存ゼロ
npm test        # run_matrix_test.js + run_parity.js + run_html_matrix_test.js + run_encoding_matrix_test.js を一括実行
```

個別実行:

```bash
node run_parity.js             # 初期パリティテスト(7行 x 4種別)
node run_matrix_test.js        # 網羅的マトリクステスト(302行 x 46設定、predict-core.js単体)
node run_html_matrix_test.js   # 同上を本番配布物 predict_template.html に対して実行
node run_encoding_matrix_test.js [native_exe_path]  # E14: エンコーディング境界ケース(4種)
```

## CI(必須ゲート)

`.github/workflows/predict-parity.yml` により、以下のパスを変更した push/PR では
上記3テストが自動実行される: `web/js_predict_poc/**`、`web/predict_template.html`、
`native_predictor/**`、`web/py/train_bridge.py`。

予測ロジックは現在「ネイティブC++(`predict_native_v2.cpp`)」「Python/Pyodide
(`predict_template.py`)」「JS(`predict-core.js`と`predict_template.html`に埋め込み)」の
3系統が並行して存在する。どれか1つを変更したら必ず`npm test`(またはCI)を通し、
3系統の数値が一致していることを確認してからマージ・配布すること。

C++参照実装を再ビルドしたい場合:

```bash
g++ -O2 -std=c++17 ../../native_predictor/predict_native_v2.cpp -o predict_native_ref
./predict_native_ref sample_strict.csv sample_<type>_model.treg
# → sample_strict_pred.csv が生成される。run_parity.js の対応ファイル名を差し替えて再実行。
```

## 本番導入(完了、2026-07)

このPoCの成果は `web/predict_template.html`(単体HTML配布テンプレート)として本番導入済み。

- 「学習済みモデルのDL」ボタン(`frontend/index.html`の`makeWebPlatform().exportModel()`)は、
  従来の exe+treg埋め込み方式から、`predict_template.html`の`__TREG_BASE64__`プレースホルダを
  base64化したtregで置き換えて単体HTMLとして書き出す方式に変更済み。
  HTMLファイルはWindowsのMOTW/SmartScreenによる「安全でない実行ファイル」判定の対象外のため、
  未署名でも警告なしで開ける。
- 全6種別(linear/lgbm/gp/mlp/linear_poly/blend、rf・xtはlgbmにエイリアス)が`.treg`対応済み
  (2026-07拡張)のため、UIで表示されるベストモデル(`best_name`)がそのまま書き出される。
  技術的な失敗(壊れたsidecarファイル等)で`best_name`の書き出しに失敗した場合のみ、
  R²降順で次点にフォールバックする(`train_bridge.py`の`deploy_order`ロジック)。
  Blend採用マージン判定で表示上は単体モデルへ格下げされているのに、デプロイだけ
  Blendのまま、という逆転が起きないよう`best_name`を必ず優先する設計になっている。
- オフライン版(`web/offline.html`)は`web/build_offline.mjs`が`predict_template.html`の中身を
  `offline-embed.js`に文字列同梱する(旧`predict_native.exe`の同梱は廃止)。
- 検証: `predict_template.html`をjsdomで実ブラウザ相当に実行し、(1) 46設定×302行の
  マトリクステスト全PASS、(2) CSV往復エスケープテスト、(3) 実際のFile/FileReaderドロップ経路
  シミュレーション、(4) エクスポート→配布物として開く→予測、の一気通貫シナリオ、を
  すべて確認済み(いずれもC++参照実装と誤差ゼロ〜float32精度限界内で一致)。
