// offline.html 用の埋め込みデータ(offline-embed.js)を生成するビルドスクリプト。
// 対象は「JSがfetch/importで読みに行くもの」だけ:
//   - vendor/pyodide/ 一式(Pyodide本体 + numpy/pandas/scipy/lightgbm等)
//   - py/ 配下のPythonソース(train_bridge.py / _light.py / predict_template.py)
//   - sample_data.csv
// <img>等の通常のHTMLサブリソース読み込みはfile://でも動作するため対象外(assets/はそのまま相対参照)。
//
// 実行: node build_offline.mjs   (vendor/pyodide/ の中身を更新した後は再実行すること)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const VENDOR_DIR = path.join(ROOT, "vendor", "pyodide");
const PY_DIR = path.join(ROOT, "py");
const PREDICT_TEMPLATE = path.join(ROOT, "predict_template.html");
const OUT = path.join(ROOT, "offline-embed.js");

const embed = { pyodide: {}, py: {}, csv: "", predictTemplate: "" };

for (const f of fs.readdirSync(VENDOR_DIR)) {
  embed.pyodide[f] = fs.readFileSync(path.join(VENDOR_DIR, f)).toString("base64");
}
for (const f of ["_light.py", "train_bridge.py", "predict_template.py"]) {
  embed.py[f] = fs.readFileSync(path.join(PY_DIR, f), "utf-8");
}
embed.csv = fs.readFileSync(path.join(ROOT, "sample_data.csv"), "utf-8");
// 「学習済モデルのDL」が単体HTMLを組み立てられるよう、predict_template.html
// (プレースホルダ __TREG_BASE64__ 入りのテンプレート文字列、無改変)も同梱する。
// 旧: predict_native.exe を同梱してtregを追記する方式だったが、Windows MOTW/
// SmartScreenで未署名exeがブロックされる問題があったためHTML方式に置換(2026-07)。
embed.predictTemplate = fs.readFileSync(PREDICT_TEMPLATE, "utf-8");

const src = `// 自動生成ファイル。編集しないこと。生成: node build_offline.mjs\n` +
  `window.__TREG_OFFLINE_EMBED = ${JSON.stringify(embed)};\n`;
fs.writeFileSync(OUT, src);

const mb = (n) => (n / 1024 / 1024).toFixed(1);
console.log(`生成: ${OUT}`);
console.log(`  pyodide同梱: ${Object.keys(embed.pyodide).length}ファイル`);
console.log(`  pyソース: ${Object.keys(embed.py).length}ファイル`);
console.log(`  predict_template.html: ${mb(fs.statSync(PREDICT_TEMPLATE).size)} MB`);
console.log(`  出力サイズ: ${mb(fs.statSync(OUT).size)} MB`);
