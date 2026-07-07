// run_parity.js — predict-core.js (JS版) と C++参照実装(predict_native_v2.cpp)の
// 予測結果を、linear/lgbm/gp/mlp の4モデル種別すべてで比較する検証スクリプト。
// Node で実行する: node run_parity.js
//
// 各モデル種別のテスト用 .treg は、train_bridge.py の _export_treg(model_type, ...) を
// 既存の trained_model/ に対して直接呼び出して生成したもの(合成データ、実データではない)。
// 正解側(sample_pred_*.csv)は、同じ .treg と同じ sample_strict.csv を
// native_predictor/predict_native_v2.cpp (g++でLinux/Mac向けにビルド可能) に通した結果。
//
// C++参照実装を自分でビルドし直したい場合:
//   g++ -O2 -std=c++17 ../../native_predictor/predict_native_v2.cpp -o predict_native_ref
//   ./predict_native_ref sample_strict.csv sample_<type>_model.treg
//   (→ sample_strict_pred.csv が生成されるので、本スクリプトの対応ファイル名を差し替える)

const fs = require("fs");
const path = require("path");
const { loadTreg, predictRow } = require("./predict-core.js");

const DIR = __dirname;

function readTreg(p) {
    const buf = fs.readFileSync(p);
    // Node の Buffer はプールされた大きい ArrayBuffer の一部を指すことがあるため、
    // オフセットを考慮して正確に slice する(そのまま .buffer を渡すとズレる場合がある)。
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function parseCsv(text) {
    const lines = text.trim().split(/\r?\n/);
    const headers = lines[0].split(",");
    return lines.slice(1).map((line) => {
        const vals = line.split(",");
        const row = {};
        headers.forEach((h, i) => {
            const raw = vals[i];
            row[h] = raw === undefined || raw === "" ? NaN : parseFloat(raw);
        });
        return row;
    });
}

const MODEL_TYPES = ["linear", "lgbm", "gp", "mlp"];
const TREG_FILES = {
    linear: "sample_linear_model.treg",
    lgbm:   "sample_lgbm_model_noround.treg",
    gp:     "sample_gp_model.treg",
    mlp:    "sample_mlp_model.treg",
};
const CPP_PRED_FILES = {
    linear: "sample2_pred_linear.csv",
    lgbm:   "sample_strict_pred_lgbm.csv",
    gp:     "sample2_pred_gp.csv",
    mlp:    "sample2_pred_mlp.csv",
};

const rows = parseCsv(fs.readFileSync(path.join(DIR, "sample_strict.csv"), "utf-8"));

let overallMax = 0;
let allPass = true;
for (const mt of MODEL_TYPES) {
    const model = loadTreg(readTreg(path.join(DIR, TREG_FILES[mt])));
    console.log(`\n=== ${mt} === (v${model.file_version}, n_feat=${model.n_feat}, target=${model.target_col})`);

    const jsPreds = rows.map((row) => predictRow(model, row));
    const cppLines = fs.readFileSync(path.join(DIR, CPP_PRED_FILES[mt]), "utf-8").trim().split(/\r?\n/);
    const cppHeaders = cppLines[0].split(",");
    const targetIdx = cppHeaders.indexOf(model.target_col);
    const cppPreds = cppLines.slice(1).map((line) => parseFloat(line.split(",")[targetIdx]));

    let maxDiff = 0;
    jsPreds.forEach((jsP, i) => {
        const cppP = cppPreds[i];
        const diff = Math.abs(jsP - cppP);
        maxDiff = Math.max(maxDiff, diff);
        console.log(`  row${i}: JS=${jsP}  C++=${cppP}  diff=${diff.toExponential(3)}`);
    });
    overallMax = Math.max(overallMax, maxDiff);
    const pass = maxDiff < 1e-5;
    allPass = allPass && pass;
    console.log(`  → 最大絶対誤差: ${maxDiff.toExponential(3)}  ${pass ? "PASS" : "★要調査★"}`);
}

console.log(`\n========================================`);
console.log(`全体最大絶対誤差: ${overallMax.toExponential(3)}`);
console.log(allPass
    ? "PASS: 全モデル種別がfloat32モデル形式の精度限界内で一致(実用上は完全一致とみなせる)"
    : "FAIL: 一部モデル種別で誤差が大きすぎます");

// 参考: 速度計測(lgbmで代表計測。木の走査があるぶん最も重い部類)
const lgbmModel = loadTreg(readTreg(path.join(DIR, TREG_FILES.lgbm)));
const t0 = process.hrtime.bigint();
const lgbmModel2 = loadTreg(readTreg(path.join(DIR, TREG_FILES.lgbm)));
const t1 = process.hrtime.bigint();
const sampleRow = rows[0];
for (let i = 0; i < 10000; i++) predictRow(lgbmModel2, sampleRow);
const t2 = process.hrtime.bigint();
console.log(`\n[参考速度(lgbm)] treg読み込み: ${(Number(t1 - t0) / 1e6).toFixed(2)} ms`);
console.log(
    `[参考速度(lgbm)] 10000行の予測: ${(Number(t2 - t1) / 1e6).toFixed(2)} ms ` +
    `(1行あたり ${(Number(t2 - t1) / 1e6 / 10000 * 1000).toFixed(2)} μs)`
);
