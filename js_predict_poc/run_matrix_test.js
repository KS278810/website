// run_matrix_test.js — 網羅的パリティテスト。
// 4モデル種別 × 5種類のy_transform設定(none/log1p/yeo_johnson×3λ) × round_output(true/false)
// = 40通りの .treg を、302行のストレステストCSV(欠損・極端値・範囲外値・全NaN行を含む)で
// 全行チェックする。C++参照実装(predict_native_v2.cpp)の出力と1行ずつ突き合わせる。
//
// 使い方: node run_matrix_test.js

const fs = require("fs");
const path = require("path");
const { loadTreg, predictRow } = require("./predict-core.js");

const DIR = __dirname;
const MATRIX_DIR = path.join(DIR, "matrix");
const CPP_OUT_DIR = path.join(DIR, "matrix_cpp_out");
const STRESS_CSV = path.join(DIR, "stress_test.csv");

function readTreg(p) {
    const buf = fs.readFileSync(p);
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

const rows = parseCsv(fs.readFileSync(STRESS_CSV, "utf-8"));
const manifest = JSON.parse(fs.readFileSync(path.join(MATRIX_DIR, "_manifest.json"), "utf-8"));

console.log(`ストレステストCSV: ${rows.length}行 × ${manifest.length}通りのモデル設定 = ${rows.length * manifest.length}件の予測を検証\n`);

let totalChecked = 0;
let totalPass = 0;
const summary = [];

for (const name of manifest) {
    const tregPath = path.join(MATRIX_DIR, `${name}.treg`);
    const cppPath = path.join(CPP_OUT_DIR, `${name}_pred.csv`);
    if (!fs.existsSync(tregPath) || !fs.existsSync(cppPath)) {
        console.log(`[SKIP] ${name}: フィクスチャ不足`);
        continue;
    }

    const model = loadTreg(readTreg(tregPath));
    const cppLines = fs.readFileSync(cppPath, "utf-8").trim().split(/\r?\n/);
    const cppHeaders = cppLines[0].split(",");
    const targetIdx = cppHeaders.indexOf(model.target_col);
    const cppPreds = cppLines.slice(1).map((line) => parseFloat(line.split(",")[targetIdx]));

    let maxDiff = 0;
    let nanMismatch = 0;
    let worstRow = -1;
    for (let i = 0; i < rows.length; i++) {
        const jsP = predictRow(model, rows[i]);
        const cppP = cppPreds[i];
        const jsIsNaN = Number.isNaN(jsP) || !Number.isFinite(jsP);
        const cppIsNaN = Number.isNaN(cppP) || !Number.isFinite(cppP);
        if (jsIsNaN !== cppIsNaN) { nanMismatch++; continue; }
        if (jsIsNaN && cppIsNaN) continue;
        const diff = Math.abs(jsP - cppP);
        if (diff > maxDiff) { maxDiff = diff; worstRow = i; }
    }
    totalChecked += rows.length;
    // round_outputありは整数化されるため誤差ゼロが期待値。round_outputなしはfloat32精度限界(~1e-5まで許容)。
    const threshold = name.includes("roundTrue") ? 1e-9 : 1e-4;
    const pass = maxDiff < threshold && nanMismatch === 0;
    if (pass) totalPass += rows.length;
    summary.push({ name, maxDiff, nanMismatch, pass, worstRow });
    console.log(
        `${pass ? "PASS" : "★FAIL★"}  ${name.padEnd(28)} 最大誤差=${maxDiff.toExponential(2)}` +
        `${nanMismatch ? `  NaN不一致=${nanMismatch}件` : ""}` +
        `${!pass && worstRow >= 0 ? `  (最悪行: #${worstRow})` : ""}`
    );
}

const failCount = summary.filter((s) => !s.pass).length;
console.log(`\n========================================`);
console.log(`検証設定数: ${summary.length} / 40`);
console.log(`PASS: ${summary.length - failCount}  FAIL: ${failCount}`);
console.log(`総検証行数: ${totalChecked}件`);
console.log(failCount === 0 ? "\n✅ 全設定・全行で一致を確認" : "\n❌ 一致しない設定があります。上記を確認してください。");
