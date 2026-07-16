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
// 低-M17: 以前はフィクスチャ欠損時に[SKIP]してcontinueするだけで、summary/failCountの
// 集計対象から外れていた。そのためmatrix/フィクスチャがまるごと生成されていない
// (例: 生成スクリプトの実行漏れ・パス変更)状態でも「検証設定数: 0/40」のまま
// failCount=0でexit 0（CI緑）になり、実質何も検証していないことに気づけなかった。
// フィクスチャ欠損はFAIL相当として数える。
let skippedCount = 0;

for (const name of manifest) {
    const tregPath = path.join(MATRIX_DIR, `${name}.treg`);
    const cppPath = path.join(CPP_OUT_DIR, `${name}_pred.csv`);
    if (!fs.existsSync(tregPath) || !fs.existsSync(cppPath)) {
        console.log(`★FAIL★ [SKIP] ${name}: フィクスチャ不足`);
        skippedCount++;
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

const failCount = summary.filter((s) => !s.pass).length + skippedCount;
console.log(`\n========================================`);
console.log(`検証設定数: ${summary.length} / ${manifest.length}  (フィクスチャ欠損: ${skippedCount})`);
console.log(`PASS: ${summary.length - summary.filter((s) => !s.pass).length}  FAIL: ${failCount}`);
console.log(`総検証行数: ${totalChecked}件`);
// 低-M17: フィクスチャが1件でも欠損していれば「40通りすべて検証できた」という
// 前提が崩れるため、実際の不一致がゼロでも全体としてはFAIL扱いにする。
if (skippedCount > 0) {
    console.log(`\n❌ フィクスチャが${skippedCount}件欠損しています。matrix/生成スクリプトを再実行してください。`);
} else {
    console.log(failCount === 0 ? "\n✅ 全設定・全行で一致を確認" : "\n❌ 一致しない設定があります。上記を確認してください。");
}

// ── 追加検証: linear_poly/blend (type4/5) の Python独立実装との突合せ ──────────
// C++参照実装(predict_native_v2.cpp)は2026-07-15よりtype4(linear_poly)/type5(blend)にも
// 対応した(native_predictor/predict_native_v2.cpp参照。手動検証: blend/linear_polyとも
// Python参照実装に対し相対誤差1e-6未満で一致済み)が、この40設定の自動マトリクスは
// 学習を経て生成した .treg フィクスチャのみを対象にしており、type4/5用のフィクスチャは
// まだここに含めていない。代わりに、predict-core.js を独立に移植したPython版パーサ/
// 予測ロジック(treg_reader_ref.py相当。学習は経ずtrain_bridge._write_treg_streamの実装
// だけを転記したもの)の出力を `matrix_python_out/` に事前生成して保存しておき、それと
// JS(predict-core.js本体)を突き合わせる。高-N1(blend .tregのy逆変換二重適用)は
// この突合せがCIに存在しなかったために2度発生したため、再発防止として追加した。
const PY_MANIFEST_PATH = path.join(MATRIX_DIR, "_manifest_pyref.json");
const PY_OUT_DIR = path.join(DIR, "matrix_python_out");
let pyFailCount = 0;
let pyChecked = 0;
let pySkippedCount = 0;
let pyManifestLen = 0;
if (fs.existsSync(PY_MANIFEST_PATH)) {
    const pyManifest = JSON.parse(fs.readFileSync(PY_MANIFEST_PATH, "utf-8"));
    pyManifestLen = pyManifest.length;
    console.log(`\n── Python独立実装との突合せ(type4/5、自動マトリクス未収録分): ${pyManifest.length}件 ──`);
    for (const name of pyManifest) {
        const tregPath = path.join(MATRIX_DIR, `${name}.treg`);
        const pyPath = path.join(PY_OUT_DIR, `${name}_pred.csv`);
        if (!fs.existsSync(tregPath) || !fs.existsSync(pyPath)) {
            console.log(`★FAIL★ [SKIP] ${name}: フィクスチャ不足`);
            pySkippedCount++;
            continue;
        }
        const model = loadTreg(readTreg(tregPath));
        const pyLines = fs.readFileSync(pyPath, "utf-8").trim().split(/\r?\n/);
        const pyPreds = pyLines.slice(1).map((line) => parseFloat(line));

        let maxDiff = 0;
        let nanMismatch = 0;
        let worstRow = -1;
        for (let i = 0; i < rows.length; i++) {
            const jsP = predictRow(model, rows[i]);
            const pyP = pyPreds[i];
            const jsIsNaN = Number.isNaN(jsP) || !Number.isFinite(jsP);
            const pyIsNaN = Number.isNaN(pyP) || !Number.isFinite(pyP);
            if (jsIsNaN !== pyIsNaN) { nanMismatch++; continue; }
            if (jsIsNaN && pyIsNaN) continue;
            const diff = Math.abs(jsP - pyP);
            if (diff > maxDiff) { maxDiff = diff; worstRow = i; }
        }
        pyChecked += rows.length;
        const threshold = name.includes("roundTrue") ? 1e-9 : 1e-4;
        const pass = maxDiff < threshold && nanMismatch === 0;
        if (!pass) pyFailCount++;
        console.log(
            `${pass ? "PASS" : "★FAIL★"}  ${name.padEnd(40)} 最大誤差=${maxDiff.toExponential(2)}` +
            `${nanMismatch ? `  NaN不一致=${nanMismatch}件` : ""}` +
            `${!pass && worstRow >= 0 ? `  (最悪行: #${worstRow})` : ""}`
        );
    }
    console.log(`Python参照実装との突合せ: ${pyManifestLen - pyFailCount - pySkippedCount} / ${pyManifestLen} PASS` +
        `${pySkippedCount ? `  (フィクスチャ欠損: ${pySkippedCount})` : ""} (${pyChecked}行)`);
    if (pySkippedCount > 0) {
        console.log(`❌ Python参照フィクスチャが${pySkippedCount}件欠損しています。`);
    }
}
pyFailCount += pySkippedCount;

process.exit((failCount + pyFailCount) === 0 ? 0 : 1);
