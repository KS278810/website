// run_matrix_test.js — 網羅的パリティテスト。
// 4モデル種別(linear/lgbm/gp/mlp) × 5種類のy_transform設定(none/log1p/yeo_johnson×3λ) ×
// round_output(true/false) = 40通りに加え、高-3(2026-07-16)で追加した
// linear_poly(type4、単項のみ/積・二乗混在/派生特徴併用の3設定)と
// blend(type5、2メンバー/全種別混成/log1p+smear+round付きの3設定)を合わせた
// 計46通りの .treg を、302行のストレステストCSV(欠損・極端値・範囲外値・全NaN行を含む)で
// 全行チェックする。C++参照実装(predict_native_v2.cpp)はtype0〜5すべてに対応済みのため、
// 全設定をその出力(matrix_cpp_out/)と1行ずつ突き合わせる。
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

// 低-L2: 本番(web/predict_template.html の toNum)と同じ「全体一致」パース。
// parseFloat は部分パース("1.5abc" → 1.5 を許してしまう)のため、本番と異なる
// 入力を通してしまう恐れがあった。Number() は空白以外の非数値混入を全体でNaN化する。
function toNum(s) {
    if (s === undefined || s === null) return NaN;
    const t = String(s).trim();
    if (t === "") return NaN;
    const v = Number(t);
    return Number.isFinite(v) ? v : NaN;
}

// 精度レバー4/.treg v5: カテゴリエンコーダ(one-hot/target encoding)は元のCSV文字列
// (rawRow)同士でマッチングするため、数値化済みのrowだけでなく生の文字列行も返す。
function parseCsv(text) {
    const lines = text.trim().split(/\r?\n/);
    const headers = lines[0].split(",");
    return lines.slice(1).map((line) => {
        const vals = line.split(",");
        const row = {};
        const rawRow = {};
        headers.forEach((h, i) => {
            row[h] = toNum(vals[i]);
            rawRow[h] = vals[i];
        });
        return { row, rawRow };
    });
}

// 低-L3: 許容誤差を絶対値1e-4固定から相対混合閾値へ。値のスケールが大きい設定
// (積・二乗混在のpoly項、blendの加重和等)では絶対1e-4が厳しすぎたり緩すぎたり
// するため、期待値の絶対値に応じて相対的にスケールする。
function tolerance(expected) {
    return Math.max(1e-6, 1e-4 * Math.abs(expected));
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
    const cppPreds = cppLines.slice(1).map((line) => toNum(line.split(",")[targetIdx]));

    let maxDiff = 0;
    let nanMismatch = 0;
    let worstRow = -1;
    let rowFailCount = 0;
    for (let i = 0; i < rows.length; i++) {
        const jsP = predictRow(model, rows[i].row, rows[i].rawRow);
        const cppP = cppPreds[i];
        const jsIsNaN = Number.isNaN(jsP) || !Number.isFinite(jsP);
        const cppIsNaN = Number.isNaN(cppP) || !Number.isFinite(cppP);
        if (jsIsNaN !== cppIsNaN) { nanMismatch++; continue; }
        if (jsIsNaN && cppIsNaN) continue;
        const diff = Math.abs(jsP - cppP);
        // 低-L3: 期待値に応じた相対混合閾値(絶対1e-6下限、期待値に対し相対1e-4)で判定する。
        // 以前は round_output=True の設定を「整数化されるため誤差ゼロのはず」として
        // 絶対1e-9固定にしていたが、type4(linear_poly)フィクスチャに極端な入力値
        // (x=1e6等)を通すと、標準化後の値を二乗する項で絶対値~1e11に達し、float32の
        // ULPレベルの丸め順序差(相対誤差~4e-8)がround_output後も絶対差~3.8e3として
        // 残ることが判明した(linear_poly_derived_none_roundTrueで実測)。丸めは1未満の
        // ノイズしか消さないため、巨大な値では絶対1e-9固定は本質的に無理な要求だった。
        // round_output有無を問わず同じ相対混合閾値にする。
        const tol = tolerance(cppP);
        if (diff > tol) rowFailCount++;
        if (diff > maxDiff) { maxDiff = diff; worstRow = i; }
    }
    totalChecked += rows.length;
    const pass = rowFailCount === 0 && nanMismatch === 0;
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

// ── 追加検証: C++非対応の型が出てきた場合のPython独立実装との突合せ(現状未使用) ──
// 高-3: C++参照実装(predict_native_v2.cpp)は2026-07-15よりtype4(linear_poly)/
// type5(blend)に対応済みで、これらのフィクスチャは上のメインループで
// matrix_cpp_out/ とのC++突合せとして検証されている(matrix/_manifest.json に統合済み。
// linear_poly: 単項のみ/積・二乗混在/派生特徴併用の3設定、blend: 2メンバー/
// 全種別混成/log1p+smear+round付きの3設定を追加)。
// このPython独立実装(matrix_python_out/)との突合せ経路は、将来C++がまだ対応しない
// 新しい .treg 型(例: 将来のv5カテゴリエンコーダ等)が出た場合のための予備の仕組みとして
// 残してある。matrix/_manifest_pyref.json が空でない場合のみ実行される
// (現状は空 = 未使用)。
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
        const pyPreds = pyLines.slice(1).map((line) => toNum(line));

        let maxDiff = 0;
        let nanMismatch = 0;
        let worstRow = -1;
        let rowFailCount = 0;
        for (let i = 0; i < rows.length; i++) {
            const jsP = predictRow(model, rows[i].row, rows[i].rawRow);
            const pyP = pyPreds[i];
            const jsIsNaN = Number.isNaN(jsP) || !Number.isFinite(jsP);
            const pyIsNaN = Number.isNaN(pyP) || !Number.isFinite(pyP);
            if (jsIsNaN !== pyIsNaN) { nanMismatch++; continue; }
            if (jsIsNaN && pyIsNaN) continue;
            const diff = Math.abs(jsP - pyP);
            const tol = tolerance(pyP);
            if (diff > tol) rowFailCount++;
            if (diff > maxDiff) { maxDiff = diff; worstRow = i; }
        }
        pyChecked += rows.length;
        const pass = rowFailCount === 0 && nanMismatch === 0;
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
