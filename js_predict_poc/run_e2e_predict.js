// run_e2e_predict.js — 中-M4(CODE_REVIEW_2026-07-16.md): アプリ内予測(predict_template.py)
// と .treg 予測(native/JS)のE2Eパリティ検証用ジョブが呼び出すJS側予測ランナー。
//
// これまでのCI(predict-parity.yml)は「.tregを読む3実装同士」の閉じた比較のみで、
// 「画面のR²と配布物の予測が同じ」という保証がなかった。このスクリプトは実際に
// train_bridge.pyで学習した .treg を読み込み、指定CSVの各行を predict-core.js
// (predict_template.html/predict-core.js本体と同一ロジック)で予測してCSVに書き出す。
// これを predict_template.py(アプリ内Python予測)・native(C++参照実装)の出力と
// 突き合わせることで、3経路の数値一致を検証する(tests/e2e_compare_predictions.py 側で実施)。
//
// 使い方: node run_e2e_predict.js <入力CSV> <treg> <出力CSV>
const fs = require("fs");
const path = require("path");
const { loadTreg, predictRow } = require("./predict-core.js");

const [, , inCsv, tregPath, outCsv] = process.argv;
if (!inCsv || !tregPath || !outCsv) {
    console.error("使い方: node run_e2e_predict.js <入力CSV> <treg> <出力CSV>");
    process.exit(1);
}

// toNum: run_matrix_test.js / web/predict_template.html の toNum と同じ「全体一致」パース。
function toNum(s) {
    if (s === undefined || s === null) return NaN;
    const t = String(s).trim();
    if (t === "") return NaN;
    const v = Number(t);
    return Number.isFinite(v) ? v : NaN;
}

// 最小限のCSVパーサ(引用符対応)。E2Eの学習用サンプルデータは引用符付きセルを
// 含まない想定だが、predict_template.html の parseCSV と同じ状態機械にしておくことで
// 将来引用符付きセルを含むサンプルへ差し替えても壊れないようにする。
function parseCsv(text) {
    const rows = [];
    let row = [], field = "", inQuotes = false;
    let i = 0;
    const n = text.length;
    while (i < n) {
        const c = text[i];
        if (inQuotes) {
            if (c === '"') {
                if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
                inQuotes = false; i++; continue;
            }
            field += c; i++; continue;
        }
        if (c === '"') { inQuotes = true; i++; continue; }
        if (c === ",") { row.push(field); field = ""; i++; continue; }
        if (c === "\r") { i++; continue; }
        if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; i++; continue; }
        field += c; i++;
    }
    if (field !== "" || row.length > 0) { row.push(field); rows.push(row); }
    return rows.filter((r) => !(r.length === 1 && r[0] === ""));
}

function readTreg(p) {
    const buf = fs.readFileSync(p);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

const model = loadTreg(readTreg(tregPath));
const csvText = fs.readFileSync(inCsv, "utf-8").replace(/^﻿/, "");
const table = parseCsv(csvText);
const headers = table[0];
const dataRows = table.slice(1);

const targetIdx = headers.indexOf(model.target_col);
const outHeaders = targetIdx === -1 ? headers.concat([model.target_col]) : headers.slice();

const outLines = [outHeaders.join(",")];
for (const r of dataRows) {
    const rowObj = {};
    headers.forEach((h, i) => { rowObj[h] = toNum(r[i]); });
    const pred = predictRow(model, rowObj);
    const predStr = Number.isFinite(pred) ? String(pred) : "";
    const outRow = r.slice();
    if (targetIdx === -1) outRow.push(predStr); else outRow[targetIdx] = predStr;
    outLines.push(outRow.join(","));
}

fs.writeFileSync(outCsv, outLines.join("\n") + "\n", "utf-8");
console.log(`[OK] JS予測 ${dataRows.length}行 → ${outCsv}`);
