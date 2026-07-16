// run_html_matrix_test.js — 本番配布物 web/predict_template.html そのものを jsdom で
// 実ブラウザ相当に実行し、46通りの.tregフィクスチャ(type0〜5すべて) x 302行の
// ストレステストCSVで
// C++参照実装(predict_native_v2.cpp)の出力と突き合わせる。
//
// predict-core.js単体のテスト(run_matrix_test.js)とは別に、実際にユーザーへ配布される
// HTMLファイル自体(インライン化されたJS + DOM配線)が壊れていないかを検証する。
// CI(GitHub Actions)からも呼ばれる。予測ロジックに触れる変更をした場合は必ず通すこと。
//
// 使い方:
//   cd web/js_predict_poc && npm install && node run_html_matrix_test.js
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const DIR = __dirname;
const WEB_DIR = path.join(DIR, "..");
const MATRIX_DIR = path.join(DIR, "matrix");
const CPP_OUT_DIR = path.join(DIR, "matrix_cpp_out");
const TEMPLATE_PATH = path.join(WEB_DIR, "predict_template.html");

function readTregB64(p) {
    return fs.readFileSync(p).toString("base64");
}

function readTregArrayBuffer(p) {
    const buf = fs.readFileSync(p);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

const manifest = JSON.parse(fs.readFileSync(path.join(MATRIX_DIR, "_manifest.json"), "utf-8"));
if (!fs.existsSync(TEMPLATE_PATH)) {
    console.error(`FATAL: ${TEMPLATE_PATH} が見つかりません`);
    process.exit(1);
}
const dummyB64 = readTregB64(path.join(MATRIX_DIR, manifest[0] + ".treg"));

let html = fs.readFileSync(TEMPLATE_PATH, "utf-8");
if (!html.includes("__TREG_BASE64__")) {
    console.error("FATAL: predict_template.html に __TREG_BASE64__ プレースホルダが見つかりません(テンプレート形式が変わった?)");
    process.exit(1);
}
html = html.replace("__TREG_BASE64__", dummyB64);

const dom = new JSDOM(html, { runScripts: "dangerously", resources: "usable", url: "http://localhost/predict.html",
    // jsdomはTextDecoder/TextEncoderをwindowに公開しないため、スクリプト実行(パース中に
    // 同期実行される)より前にNode組み込みのものを注入しておく必要がある
    // (post-construction代入では<script>が既に実行済みで手遅れになる)。
    beforeParse(window) { window.TextDecoder = TextDecoder; window.TextEncoder = TextEncoder; } });
const { window } = dom;

let pageError = null;
window.addEventListener("error", (e) => {
    pageError = e.error ? (e.error.stack || e.error.message) : e.message;
    console.error("PAGE ERROR:", pageError);
});

function waitTick(ms) { return new Promise((r) => setTimeout(r, ms || 10)); }

// 低-L3: run_matrix_test.js と同じ相対混合閾値(絶対1e-6下限、期待値に対し相対1e-4)。
// 積・二乗混在のpoly項やblendの加重和等、値のスケールが大きい設定を絶対1e-4固定では
// 誤判定しかねないため。
function tolerance(expected) {
    return Math.max(1e-6, 1e-4 * Math.abs(expected));
}

async function main() {
    await waitTick();
    if (pageError) throw new Error("ページ読み込み中にJSエラーが発生: " + pageError);

    const T = window.__T_PREDICT_TEST__;
    if (!T) throw new Error("window.__T_PREDICT_TEST__ が見つかりません(スクリプト実行に失敗した可能性)");

    const initialModel = T.getModel();
    if (!initialModel) throw new Error("初期モデルの自動読み込みに失敗しています");
    console.log(`[OK] 初期モデル読み込み: type=${initialModel.type} target=${initialModel.target_col}`);
    const dropDisabled = window.document.getElementById("dropZone").classList.contains("disabled");
    if (dropDisabled) throw new Error("モデル読み込み成功後もdropZoneがdisabledのまま");

    const csvText = fs.readFileSync(path.join(DIR, "stress_test.csv"), "utf-8");
    const { headers, rows } = T.parseCSV(csvText);
    console.log(`[OK] parseCSV: headers=${JSON.stringify(headers)} rows=${rows.length}`);

    let totalChecked = 0, totalPass = 0;
    const summary = [];
    for (const name of manifest) {
        const tregPath = path.join(MATRIX_DIR, `${name}.treg`);
        const cppPath = path.join(CPP_OUT_DIR, `${name}_pred.csv`);
        if (!fs.existsSync(tregPath) || !fs.existsSync(cppPath)) { console.log(`[SKIP] ${name}`); continue; }

        const model = T.loadTreg(readTregArrayBuffer(tregPath));
        const { outRows, targetColIdxInOut } = T.runPrediction(model, headers, rows);

        const cppLines = fs.readFileSync(cppPath, "utf-8").trim().split(/\r?\n/);
        const cppHeaders = cppLines[0].split(",");
        const targetIdx = cppHeaders.indexOf(model.target_col);
        const cppPreds = cppLines.slice(1).map((line) => parseFloat(line.split(",")[targetIdx]));

        let maxDiff = 0, nanMismatch = 0, worstRow = -1, rowFailCount = 0;
        for (let i = 0; i < rows.length; i++) {
            const jsValStr = outRows[i][targetColIdxInOut];
            const jsP = jsValStr === "" ? NaN : parseFloat(jsValStr);
            const cppP = cppPreds[i];
            const jsIsNaN = Number.isNaN(jsP) || !Number.isFinite(jsP);
            const cppIsNaN = Number.isNaN(cppP) || !Number.isFinite(cppP);
            if (jsIsNaN !== cppIsNaN) { nanMismatch++; continue; }
            if (jsIsNaN && cppIsNaN) continue;
            const diff = Math.abs(jsP - cppP);
            // 低-L3: run_matrix_test.js と同じ理由でroundTrueも相対混合閾値に統一(極端な
            // 入力値でのpoly項二乗により、丸め後も絶対差が大きくなり得るため)。
            const tol = tolerance(cppP);
            if (diff > tol) rowFailCount++;
            if (diff > maxDiff) { maxDiff = diff; worstRow = i; }
        }
        totalChecked += rows.length;
        const pass = rowFailCount === 0 && nanMismatch === 0;
        if (pass) totalPass += rows.length;
        summary.push({ name, maxDiff, nanMismatch, pass, worstRow });
        console.log(`${pass ? "PASS" : "★FAIL★"}  ${name.padEnd(30)} 最大誤差=${maxDiff.toExponential(2)}` +
            `${nanMismatch ? `  NaN不一致=${nanMismatch}件` : ""}${!pass && worstRow >= 0 ? `  (最悪行:#${worstRow})` : ""}`);
    }
    const failCount = summary.filter((s) => !s.pass).length;
    console.log(`\n検証設定数: ${summary.length}/${manifest.length}  PASS: ${summary.length - failCount}  FAIL: ${failCount}  総検証行数: ${totalChecked}`);

    // CSV往復エスケープテスト
    const escHeaders = ["a", "b"];
    const escRows = [["hello, world", 'say "hi"'], ["line1\nline2", "normal"]];
    const csvOut = T.buildCSVText(escHeaders, escRows);
    const reparsed = T.parseCSV(csvOut);
    const roundTripOk = JSON.stringify(reparsed.headers) === JSON.stringify(escHeaders) &&
        JSON.stringify(reparsed.rows) === JSON.stringify(escRows);
    console.log(`[${roundTripOk ? "OK" : "★FAIL★"}] CSV往復エスケープテスト(カンマ・引用符・改行を含む値)`);

    // File/FileReaderドロップ経路シミュレーション(1設定)
    const dropName = manifest.find((n) => n.startsWith("lgbm_")) || manifest[0];
    const dropTregPath = path.join(MATRIX_DIR, `${dropName}.treg`);
    let dropOk = false;
    if (fs.existsSync(dropTregPath)) {
        const dropB64 = readTregB64(dropTregPath);
        const html2 = fs.readFileSync(TEMPLATE_PATH, "utf-8").replace("__TREG_BASE64__", dropB64);
        const dom2 = new JSDOM(html2, { runScripts: "dangerously", resources: "usable", url: "http://localhost/predict.html",
            beforeParse(window) { window.TextDecoder = TextDecoder; window.TextEncoder = TextEncoder; } });
        const w2 = dom2.window;
        await waitTick();
        const csvBuf = fs.readFileSync(path.join(DIR, "stress_test.csv"));
        const file = new w2.File([csvBuf], "stress_test.csv", { type: "text/csv" });
        w2.__T_PREDICT_TEST__.handleFile(file);
        await waitTick(200);
        const lastResult = w2.__T_PREDICT_TEST__.getLastResult();
        dropOk = !!lastResult && w2.document.getElementById("resultPanel").style.display === "block";
        console.log(`[${dropOk ? "OK" : "★FAIL★"}] File/FileReaderドロップ経路シミュレーション(${dropName}): ${lastResult ? lastResult.outRows.length + "行処理" : "失敗"}`);
    }

    const allOk = failCount === 0 && roundTripOk && dropOk;
    console.log(allOk ? "\n✅ 全テストPASS" : "\n❌ 一部テストFAIL");
    process.exit(allOk ? 0 : 1);
}

main().catch((e) => { console.error("FATAL:", e); process.exit(1); });
