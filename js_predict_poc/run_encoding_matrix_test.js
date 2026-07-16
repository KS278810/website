// run_encoding_matrix_test.js — E14: CSVエンコーディング境界ケースのパリティテスト。
//
// 数値ロジックの網羅テスト(run_matrix_test.js/run_html_matrix_test.js)とは別に、
// 「CSVの読み方」そのものを検証する: 大きなスケールの値(~1e6)・日本語Excel既定の
// cp932(Shift-JIS)・UTF-8 BOM付き・半角カナ+全角混在ヘッダ、の4パターンを
// 本番配布物 predict_template.html に実際にFile/FileReader経由で読ませ(jsdom)、
// 期待値(encoding_matrix/_manifest.jsonに手計算で記録済み)と突き合わせる。
// cp932はC++参照実装(predict_native_v2.cpp)が非対応(BOM除去のみでcp932デコードは
// 実装していない)ため、native_supported=falseのフィクスチャはnative比較をスキップし、
// JS側の期待値一致のみを確認する(README.md参照)。
//
// フィクスチャは encoding_matrix/gen_encoding_fixtures.py で生成する
// (係数を単純な整数にしてあるので期待値は手計算で検証可能)。
//
// 使い方:
//   python3 encoding_matrix/gen_encoding_fixtures.py encoding_matrix ../..
//   (native参照実装が必要な場合) g++ -O2 -std=c++17 ../../native_predictor/predict_native_v2.cpp -o /tmp/predict_native_ref
//   node run_encoding_matrix_test.js [native_exe_path]
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const DIR = __dirname;
const WEB_DIR = path.join(DIR, "..");
const ENC_DIR = path.join(DIR, "encoding_matrix");
const TEMPLATE_PATH = path.join(WEB_DIR, "predict_template.html");
const NATIVE_BIN = process.argv[2] || path.join(DIR, "predict_native_ref");

if (!fs.existsSync(ENC_DIR) || !fs.existsSync(path.join(ENC_DIR, "_manifest.json"))) {
    console.log("[SKIP] encoding_matrix/ フィクスチャが見つかりません。" +
        "python3 encoding_matrix/gen_encoding_fixtures.py encoding_matrix <repo_root> で生成してください。");
    process.exit(0);
}
const manifest = JSON.parse(fs.readFileSync(path.join(ENC_DIR, "_manifest.json"), "utf-8"));

function readTregB64(p) { return fs.readFileSync(p).toString("base64"); }

function waitTick(ms) { return new Promise((r) => setTimeout(r, ms || 30)); }

function loadPredictTemplate(tregB64) {
    let html = fs.readFileSync(TEMPLATE_PATH, "utf-8");
    if (!html.includes("__TREG_BASE64__")) {
        throw new Error("predict_template.html に __TREG_BASE64__ プレースホルダが見つかりません");
    }
    html = html.replace("__TREG_BASE64__", tregB64);
    const dom = new JSDOM(html, {
        runScripts: "dangerously", resources: "usable", url: "http://localhost/predict.html",
        beforeParse(window) { window.TextDecoder = TextDecoder; window.TextEncoder = TextEncoder; },
    });
    return dom.window;
}

async function runFixture(m) {
    const tregPath = path.join(ENC_DIR, m.treg);
    const csvPath = path.join(ENC_DIR, m.csv);
    if (!fs.existsSync(tregPath) || !fs.existsSync(csvPath)) {
        console.log(`★FAIL★ [SKIP] ${m.name}: フィクスチャ不足(${m.treg}/${m.csv})`);
        return false;
    }

    const win = loadPredictTemplate(readTregB64(tregPath));
    let pageError = null;
    win.addEventListener("error", (e) => { pageError = e.error ? (e.error.stack || e.error.message) : e.message; });
    await waitTick();
    if (pageError) {
        console.log(`★FAIL★ ${m.name}: ページ読み込みエラー: ${pageError}`);
        return false;
    }

    const csvBuf = fs.readFileSync(csvPath);
    const file = new win.File([csvBuf], m.csv, { type: "text/csv" });
    win.__T_PREDICT_TEST__.handleFile(file);
    await waitTick(100);
    const result = win.__T_PREDICT_TEST__.getLastResult();
    if (!result) {
        console.log(`★FAIL★ ${m.name}: JS側で予測結果が得られませんでした(${m.note})`);
        return false;
    }

    const targetIdx = result.outHeaders.indexOf(m.target_col);
    if (targetIdx < 0) {
        console.log(`★FAIL★ ${m.name}: target列 '${m.target_col}' が出力ヘッダに見つかりません: ${result.outHeaders}`);
        return false;
    }
    const jsPreds = result.outRows.map((r) => parseFloat(r[targetIdx]));
    let jsOk = true;
    for (let i = 0; i < m.expected.length; i++) {
        const diff = Math.abs(jsPreds[i] - m.expected[i]);
        const rel = diff / Math.max(1, Math.abs(m.expected[i]));
        if (rel > 1e-4) {
            jsOk = false;
            console.log(`  JS不一致 行#${i}: got=${jsPreds[i]} expected=${m.expected[i]}`);
        }
    }
    console.log(`${jsOk ? "PASS" : "★FAIL★"}  ${m.name.padEnd(20)} [JS]   予測=${JSON.stringify(jsPreds)}  (${m.note})`);

    // native参照実装(cp932非対応のフィクスチャはスキップ。README.md参照)
    let nativeOk = true;
    if (m.native_supported) {
        if (!fs.existsSync(NATIVE_BIN)) {
            console.log(`  [SKIP native] ${NATIVE_BIN} が見つかりません(g++でビルドしてから再実行してください)`);
        } else {
            const { execFileSync } = require("child_process");
            const tmpCsv = path.join(require("os").tmpdir(), m.csv);
            fs.copyFileSync(csvPath, tmpCsv);
            const predCsv = tmpCsv.replace(/\.csv$/, "_pred.csv");
            if (fs.existsSync(predCsv)) fs.unlinkSync(predCsv);
            try {
                execFileSync(NATIVE_BIN, [tmpCsv, tregPath], { stdio: "pipe" });
            } catch (e) { /* native側はGUIメッセージボックス呼び出し以外は例外を投げない想定 */ }
            if (!fs.existsSync(predCsv)) {
                nativeOk = false;
                console.log(`  ★FAIL★ native: 出力CSVが生成されませんでした`);
            } else {
                const lines = fs.readFileSync(predCsv, "utf-8").trim().split(/\r?\n/);
                const headers = lines[0].split(",");
                const nTargetIdx = headers.indexOf(m.target_col);
                const nativePreds = lines.slice(1).map((l) => parseFloat(l.split(",")[nTargetIdx]));
                for (let i = 0; i < m.expected.length; i++) {
                    const diff = Math.abs(nativePreds[i] - m.expected[i]);
                    const rel = diff / Math.max(1, Math.abs(m.expected[i]));
                    if (rel > 1e-4) {
                        nativeOk = false;
                        console.log(`  native不一致 行#${i}: got=${nativePreds[i]} expected=${m.expected[i]}`);
                    }
                }
                console.log(`${nativeOk ? "PASS" : "★FAIL★"}  ${m.name.padEnd(20)} [native] 予測=${JSON.stringify(nativePreds)}`);
            }
        }
    } else {
        console.log(`  [SKIP native] ${m.name}: native実装はこのエンコーディング(${m.encoding})に非対応(既知の制約、README.md参照)`);
    }

    return jsOk && nativeOk;
}

async function main() {
    let allOk = true;
    for (const m of manifest) {
        const ok = await runFixture(m);
        allOk = allOk && ok;
    }
    console.log(allOk ? "\n✅ E14 エンコーディング境界ケース: 全PASS" : "\n❌ E14 エンコーディング境界ケース: FAILあり");
    process.exit(allOk ? 0 : 1);
}

main().catch((e) => { console.error("FATAL:", e); process.exit(1); });
