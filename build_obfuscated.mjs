// build_obfuscated.mjs — index.html / offline.html のアプリ本体ロジック(inline <script>)
// と自作アダプタJS(treg-engine.js等)を難読化した配布物を dist_obfuscated/ に生成する。
//
// リポジトリ直下の index.html / offline.html はあえて「そのまま読める配布物」として
// 追跡している(README参照)。この方針とは矛盾させず、難読化版はビルド成果物として
// dist_obfuscated/ にのみ出力し、コミット対象外(.gitignore)とする。
//
// 実行前提: index.html / offline.html / vendor / assets / py 等が最新であること
// (`node build_frontend.mjs` [`node build_offline.mjs`] を先に実行しておくこと)。
//
// 実行方法:
//   cd web
//   npm install
//   npm run build:obfuscated

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { minify } from 'terser';
import JavaScriptObfuscator from 'javascript-obfuscator';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, 'dist_obfuscated');

const OBFUSCATOR_OPTS = {
    compact: true,
    controlFlowFlattening: false,
    stringArray: true,
    stringArrayThreshold: 0.75,
    stringArrayEncoding: ['base64'],
    renameGlobals: false,
    numbersToExpressions: false,
    splitStrings: false,
};

// 難読化対象: アプリ本体ロジックのみ。vendor/pyodide/(サードパーティ)・
// offline-embed.js(埋め込みデータblob、約59MB)は対象外(壊れる/意味がないため)。
const ADAPTER_JS_FILES = [
    { name: 'treg-engine.js', module: true },
    { name: 'treg-worker.js', module: true },
    { name: 'treg-worker-client.js', module: true },
    { name: 'offline-engine.js', module: false },
];

const HTML_FILES = [
    { name: 'index.html' },
    { name: 'offline.html' },
];

async function obfuscateJs(code, { module }) {
    const minified = await minify(code, {
        module,
        compress: { drop_console: true },
        mangle: { toplevel: true },
        format: { comments: false },
    });
    const result = JavaScriptObfuscator.obfuscate(minified.code, {
        ...OBFUSCATOR_OPTS,
        sourceType: module ? 'module' : 'script',
    });
    return result.getObfuscatedCode();
}

function replaceLastInlineScript(html, obfuscatedCode) {
    const openTag = '<script>';
    const closeTag = '</script>';
    const openIdx = html.lastIndexOf(openTag);
    const closeIdx = html.indexOf(closeTag, openIdx);
    if (openIdx === -1 || closeIdx === -1) {
        throw new Error('inline <script> ブロックが見つかりません');
    }
    return (
        html.slice(0, openIdx + openTag.length) +
        '\n' + obfuscatedCode + '\n' +
        html.slice(closeIdx)
    );
}

async function main() {
    fs.mkdirSync(OUT_DIR, { recursive: true });

    for (const { name, module } of ADAPTER_JS_FILES) {
        const srcPath = path.join(__dirname, name);
        const code = fs.readFileSync(srcPath, 'utf8');
        const obfuscated = await obfuscateJs(code, { module });
        fs.writeFileSync(path.join(OUT_DIR, name), obfuscated);
        console.log(`難読化: ${name}`);
    }

    for (const { name } of HTML_FILES) {
        const srcPath = path.join(__dirname, name);
        const html = fs.readFileSync(srcPath, 'utf8');
        const openTag = '<script>';
        const closeTag = '</script>';
        const openIdx = html.lastIndexOf(openTag);
        const closeIdx = html.indexOf(closeTag, openIdx);
        const inlineCode = html.slice(openIdx + openTag.length, closeIdx);
        // index.html の inline script は動的 import() を使う(HTTP版のみ) → module想定で処理
        const isModuleLike = name === 'index.html';
        const obfuscated = await obfuscateJs(inlineCode, { module: isModuleLike });
        const outHtml = replaceLastInlineScript(html, obfuscated);
        fs.writeFileSync(path.join(OUT_DIR, name), outHtml);
        console.log(`難読化: ${name}`);
    }

    // オフライン版が参照する残りのアセット(vendor/pyodide, offline-embed.js, py, assets等)は
    // 無改変でそのままコピーする(難読化対象外・壊れるため)。
    const PASSTHROUGH = ['offline-embed.js', 'vendor', 'assets', 'py', 'predict_template.html', 'sample_data.csv'];
    for (const entry of PASSTHROUGH) {
        const srcPath = path.join(__dirname, entry);
        if (!fs.existsSync(srcPath)) continue;
        fs.cpSync(srcPath, path.join(OUT_DIR, entry), { recursive: true });
    }

    console.log(`\n完了: ${path.relative(process.cwd(), OUT_DIR)}/ に難読化配布物を生成しました。`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
