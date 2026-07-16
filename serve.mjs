// ローカル動作確認用の最小静的サーバー。  node serve.mjs → http://localhost:8000
// COOP/COEP は不要（本アプリはPythonスレッドを使わないため）。参考までにヘッダは付けてある。
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 8000;
const MIME = {
  ".html":"text/html; charset=utf-8", ".js":"text/javascript; charset=utf-8",
  ".mjs":"text/javascript; charset=utf-8", ".css":"text/css; charset=utf-8",
  ".json":"application/json", ".py":"text/plain; charset=utf-8",
  ".csv":"text/csv; charset=utf-8", ".png":"image/png", ".gif":"image/gif",
  ".wasm":"application/wasm", ".treg":"application/octet-stream", ".tregz":"application/zip",
};

http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split("?")[0]);
  if (p === "/") p = "/index.html";
  const file = path.join(DIR, p);
  // 低-M21: file.startsWith(DIR) は区切り文字を見ないため、DIR="/foo/bar" のとき
  // 兄弟ディレクトリ "/foo/bar-secret/x" のような文字列も「startsWith(DIR)」を
  // 満たしてしまい、path.join由来の ".." 正規化と組み合わさるとDIR外への読み出しを
  // 防ぎきれない。path.relative で実際に外側に出ていないかを確認する。
  const rel = path.relative(DIR, file);
  const isInside = rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
  if (!isInside || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404); res.end("404"); return;
  }
  res.writeHead(200, {
    "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "cross-origin",
  });
  fs.createReadStream(file).pipe(res);
// 低-M21: 引数なしlisten(PORT)はデフォルトで全インターフェース(0.0.0.0)にbindされ、
// 「ローカル動作確認用」の想定に反して同一LAN内の他端末からも到達可能になる。
// ローカルホストのみに明示的にbindする(外部公開したい場合はHOST環境変数で上書き可能)。
}).listen(PORT, process.env.HOST || "127.0.0.1",
  () => console.log(`T-regressor web → http://localhost:${PORT}`));
