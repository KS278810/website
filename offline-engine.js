// offline-engine.js — T-regressor を file:// で直接ダブルクリックして動かすための Pyodide アダプタ。
// treg-engine.js と同じロジックだが、以下が異なる:
//   - ESモジュールを使わない(file:// では <script type="module"> / 動的import() がCORSでブロックされるため)
//   - Pyodide本体・依存ライブラリ・Pythonソース・サンプルCSVは全て offline-embed.js に
//     base64/テキストとして埋め込み済みのものを使う(fetchを使わない)
//   - Pyodide内部が pyodide.asm.js だけを動的import()で読みに行くため、
//     Import Maps でその指定子(実行時に解決される絶対URL)を data:URL にリマップして回避する
// 計算はすべて利用者の端末(WASM)で実行され、通信は一切発生しない(実機で検証済み)。
window.TregEngine = (function () {
  const EMBED = window.__TREG_OFFLINE_EMBED;
  if (!EMBED) throw new Error("offline-embed.js が読み込まれていません");

  function b64ToBytes(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  // pyodide.asm.js の動的import()を data:URL にリマップする import map を
  // document.write でHTMLに同期注入する。他のモジュール解決が始まる前(スクリプト先頭)で
  // 呼ぶ必要があるため、このファイルは <body> の先頭・pyodide.js 読込より前に置くこと。
  (function injectImportMap() {
    const absAsmJs = new URL("./vendor/pyodide/pyodide.asm.js", location.href).toString();
    const dataUrl = "data:text/javascript;base64," + EMBED.pyodide["pyodide.asm.js"];
    const map = { imports: { [absAsmJs]: dataUrl } };
    document.write('<script type="importmap">' + JSON.stringify(map) + "<\/script>");
  })();

  // fetch を実ネットワークに出さず、埋め込みデータから Response を合成する版に差し替える。
  // pyodide.asm.wasm / pyodide-lock.json / python_stdlib.zip / 各依存wheel はこちらで解決される。
  const _origFetch = window.fetch.bind(window);
  window.fetch = function (url, opts) {
    const name = decodeURIComponent(String(url).split(/[\\/]/).pop().split("?")[0]);
    if (EMBED.pyodide[name] !== undefined) {
      const bytes = b64ToBytes(EMBED.pyodide[name]);
      const ct = name.endsWith(".wasm") ? "application/wasm"
        : name.endsWith(".json") ? "application/json" : "application/octet-stream";
      return Promise.resolve(new Response(bytes, { status: 200, headers: { "Content-Type": ct } }));
    }
    return _origFetch(url, opts);
  };

  const DEPS = ["numpy", "pandas", "scipy", "lightgbm", "joblib", "threadpoolctl"];

  // WASM は Python スレッド不可 → ThreadPoolExecutor を逐次実行版に差し替える。
  const BOOTSTRAP_PY = `
import sys, os, io, zipfile, shutil
import concurrent.futures as _cf
class _SyncFuture:
    def __init__(self, fn, a, k):
        try: self._r, self._e = fn(*a, **k), None
        except BaseException as e: self._r, self._e = None, e
    def result(self, timeout=None):
        if self._e is not None: raise self._e
        return self._r
    def exception(self, timeout=None): return self._e
class _SyncExecutor:
    def __init__(self, *a, **k): pass
    def submit(self, fn, *a, **k): return _SyncFuture(fn, a, k)
    def map(self, fn, *its): return [fn(*z) for z in zip(*its)]
    def shutdown(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
_cf.ThreadPoolExecutor = _SyncExecutor

os.chdir("/treg")

def _treg_zip_dir(dir_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dir_path):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, dir_path))
    return buf.getvalue()

def _treg_unzip(data, dir_path):
    if os.path.exists(dir_path): shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as z:
        z.extractall(dir_path)

def _treg_read_bytes(p):
    with open(p, "rb") as f: return f.read()

def _treg_has_model():
    return os.path.exists("/treg/trained_model/model_meta.json")
`;

  let _pyodide = null;
  let _ready = false;

  function _emitLine(line, onLog, onProgress) {
    if (line.startsWith("PROGRESS:")) {
      const rest = line.slice("PROGRESS:".length);
      const idx = rest.indexOf(":");
      const pct = parseInt(rest.slice(0, idx), 10);
      const msg = rest.slice(idx + 1);
      onProgress?.(isNaN(pct) ? null : pct, msg);
    } else if (line.startsWith("RESULT_JSON:") || line.startsWith("PREDICT_JSON:")) {
      // 結果行はログに出さない
    } else if (line.trim()) {
      onLog?.(line);
    }
  }

  function isReady() { return _ready; }

  async function initEngine({ onStatus } = {}) {
    if (_ready) return;
    // onStatusは表示用の文字列ではなく「ステージキー」で通知する。呼び出し元
    // (frontend/index.htmlのwarmup())が翻訳し、進捗%算出のstartsWith前方一致にも
    // 依存しない(treg-engine.jsと同一のキー4種を使うこと)。
    onStatus?.("boot");
    // pyodide.js は UMD 版(非ESモジュール)を通常<script>で事前読込済み想定 → window.loadPyodide
    _pyodide = await window.loadPyodide({ indexURL: "./vendor/pyodide/" });

    onStatus?.("libs");
    await _pyodide.loadPackage(DEPS, { messageCallback: () => {} });

    onStatus?.("extract");
    _pyodide.FS.mkdirTree("/treg");
    for (const [name, text] of Object.entries(EMBED.py)) {
      _pyodide.FS.writeFile(`/treg/${name}`, text);
    }
    await _pyodide.runPythonAsync(BOOTSTRAP_PY);
    _ready = true;
    onStatus?.("ready");
  }

  async function train(csvText, target, strategy, { onLog, onProgress } = {}) {
    if (!_ready) throw new Error("エンジン未初期化");
    _pyodide.FS.writeFile("/treg/input.csv", csvText);

    let resultJson = null;
    let errorLine = null;
    _pyodide.setStdout({ batched: (s) => {
      if (s.startsWith("RESULT_JSON:")) resultJson = s.slice("RESULT_JSON:".length);
      else if (s.startsWith("ERROR:")) errorLine = s;
      _emitLine(s, onLog, onProgress);
    }});
    _pyodide.setStderr({ batched: (s) => onLog?.("[err] " + s) });

    _pyodide.globals.set("_ARG_TARGET", target ?? "");
    _pyodide.globals.set("_ARG_STRATEGY", strategy);

    try {
      // train_bridge.py の学習本体は _run_main()(async def)に切り出されており、
      // Pyodide環境ではLightGBM予選/GP/MLPのfoldループ内でawait _maybe_yield()する
      // (候補/fold単位でブラウザに制御を返し、ロボアニメーション等の描画機会を作る)。
      // runpy.run_path(run_name="__main__")のままだと内部の`if __name__=='__main__':
      // asyncio.run(_run_main())`がWebLoop上でネストしたasyncio.run()を呼んでしまい
      // NotImplementedErrorになるため、importしてトップレベルawaitで直接呼ぶ必要がある。
      await _pyodide.runPythonAsync(`
import sys, os
os.chdir("/treg")
if "/treg" not in sys.path: sys.path.insert(0, "/treg")
sys.argv = ["train_bridge.py", "/treg/input.csv", _ARG_TARGET, "0", _ARG_STRATEGY, "1"]
# importはモジュールキャッシュされるため、2回目以降の学習でも sys.modules から強制的に
# 外して再ロードする(train_bridge.py先頭の_NUM_JOBS計算等、sys.argvに依存するモジュール
# レベルの初期化コードを毎回のargvで再実行させるため)。
sys.modules.pop("train_bridge", None)
import train_bridge
await train_bridge._run_main()
`);
    } catch (e) {
      // train_bridge.py が ERROR: を出して sys.exit(1) すると、Pyodide側の例外メッセージは
      // 汎用的で原因が分からない(中-6)。捕捉していた ERROR: 行があればそちらを優先する。
      if (errorLine) throw new Error(errorLine.slice("ERROR:".length).trim());
      throw e;
    }

    if (!resultJson) throw new Error("学習結果(RESULT_JSON)が取得できませんでした");
    const result = JSON.parse(resultJson);

    // export_available=false のケース(配布可能なモデルが無い)では model.treg が
    // 書き出されておらず読み込みに失敗する。これは学習自体の失敗ではないため、
    // ここで throw して train() 全体を reject させず、tregBytes=null で続行する(中-5)。
    let tregBytes = null;
    try {
      tregBytes = _readBytes("/treg/trained_model/model.treg");
    } catch (e) {
      console.warn("[OfflineEngine] model.treg の読み込みに失敗（配布可能なモデルなし）:", e);
    }
    const modelZipBytes = _pyodide.runPython(`_treg_zip_dir("/treg/trained_model")`).toJs();
    return { result, tregBytes, modelZipBytes };
  }

  async function predict(csvText, { onLog } = {}) {
    if (!_ready) throw new Error("エンジン未初期化");
    const hasModel = _pyodide.runPython(`_treg_has_model()`);
    if (!hasModel) throw new Error("学習済みモデルがありません。先に学習するかモデルを読み込んでください");

    _pyodide.FS.writeFile("/treg/pred_input.csv", csvText);
    let predJson = null;
    let errorLine = null;
    _pyodide.setStdout({ batched: (s) => {
      if (s.startsWith("PREDICT_JSON:")) predJson = s.slice("PREDICT_JSON:".length);
      else if (s.includes("PREDICT_ERROR:")) errorLine = s;
      _emitLine(s, onLog, null);
    }});
    _pyodide.setStderr({ batched: (s) => onLog?.("[err] " + s) });

    try {
      await _pyodide.runPythonAsync(`
import sys, runpy, os
os.chdir("/treg")
sys.argv = ["predict_template.py", "/treg/pred_input.csv"]
runpy.run_path("/treg/predict_template.py", run_name="__main__")
`);
    } catch (e) {
      // predict_template.py が「[Robot] PREDICT_ERROR:predict_failed:{...}」を出して
      // sys.exit(1) すると、Pyodide側の例外メッセージは汎用的で原因が分からない(中-6)。
      // フロント側のparseKeyedMessageが[Robot]/PREDICT_ERROR:を解析してキー+パラメータへ翻訳する。
      if (errorLine) throw new Error(errorLine.replace(/^\[Robot\]\s*/, ""));
      throw e;
    }

    if (!predJson) throw new Error("予測結果(PREDICT_JSON)が取得できませんでした");
    const result = JSON.parse(predJson);
    // TextDecoder().decode()はデフォルトでBOMを除去してしまい、その後Blob化で
    // UTF-8(BOM無し)として再エンコードされるため「CSVをDL」がBOMを落とす(中-M3)。
    // デコードせず生バイトのままdownloadFileに渡し、Python側が書いたBOMを保持する。
    const predictedCsv = _readBytes("/treg/pred_input_predicted.csv"); // Uint8Array
    return { result, predictedCsv };
  }

  async function loadModel(zipBytes) {
    if (!_ready) throw new Error("エンジン未初期化");
    _pyodide.globals.set("_ZIP_DATA", _pyodide.toPy(new Uint8Array(zipBytes)));
    _pyodide.runPython(`_treg_unzip(_ZIP_DATA, "/treg/trained_model")`);
    const hasModel = _pyodide.runPython(`_treg_has_model()`);
    if (!hasModel) throw new Error("無効なモデルファイルです（model_meta.json が見つかりません）");
    return true;
  }

  function getSampleCsv() { return EMBED.csv; }

  // 「学習済モデルのDL」が単体HTMLを組み立てるためのベーステンプレート文字列
  // (predict_template.html、プレースホルダ __TREG_BASE64__ 入り・無改変)。
  function getPredictTemplate() {
    if (!EMBED.predictTemplate) throw new Error("predict_template.html が同梱されていません(offline-embed.js を再生成してください)");
    return EMBED.predictTemplate;
  }

  function _readBytes(path) { return _pyodide.FS.readFile(path); }

  return { isReady, initEngine, train, predict, loadModel, getSampleCsv, getPredictTemplate };
})();
