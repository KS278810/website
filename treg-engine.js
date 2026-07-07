// treg-engine.js — T-regressor をブラウザ完結で動かす Pyodide アダプタ。
// 既存の train_bridge.py / _light.py / predict_template.py を無改変で駆動する。
// 計算はすべて利用者の端末(WASM)で実行され、データは一切サーバへ送信されない。

// Pyodide 本体・依存wheelはすべて ./vendor/pyodide/ にローカル同梱している。
// CDN(外部ネットワーク)には一切アクセスしない — 社内プロキシ等で外部CDNが遅い/塞がれている環境でも
// 動作するようにするため。フォルダをまるごと配布すればオフラインでも動く。
const PYODIDE_BASE = "./vendor/pyodide/";
const PY_FILES = ["_light.py", "train_bridge.py", "predict_template.py"];
const DEPS = ["numpy", "pandas", "scipy", "lightgbm", "joblib", "threadpoolctl"];

// WASM は Python スレッド不可 → ThreadPoolExecutor を逐次実行版に差し替える。
// また zip 入出力ヘルパ (_treg_zip_dir / _treg_unzip) を Python 側に用意する。
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
    // PROGRESS:<pct>:<msg>
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

export function isReady() { return _ready; }

// Pyodide 初期化 + 依存ロード + Python ソース展開（初回のみ重い）
export async function initEngine({ onStatus } = {}) {
  if (_ready) return;
  onStatus?.("エンジンを起動中…");
  const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE_BASE}pyodide.mjs`);
  _pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });

  onStatus?.("計算ライブラリを読込中…（同梱データのみ・ネットワーク不要）");
  await _pyodide.loadPackage(DEPS, { messageCallback: () => {} });

  onStatus?.("モデルコードを展開中…");
  _pyodide.FS.mkdirTree("/treg");
  for (const f of PY_FILES) {
    const res = await fetch(`./py/${f}`);
    if (!res.ok) throw new Error(`Python ソース取得失敗: ${f} (${res.status})`);
    _pyodide.FS.writeFile(`/treg/${f}`, await res.text());
  }
  await _pyodide.runPythonAsync(BOOTSTRAP_PY);
  _ready = true;
  onStatus?.("準備完了");
}

// 学習: CSV文字列 + ターゲット列 + モード('quick'|'thorough')
// 戻り値 { result, tregBytes, modelZipBytes }
export async function train(csvText, target, strategy, { onLog, onProgress } = {}) {
  if (!_ready) throw new Error("エンジン未初期化");
  _pyodide.FS.writeFile("/treg/input.csv", csvText);

  let resultJson = null;
  _pyodide.setStdout({ batched: (s) => {
    if (s.startsWith("RESULT_JSON:")) resultJson = s.slice("RESULT_JSON:".length);
    _emitLine(s, onLog, onProgress);
  }});
  _pyodide.setStderr({ batched: (s) => onLog?.("[err] " + s) });

  _pyodide.globals.set("_ARG_TARGET", target ?? "");
  _pyodide.globals.set("_ARG_STRATEGY", strategy);

  await _pyodide.runPythonAsync(`
import sys, runpy, os
os.chdir("/treg")
sys.argv = ["train_bridge.py", "/treg/input.csv", _ARG_TARGET, "0", _ARG_STRATEGY, "1"]
runpy.run_path("/treg/train_bridge.py", run_name="__main__")
`);

  if (!resultJson) throw new Error("学習結果(RESULT_JSON)が取得できませんでした");
  const result = JSON.parse(resultJson);

  const tregBytes = _readBytes("/treg/trained_model/model.treg");
  const modelZipBytes = _pyodide.runPython(`_treg_zip_dir("/treg/trained_model")`).toJs();
  return { result, tregBytes, modelZipBytes };
}

// 予測: CSV文字列 → { result, predictedCsv }
// 事前に train() 済み、または loadModel() でモデル復元済みであること
export async function predict(csvText, { onLog } = {}) {
  if (!_ready) throw new Error("エンジン未初期化");
  const hasModel = _pyodide.runPython(`_treg_has_model()`);
  if (!hasModel) throw new Error("学習済みモデルがありません。先に学習するかモデルを読み込んでください");

  _pyodide.FS.writeFile("/treg/pred_input.csv", csvText);
  let predJson = null;
  _pyodide.setStdout({ batched: (s) => {
    if (s.startsWith("PREDICT_JSON:")) predJson = s.slice("PREDICT_JSON:".length);
    _emitLine(s, onLog, null);
  }});
  _pyodide.setStderr({ batched: (s) => onLog?.("[err] " + s) });

  await _pyodide.runPythonAsync(`
import sys, runpy, os
os.chdir("/treg")
sys.argv = ["predict_template.py", "/treg/pred_input.csv"]
runpy.run_path("/treg/predict_template.py", run_name="__main__")
`);

  if (!predJson) throw new Error("予測結果(PREDICT_JSON)が取得できませんでした");
  const result = JSON.parse(predJson);
  const predictedCsv = new TextDecoder().decode(_readBytes("/treg/pred_input_predicted.csv"));
  return { result, predictedCsv };
}

// 保存済みモデル(.tregz zip)を復元 → 以後 predict() 可能
export async function loadModel(zipBytes) {
  if (!_ready) throw new Error("エンジン未初期化");
  _pyodide.globals.set("_ZIP_DATA", _pyodide.toPy(new Uint8Array(zipBytes)));
  _pyodide.runPython(`_treg_unzip(_ZIP_DATA, "/treg/trained_model")`);
  const hasModel = _pyodide.runPython(`_treg_has_model()`);
  if (!hasModel) throw new Error("無効なモデルファイルです（model_meta.json が見つかりません）");
  return true;
}

function _readBytes(path) {
  return _pyodide.FS.readFile(path); // Uint8Array
}
