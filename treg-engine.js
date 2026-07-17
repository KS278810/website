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
  // onStatusは表示用の文字列ではなく「ステージキー」で通知する。呼び出し元
  // (frontend/index.htmlのwarmup())が翻訳し、進捗%算出のstartsWith前方一致にも
  // 依存しない(offline-engine.jsと同一のキー4種を使うこと)。
  onStatus?.("boot");
  const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE_BASE}pyodide.mjs`);
  _pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });

  onStatus?.("libs");
  await _pyodide.loadPackage(DEPS, { messageCallback: () => {} });

  onStatus?.("extract");
  _pyodide.FS.mkdirTree("/treg");
  for (const f of PY_FILES) {
    const res = await fetch(`./py/${f}`);
    if (!res.ok) throw new Error(`Python ソース取得失敗: ${f} (${res.status})`);
    _pyodide.FS.writeFile(`/treg/${f}`, await res.text());
  }
  await _pyodide.runPythonAsync(BOOTSTRAP_PY);
  _ready = true;
  onStatus?.("ready");
}

// 学習: CSV文字列 + ターゲット列 + モード('quick'|'thorough')
// 戻り値 { result, tregBytes, modelZipBytes }
export async function train(csvText, target, strategy, { onLog, onProgress } = {}) {
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
    // "PythonError: ... SystemExit: 1" のような汎用的なものになり、原因が分からない
    // (中-6)。捕捉していた ERROR: 行があればそちらを本来のエラーとして投げ直す
    // (exe版の lib.rs run_train が ERROR: 行を train_error イベントにそのまま使うのと同じ扱い)。
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
    console.warn("[TregEngine] model.treg の読み込みに失敗（配布可能なモデルなし）:", e);
  }
  const modelZipBytes = _pyodide.runPython(`_treg_zip_dir("/treg/trained_model")`).toJs();
  return { result, tregBytes, modelZipBytes };
}

// 予測: CSV文字列 → { result, predictedCsv }
// 事前に train() 済みであること
export async function predict(csvText, { onLog } = {}) {
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
    // sys.exit(1) すると、Pyodide側の例外メッセージは汎用的(SystemExit等)で原因が
    // 分からない(中-6)。捕捉していたエラー行があればそちらを優先して投げ直す
    // (train()と同じ方針。フロント側のparseKeyedMessageが[Robot]/PREDICT_ERROR:の
    // 両プレフィックスを解析してキー+パラメータへ翻訳する)。
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

function _readBytes(path) {
  return _pyodide.FS.readFile(path); // Uint8Array
}
