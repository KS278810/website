// treg-worker-client.js — メインスレッドから treg-worker.js を呼ぶための薄いプロキシ。
// treg-engine.js と同一の関数シグネチャ(isReady/initEngine/train/predict)をexportするため、
// frontend/index.html 側の呼び出しコードは Worker化前と全く同じ書き方のままで済む。
let _worker = null;
let _ready = false;
let _nextId = 1;
const _pending = new Map(); // requestId -> { resolve, reject, onLog?, onProgress?, onStatus? }

function ensureWorker() {
  if (_worker) return _worker;
  _worker = new Worker('./treg-worker.js', { type: 'module' });
  _worker.onmessage = (ev) => onMessage(ev.data);
  _worker.onerror = (ev) => onWorkerCrash(ev);
  return _worker;
}

function onMessage(msg) {
  const p = _pending.get(msg.requestId);
  if (!p) return; // 既にresolve/reject済みのリクエストからの残留メッセージは無視
  switch (msg.type) {
    case 'status':
      p.onStatus?.(msg.text);
      break;
    case 'log':
      p.onLog?.(msg.line);
      break;
    case 'progress':
      p.onProgress?.(msg.pct, msg.label);
      break;
    case 'complete':
      _pending.delete(msg.requestId);
      p.resolve(msg.payload);
      break;
    case 'error':
      _pending.delete(msg.requestId);
      p.reject(new Error(msg.message));
      break;
  }
}

// Workerがクラッシュ(構文エラー・OOM等)した場合、保留中の全リクエストを失敗させ、
// 次回呼び出し時に新しいWorkerをゼロから生成してやり直せるようにする(自己修復)。
function onWorkerCrash(ev) {
  const message = '学習エンジン(Worker)がクラッシュしました: ' + (ev?.message || ev);
  for (const [, p] of _pending) p.reject(new Error(message));
  _pending.clear();
  _worker = null;
  _ready = false;
}

function call(type, extra, cbs = {}) {
  return new Promise((resolve, reject) => {
    const requestId = _nextId++;
    _pending.set(requestId, { resolve, reject, ...cbs });
    try {
      ensureWorker().postMessage({ type, requestId, ...extra });
    } catch (e) {
      _pending.delete(requestId);
      reject(e);
    }
  });
}

export function isReady() { return _ready; }

export async function initEngine({ onStatus } = {}) {
  if (_ready) return;
  await call('init', {}, { onStatus });
  _ready = true;
}

export async function train(csvText, target, strategy, { onLog, onProgress } = {}) {
  return call('train', { csvText, target, strategy }, { onLog, onProgress });
}

export async function predict(csvText, { onLog } = {}) {
  return call('predict', { csvText }, { onLog });
}
