// treg-worker.js — treg-engine.js を Web Worker 上で動かすための薄い中継スクリプト。
// Pyodideの学習/予測処理(CPUバウンド)をメインスレッドから分離し、UI(ロボアニメーション・
// 進捗バー・設定操作等)がブロックされないようにする。実際のPyodideオーケストレーション
// ロジックは一切持たず、treg-engine.js(window/document依存ゼロ)をそのままimportして
// postMessageで中継するだけ。
import * as engine from './treg-engine.js';

// Uint8Array を Transferable として送るための下ごしらえ。
// FS.readFile()/.toJs() は通常バッファ全体と一致するUint8Arrayを返すが、将来の実装変化に
// 備えて「バッファの一部だけを指すview」だった場合は安全のためコピーしてから渡す
// (postMessageのtransferListにbuffer全体を渡すと、そのbufferを参照する他の値まで
//  detachされてしまうため)。
function toTransferable(u8) {
  if (!u8) return null;
  if (u8.byteOffset === 0 && u8.byteLength === u8.buffer.byteLength) return u8;
  return u8.slice();
}

self.onmessage = async (ev) => {
  const { type, requestId } = ev.data;
  try {
    if (type === 'init') {
      await engine.initEngine({
        onStatus: (text) => self.postMessage({ type: 'status', requestId, text }),
      });
      self.postMessage({ type: 'complete', requestId, payload: null });

    } else if (type === 'train') {
      const { csvText, target, strategy } = ev.data;
      const { result, tregBytes, modelZipBytes } = await engine.train(csvText, target, strategy, {
        onLog: (line) => self.postMessage({ type: 'log', requestId, line }),
        onProgress: (pct, label) => self.postMessage({ type: 'progress', requestId, pct, label }),
      });
      const t = toTransferable(tregBytes);
      const z = toTransferable(modelZipBytes);
      self.postMessage(
        { type: 'complete', requestId, payload: { result, tregBytes: t, modelZipBytes: z } },
        [t?.buffer, z?.buffer].filter(Boolean)
      );

    } else if (type === 'predict') {
      const { csvText } = ev.data;
      const { result, predictedCsv } = await engine.predict(csvText, {
        onLog: (line) => self.postMessage({ type: 'log', requestId, line }),
      });
      const p = toTransferable(predictedCsv);
      self.postMessage(
        { type: 'complete', requestId, payload: { result, predictedCsv: p } },
        [p?.buffer].filter(Boolean)
      );

    } else {
      self.postMessage({ type: 'error', requestId, message: `未知のリクエスト種別: ${type}` });
    }
  } catch (e) {
    self.postMessage({ type: 'error', requestId, message: String(e?.message || e) });
  }
};
