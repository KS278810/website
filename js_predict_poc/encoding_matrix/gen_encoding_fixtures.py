"""E14: エンコーディング境界ケースの .treg + CSV フィクスチャ生成スクリプト。

native C++(predict_native_v2.cpp) / Python(predict_template.py) / JS(predict_template.html)
の3系統が「大きなスケールの数値」「cp932(Shift-JIS)」「UTF-8 BOM」「半角カナ+全角混在
ヘッダ」を同じように解釈できるかを検証するための最小 .treg + CSV フィクスチャを生成する。

係数を単純な整数にしてあるので、期待値は手計算で検証可能(スケーラは center=0,scale=1の
恒等変換なので予測値 = coef・x + intercept そのもの)。生成物は本スクリプトから再現可能
なので、フィクスチャ自体(.treg/.csv/_manifest.json)はコミットしておき、モデル書き出し
ロジック(train_bridge._write_treg_stream)を変更した場合のみ再生成すればよい。

実行: python3 gen_encoding_fixtures.py [出力先ディレクトリ] [リポジトリルート]
  (省略時: 出力先=このスクリプトと同じディレクトリ、リポジトリルート=../../..)
"""
import sys
import os
import io
import json
from types import SimpleNamespace

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
import train_bridge as tb  # noqa: E402


def make_linear_treg(feat_cols, coef, intercept, target_col="target"):
    scaler = SimpleNamespace(center_=np.zeros(len(feat_cols)), scale_=np.ones(len(feat_cols)))
    model = SimpleNamespace(coef_=np.array(coef, dtype=float), intercept_=float(intercept))
    payload = {"scaler": scaler, "model": model}
    buf = io.BytesIO()
    tb._write_treg_stream(
        buf, "linear", feat_cols, {}, payload, model_dir=None,
        target_col=target_col, y_transform="none", y_params={},
        smear=1.0, y_clip=(-3.4e38, 3.4e38), round_output=False,
        x_clip_all={}, derived_recipe=[],
    )
    return buf.getvalue()


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else THIS_DIR
    os.makedirs(out_dir, exist_ok=True)
    manifest = []

    # ── 1. large_scale: ~1e6 スケールの値(桁落ち・指数表記誤読の検出用) ──────────
    feat_cols = ["big_x1", "big_x2"]
    coef = [2.0, 3.0]
    intercept = 1000.0
    rows = [
        {"big_x1": 1000000.0, "big_x2": 2000000.0},
        {"big_x1": -1234567.891, "big_x2": 500000.5},
        {"big_x1": 999999.999, "big_x2": -999999.999},
    ]
    treg = make_linear_treg(feat_cols, coef, intercept)
    with open(os.path.join(out_dir, "large_scale.treg"), "wb") as f:
        f.write(treg)
    with open(os.path.join(out_dir, "large_scale.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(",".join(feat_cols) + "\n")
        for r in rows:
            f.write(f"{r['big_x1']},{r['big_x2']}\n")
    expected = [coef[0] * r["big_x1"] + coef[1] * r["big_x2"] + intercept for r in rows]
    manifest.append({
        "name": "large_scale", "csv": "large_scale.csv", "treg": "large_scale.treg",
        "encoding": "utf-8", "native_supported": True,
        "feat_cols": feat_cols, "target_col": "target", "expected": expected,
        "note": "1e6スケールの大きな値・負の大きな値で桁落ち/指数表記誤読が無いか検証",
    })

    # ── 2. cp932_japanese: 日本語Excel既定のShift-JIS(cp932)エンコード ──────────
    # native(predict_native_v2.cpp)はUTF-8 BOM除去のみでcp932デコードには非対応のため
    # native_supported=False とし、Python/JSの2系統のみで突き合わせる。
    feat_cols = ["コスト", "売上"]
    coef = [2.0, 5.0]
    intercept = 0.0
    rows = [{"コスト": 1.0, "売上": 3.0}, {"コスト": 2.0, "売上": 4.0}]
    treg = make_linear_treg(feat_cols, coef, intercept)
    with open(os.path.join(out_dir, "cp932_japanese.treg"), "wb") as f:
        f.write(treg)
    csv_text = ",".join(feat_cols) + "\n" + "".join(
        f"{r['コスト']},{r['売上']}\n" for r in rows)
    with open(os.path.join(out_dir, "cp932_japanese.csv"), "wb") as f:
        f.write(csv_text.encode("cp932"))
    expected = [coef[0] * r["コスト"] + coef[1] * r["売上"] + intercept for r in rows]
    manifest.append({
        "name": "cp932_japanese", "csv": "cp932_japanese.csv", "treg": "cp932_japanese.treg",
        "encoding": "cp932", "native_supported": False,
        "feat_cols": feat_cols, "target_col": "target", "expected": expected,
        "note": "日本語Excel既定のcp932(Shift-JIS)CSV。native実装はcp932デコード非対応のためskip",
    })

    # ── 3. bom_utf8: UTF-8 BOM付きCSV ────────────────────────────────────────
    feat_cols = ["a", "b"]
    coef = [10.0, 1.0]
    intercept = 0.0
    rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
    treg = make_linear_treg(feat_cols, coef, intercept)
    with open(os.path.join(out_dir, "bom_utf8.treg"), "wb") as f:
        f.write(treg)
    csv_text = ",".join(feat_cols) + "\n" + "".join(f"{r['a']},{r['b']}\n" for r in rows)
    with open(os.path.join(out_dir, "bom_utf8.csv"), "wb") as f:
        f.write(b"\xef\xbb\xbf" + csv_text.encode("utf-8"))
    expected = [coef[0] * r["a"] + coef[1] * r["b"] + intercept for r in rows]
    manifest.append({
        "name": "bom_utf8", "csv": "bom_utf8.csv", "treg": "bom_utf8.treg",
        "encoding": "utf-8-sig", "native_supported": True,
        "feat_cols": feat_cols, "target_col": "target", "expected": expected,
        "note": "UTF-8 BOM(EF BB BF)付きCSV。ヘッダ1文字目が誤ってBOM残骸として削られないか検証",
    })

    # ── 4. halfwidth_kana: 半角カナ+全角英数字混在ヘッダ(UTF-8) ──────────────────
    # 半角カナ"ｺｽﾄ"のUTF-8 1バイト目はBOMの1バイト目(0xEF)と一致するため、雑なBOM除去だと
    # 誤って文字化けさせてしまう(中-M5で修正済み)。
    feat_cols = ["ｺｽﾄ", "１号機"]
    coef = [2.0, 3.0]
    intercept = 0.0
    rows = [{"ｺｽﾄ": 1.0, "１号機": 5.0}, {"ｺｽﾄ": 2.0, "１号機": 1.0}]
    treg = make_linear_treg(feat_cols, coef, intercept)
    with open(os.path.join(out_dir, "halfwidth_kana.treg"), "wb") as f:
        f.write(treg)
    csv_text = ",".join(feat_cols) + "\n" + "".join(
        f"{r['ｺｽﾄ']},{r['１号機']}\n" for r in rows)
    with open(os.path.join(out_dir, "halfwidth_kana.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)
    expected = [coef[0] * r["ｺｽﾄ"] + coef[1] * r["１号機"] + intercept for r in rows]
    manifest.append({
        "name": "halfwidth_kana", "csv": "halfwidth_kana.csv", "treg": "halfwidth_kana.treg",
        "encoding": "utf-8", "native_supported": True,
        "feat_cols": feat_cols, "target_col": "target", "expected": expected,
        "note": "半角カナ・全角英数字混在ヘッダ。UTF-8バイト列の1バイト目がBOMと衝突するケース",
    })

    with open(os.path.join(out_dir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"生成完了: {len(manifest)}件 -> {out_dir}")
    for m in manifest:
        print(f"  {m['name']}: expected={m['expected']}")


if __name__ == "__main__":
    main()
