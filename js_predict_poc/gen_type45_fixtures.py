"""gen_type45_fixtures.py — .treg type4(linear_poly)/type5(blend) パリティフィクスチャ生成。

高-3(CODE_REVIEW_2026-07-16.md): web/js_predict_poc/matrix/ の40設定はtype0-3のみで、
predict_native_v2.cpp の type4/5 読込対応(Phase2)を検証する専用フィクスチャが
blend 1件(blend_lgbm_linear_log1p_roundFalse.treg)しか無かった。ここで
train_bridge.py の内部関数を直接呼び出して実際にモデルを学習し(手書きバイナリ禁止)、
以下を追加生成する:

  type4 (linear_poly):
    - linear_poly_univariate_none_roundFalse   : 特徴量1本(単項のみ、交差項なし)
    - linear_poly_mixed_log1p_roundFalse        : 特徴量4本(積・二乗混在)
    - linear_poly_derived_none_roundTrue        : 派生特徴(x1*x2)を明示投入した上でpoly

  type5 (blend):
    - blend_all_types_mixed_none_roundFalse     : linear_poly/lgbm/gp/mlp の4種混成
    - blend_log1p_smear_round                   : 2メンバー、log1p+smear+round_output=True
    (既存の blend_lgbm_linear_log1p_roundFalse は2メンバーblendとして流用、再生成しない)

再現方法:
    python gen_type45_fixtures.py
    (matrix/*.treg を書き込み、_manifest.json に統合する。stress_test.csv を使って
     matrix_cpp_out/*_pred.csv を再生成するのは .github/workflows/predict-parity.yml と
     同じ手順を README に従い別途実行すること)
"""
import io
import json
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _ROOT_DIR)

import train_bridge as tb  # noqa: E402

MATRIX_DIR = os.path.join(_THIS_DIR, "matrix")
MANIFEST_PATH = os.path.join(MATRIX_DIR, "_manifest.json")


def _mk_dir():
    d = tempfile.mkdtemp(prefix="treg_gen_")
    return d


def _write_and_copy(model_dir, out_name):
    src = os.path.join(model_dir, "model.treg")
    dst = os.path.join(MATRIX_DIR, f"{out_name}.treg")
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data)
    print(f"  -> {dst} ({len(data)} bytes)")


# ─── type4: linear_poly ────────────────────────────────────────────────────

def gen_linear_poly_univariate():
    """単項のみ: 特徴量1本 → poly項は [x1, x1^2] の2項のみ(交差項が存在しない
    最小境界ケース)。"""
    rng = np.random.RandomState(1)
    n = 60
    x1 = rng.uniform(-5, 5, n)
    y = 2.0 * x1 - 0.1 * x1 ** 2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x1": x1, "y": y})

    model_dir = _mk_dir()
    r2, feat_list, model_type, preds, info = tb._try_linear(
        df, None, "y", model_dir, y_transform="none", y_params={},
        df_all=None, use_oof=False, splits=None)
    assert model_type == "linear", f"unexpected model_type={model_type}"
    with open(os.path.join(model_dir, "linear_model.pkl"), "rb") as f:
        pkl = pickle.load(f)
    assert pkl.get("use_poly"), "univariate config did not trigger poly (check POLY_MAX_ROWS/FEATS)"

    ok = tb._export_treg("linear", model_dir, "y", y_transform="none", y_params={},
                          smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                          round_output=False, x_clip_all={}, derived_recipe=[])
    assert ok, "export failed"
    _write_and_copy(model_dir, "linear_poly_univariate_none_roundFalse")
    return "linear_poly_univariate_none_roundFalse"


def gen_linear_poly_mixed():
    """積・二乗混在: 特徴量4本 → PolynomialFeatures(degree=2)が単項+全ペア積+全二乗を
    生成し、係数配列も多様な大きさになる(全モデル種別テストのうち最も項数が多いケース)。"""
    rng = np.random.RandomState(2)
    n = 150
    x1 = rng.uniform(0, 5, n)
    x2 = rng.uniform(0, 5, n)
    x3 = rng.uniform(-3, 3, n)
    x4 = rng.uniform(-3, 3, n)
    y_raw = x1 * x2 + 0.5 * x3 ** 2 - x4 + 10.0 + rng.normal(0, 0.5, n)
    y = np.clip(y_raw, 0.01, None)  # log1p用に非負を保証
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})

    y_transform, y_params = "log1p", {}
    model_dir = _mk_dir()
    r2, feat_list, model_type, preds, info = tb._try_linear(
        df, None, "y", model_dir, y_transform=y_transform, y_params=y_params,
        df_all=None, use_oof=False, splits=None)
    assert model_type == "linear"
    with open(os.path.join(model_dir, "linear_model.pkl"), "rb") as f:
        pkl = pickle.load(f)
    assert pkl.get("use_poly"), "mixed config did not trigger poly"
    assert len(pkl["feat_cols"]) == 4

    ok = tb._export_treg("linear", model_dir, "y", y_transform=y_transform, y_params=y_params,
                          smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                          round_output=False, x_clip_all={}, derived_recipe=[])
    assert ok
    _write_and_copy(model_dir, "linear_poly_mixed_log1p_roundFalse")
    return "linear_poly_mixed_log1p_roundFalse"


def gen_linear_poly_derived():
    """派生特徴併用: x1,x2,x3 に加えて明示的な派生特徴 x1*x2(mul) を投入した状態で
    poly展開する(v4の派生特徴ブロック + type4 poly項の組み合わせを検証)。
    round_output=True で整数丸め経路も同時に確認する。"""
    rng = np.random.RandomState(3)
    n = 100
    x1 = rng.uniform(0, 4, n)
    x2 = rng.uniform(0, 4, n)
    x3 = rng.uniform(0, 4, n)
    derived = x1 * x2
    y = np.round(derived + x3 + rng.normal(0, 0.2, n))
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

    recipe = [{"name": "x1*x2", "op": "mul", "cols": ["x1", "x2"]}]
    df_derived = tb._apply_derived(df, recipe)
    assert "x1*x2" in df_derived.columns

    model_dir = _mk_dir()
    r2, feat_list, model_type, preds, info = tb._try_linear(
        df_derived, None, "y", model_dir, y_transform="none", y_params={},
        df_all=None, use_oof=False, splits=None)
    assert model_type == "linear"
    with open(os.path.join(model_dir, "linear_model.pkl"), "rb") as f:
        pkl = pickle.load(f)
    assert pkl.get("use_poly"), "derived config did not trigger poly"
    assert "x1*x2" in pkl["feat_cols"], "derived feature was not selected into top_feats"

    ok = tb._export_treg("linear", model_dir, "y", y_transform="none", y_params={},
                          smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                          round_output=True, x_clip_all={}, derived_recipe=recipe)
    assert ok
    _write_and_copy(model_dir, "linear_poly_derived_none_roundTrue")
    return "linear_poly_derived_none_roundTrue"


# ─── type5: blend ──────────────────────────────────────────────────────────

def gen_blend_all_types_mixed():
    """全種別混成blend: 1回の学習で得られる4候補(Linear→この行数/特徴数条件では
    自動的にlinear_polyになる/LightGBM/GP/MLP)を全メンバーとしてblendする。
    blend自身がtype4(nested)を含む点が本フィクスチャの主眼。"""
    rng = np.random.RandomState(4)
    n = 120
    X = {f"x{i}": rng.uniform(-3, 3, n) for i in range(1, 6)}
    y = (1.5 * X["x1"] - 0.8 * X["x2"] + 0.4 * X["x1"] * X["x3"]
         + 0.2 * X["x4"] ** 2 - X["x5"] + rng.normal(0, 0.4, n))
    df = pd.DataFrame({**X, "y": y})

    model_dir = _mk_dir()
    y_transform, y_params = "none", {}

    lin = tb._try_linear(df, None, "y", model_dir, y_transform, y_params,
                          df_all=None, use_oof=False, splits=None)
    lgb = tb._try_lgbm(df, None, "y", model_dir, use_grid=False, use_oof=False,
                        y_transform=y_transform, y_params=y_params, df_all=None,
                        num_jobs=1, splits=None)
    gp = tb._try_gp(df, None, "y", model_dir, use_grid=False, use_oof=False,
                     y_transform=y_transform, y_params=y_params, df_all=None,
                     feat_cols_override=None, splits=None)
    mlp = tb._try_mlp(df, None, "y", model_dir, use_grid=False, use_oof=False,
                       y_transform=y_transform, y_params=y_params, df_all=None,
                       feat_cols_override=None, splits=None)

    with open(os.path.join(model_dir, "linear_model.pkl"), "rb") as f:
        pkl = pickle.load(f)
    assert pkl.get("use_poly"), "blend member Linear did not become linear_poly as expected"

    for name, res in (("Linear (Ridge)", lin), ("LightGBM", lgb), ("GP", gp), ("MLP", mlp)):
        assert res[0] is not None and np.isfinite(res[0]), f"{name} 学習失敗: {res}"

    candidates = {
        "Linear (Ridge)": (lin[0], lin[1], "linear", None, {}),
        "LightGBM":       (lgb[0], lgb[1], "lgbm",   None, {}),
        "GP":             (gp[0],  gp[1],  "gp",     None, {}),
        "MLP":            (mlp[0], mlp[1], "mlp",    None, {}),
    }
    weights = {"Linear (Ridge)": 0.3, "LightGBM": 0.3, "GP": 0.2, "MLP": 0.2}
    with open(os.path.join(model_dir, "blend_meta.pkl"), "wb") as f:
        pickle.dump({"models": list(candidates.keys()), "weights": weights}, f)

    ok = tb._export_treg_blend(model_dir, "y", candidates, y_transform="none", y_params={},
                                smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                                round_output=False, x_clip_all={}, derived_recipe=[])
    assert ok, "blend export failed"
    _write_and_copy(model_dir, "blend_all_types_mixed_none_roundFalse")
    return "blend_all_types_mixed_none_roundFalse"


def gen_blend_log1p_smear_round():
    """log1p+smear+round付きblend: 2メンバー(linear+lgbm)、y_transform=log1p、
    smear!=1.0、有限y_clip、round_output=True の後処理フル適用パスを検証する。"""
    rng = np.random.RandomState(5)
    n = 130
    X = {f"x{i}": rng.uniform(0, 6, n) for i in range(1, 4)}
    y_raw = 3.0 * X["x1"] + 1.5 * X["x2"] - 0.5 * X["x3"] + 8.0 + rng.normal(0, 0.6, n)
    y = np.clip(y_raw, 0.01, None)
    df = pd.DataFrame({**X, "y": y})

    model_dir = _mk_dir()
    y_transform, y_params = "log1p", {}

    lin = tb._try_linear(df, None, "y", model_dir, y_transform, y_params,
                          df_all=None, use_oof=False, splits=None)
    lgb = tb._try_lgbm(df, None, "y", model_dir, use_grid=False, use_oof=False,
                        y_transform=y_transform, y_params=y_params, df_all=None,
                        num_jobs=1, splits=None)
    for name, res in (("Linear (Ridge)", lin), ("LightGBM", lgb)):
        assert res[0] is not None and np.isfinite(res[0]), f"{name} 学習失敗: {res}"

    candidates = {
        "Linear (Ridge)": (lin[0], lin[1], "linear", None, {}),
        "LightGBM":       (lgb[0], lgb[1], "lgbm",   None, {}),
    }
    weights = {"Linear (Ridge)": 0.5, "LightGBM": 0.5}
    with open(os.path.join(model_dir, "blend_meta.pkl"), "wb") as f:
        pickle.dump({"models": list(candidates.keys()), "weights": weights}, f)

    y_min, y_max = float(np.min(y)), float(np.max(y))
    margin = (y_max - y_min) * 0.05
    ok = tb._export_treg_blend(model_dir, "y", candidates, y_transform="log1p", y_params={},
                                smear=1.08, y_clip=(y_min - margin, y_max + margin),
                                round_output=True, x_clip_all={}, derived_recipe=[])
    assert ok, "blend export failed"
    _write_and_copy(model_dir, "blend_log1p_smear_round")
    return "blend_log1p_smear_round"


def main():
    os.makedirs(MATRIX_DIR, exist_ok=True)
    new_names = []
    new_names.append(gen_linear_poly_univariate())
    new_names.append(gen_linear_poly_mixed())
    new_names.append(gen_linear_poly_derived())
    new_names.append(gen_blend_all_types_mixed())
    new_names.append(gen_blend_log1p_smear_round())

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for name in new_names:
        if name not in manifest:
            manifest.append(name)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n_manifest.json に {len(new_names)} 件追加(合計 {len(manifest)} 件)")
    print("次に stress_test.csv で予測を実行し、C++参照実装(predict_native_ref)の出力を"
          "matrix_cpp_out/ に生成すること(README/predict-parity.yml参照)。")


if __name__ == "__main__":
    main()
