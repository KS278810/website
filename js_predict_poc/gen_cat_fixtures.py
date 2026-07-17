"""gen_cat_fixtures.py — .treg v5 カテゴリエンコーダ(one-hot/target encoding)
パリティフィクスチャ生成(精度レバー4/CODE_REVIEW_2026-07-16.md 観点1レバー4)。

train_bridge.py の内部関数(_prepare_categoricals/_fit_target_encoders/_apply_target_encoders/
_try_linear/_try_lgbm/_export_treg/_export_treg_blend)を直接呼び出して実際に学習し
(手書きバイナリ禁止、gen_type45_fixtures.py と同じ方針)、以下を生成する:

  - linear_onehot_cat_none_roundFalse   : 低カーディナリティ(cat1: A/B/C) one-hot + Linear
  - lgbm_target_enc_cat_none_roundFalse : 高カーディナリティ(cat2: R01-R12) target encoding + LightGBM
  - blend_cat_mixed_none_roundFalse     : one-hot(cat1) + target encoding(cat2) 混在のblend

stress_test.csv に追加した cat1/cat2 列(学習時未知カテゴリ・空セル行を含む)と組み合わせて
run_matrix_test.js から検証する。C++参照実装(predict_native_v2.cpp)は.treg v5対応済みのため
matrix_cpp_out/ とのメインループ突合せに統合する。

再現方法:
    python gen_cat_fixtures.py
    (matrix/*.treg を書き込み、_manifest.json に統合する。stress_test.csv を使った
     matrix_cpp_out/*_pred.csv の再生成は README/predict-parity.yml と同じ手順で行うこと)
"""
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
    return tempfile.mkdtemp(prefix="treg_gen_cat_")


def _write_and_copy(model_dir, out_name, expect_v5=True):
    src = os.path.join(model_dir, "model.treg")
    dst = os.path.join(MATRIX_DIR, f"{out_name}.treg")
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data)
    with open(dst, "rb") as f:
        ver = f.read(6)[4]
    print(f"  -> {dst} ({len(data)} bytes, v{ver})")
    if expect_v5:
        assert ver == 5, f"expected file_version=5, got {ver}"


def gen_linear_onehot():
    """低カーディナリティ(cat1: A/B/C, 3クラス) → one-hot + Linear (Ridge)。
    n=260 (> POLY_MAX_ROWS=200) にして poly-Ridge に落ちないよう明示的に避ける。"""
    rng = np.random.RandomState(11)
    n = 260
    x1 = rng.uniform(-3, 3, n)
    cat1 = rng.choice(["A", "B", "C"], size=n)
    effect = {"A": 1.0, "B": -2.0, "C": 4.0}
    y = 2.0 * x1 + np.array([effect[c] for c in cat1]) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x1": x1, "cat1": cat1, "y": y})

    df2, onehot_specs, target_cols, dropped = tb._prepare_categoricals(df, "y")
    assert not target_cols and not dropped, (target_cols, dropped)
    assert len(onehot_specs) == 3, onehot_specs
    cat_encoders_all = onehot_specs

    model_dir = _mk_dir()
    r2, feat_list, model_type, preds, info = tb._try_linear(
        df2, None, "y", model_dir, y_transform="none", y_params={},
        df_all=None, use_oof=False, splits=None)
    assert model_type == "linear", model_type
    with open(os.path.join(model_dir, "linear_model.pkl"), "rb") as f:
        pkl = pickle.load(f)
    assert not pkl.get("use_poly"), "unexpectedly triggered poly path"

    ok = tb._export_treg("linear", model_dir, "y", y_transform="none", y_params={},
                         smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                         round_output=False, x_clip_all={}, derived_recipe=[],
                         cat_encoders_all=cat_encoders_all)
    assert ok
    _write_and_copy(model_dir, "linear_onehot_cat_none_roundFalse")
    return "linear_onehot_cat_none_roundFalse"


def gen_lgbm_target_enc():
    """高カーディナリティ(cat2: R01-R12, 12クラス > CAT_ONEHOT_MAX_CARD=10) →
    fold内target encoding + LightGBM。"""
    rng = np.random.RandomState(12)
    n = 300
    x2 = rng.uniform(-2, 2, n)
    classes = [f"R{i:02d}" for i in range(1, 13)]
    effect = {c: (i - 6) * 1.5 for i, c in enumerate(classes)}
    cat2 = rng.choice(classes, size=n)
    y = 1.0 * x2 + np.array([effect[c] for c in cat2]) + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"x2": x2, "cat2": cat2, "y": y})

    df2, onehot_specs, target_cols, dropped = tb._prepare_categoricals(df, "y")
    assert not onehot_specs and not dropped, (onehot_specs, dropped)
    assert target_cols == ["cat2"], target_cols
    te_specs = tb._fit_target_encoders(df2, "y", target_cols)
    df3 = tb._apply_target_encoders(df2, te_specs)
    cat_encoders_all = te_specs

    model_dir = _mk_dir()
    r2, feat_list, model_type, preds, info = tb._try_lgbm(
        df3, None, "y", model_dir, use_grid=False, use_oof=False,
        y_transform="none", y_params={}, df_all=None, num_jobs=1, splits=None)
    assert model_type == "lgbm", model_type

    ok = tb._export_treg("lgbm", model_dir, "y", y_transform="none", y_params={},
                         smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                         round_output=False, x_clip_all={}, derived_recipe=[],
                         cat_encoders_all=cat_encoders_all)
    assert ok
    _write_and_copy(model_dir, "lgbm_target_enc_cat_none_roundFalse")
    return "lgbm_target_enc_cat_none_roundFalse"


def gen_blend_cat_mixed():
    """one-hot(cat1) + target encoding(cat2) 混在の2メンバーblend
    (linear=cat1中心、lgbm=cat2中心)。blend自身のcat_encoders統合ロジックを検証する。"""
    rng = np.random.RandomState(13)
    n = 250
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-2, 2, n)
    cat1 = rng.choice(["A", "B", "C"], size=n)
    classes2 = [f"R{i:02d}" for i in range(1, 13)]
    cat2 = rng.choice(classes2, size=n)
    effect1 = {"A": 1.0, "B": -2.0, "C": 4.0}
    effect2 = {c: (i - 6) * 1.2 for i, c in enumerate(classes2)}
    y = (1.5 * x1 + 0.8 * x2 + np.array([effect1[c] for c in cat1])
         + np.array([effect2[c] for c in cat2]) + rng.normal(0, 0.3, n))
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat1": cat1, "cat2": cat2, "y": y})

    df2, onehot_specs, target_cols, dropped = tb._prepare_categoricals(df, "y")
    te_specs = tb._fit_target_encoders(df2, "y", target_cols)
    df3 = tb._apply_target_encoders(df2, te_specs)
    cat_encoders_all = onehot_specs + te_specs

    model_dir = _mk_dir()
    lin = tb._try_linear(df3, None, "y", model_dir, "none", {},
                         df_all=None, use_oof=False, splits=None)
    lgb = tb._try_lgbm(df3, None, "y", model_dir, use_grid=False, use_oof=False,
                       y_transform="none", y_params={}, df_all=None, num_jobs=1, splits=None)
    for name, res in (("Linear (Ridge)", lin), ("LightGBM", lgb)):
        assert res[0] is not None and np.isfinite(res[0]), f"{name} 学習失敗: {res}"

    candidates = {
        "Linear (Ridge)": (lin[0], lin[1], "linear", None, {}),
        "LightGBM":       (lgb[0], lgb[1], "lgbm",   None, {}),
    }
    weights = {"Linear (Ridge)": 0.5, "LightGBM": 0.5}
    with open(os.path.join(model_dir, "blend_meta.pkl"), "wb") as f:
        pickle.dump({"models": list(candidates.keys()), "weights": weights}, f)

    ok = tb._export_treg_blend(model_dir, "y", candidates, y_transform="none", y_params={},
                               smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                               round_output=False, x_clip_all={}, derived_recipe=[],
                               cat_encoders_all=cat_encoders_all)
    assert ok, "blend export failed"
    # blendの外側ラッパー自身は feat_cols=[] (blendは直接特徴を持たない、既存のv4 derived
    # ブロックと同じ挙動)のため used_cat が常に空になり、外側の file_version 自体は
    # v5に上がらない(3のまま)。カテゴリエンコーダは各メンバーの自己完結した入れ子.treg
    # 側にそれぞれ書かれ、そちらがv5になる(load_tregの再帰parseで検証される)。
    _write_and_copy(model_dir, "blend_cat_mixed_none_roundFalse", expect_v5=False)
    return "blend_cat_mixed_none_roundFalse"


def gen_bool_onehot():
    """中-8ef9191フォロー: bool dtype 列は is_numeric_dtype が True を返すため、
    修正前の _prepare_categoricals は素通りして数値列として扱ってしまっていた
    (native/JSは配布後の予測入力CSVの"True"/"False"という文字列をfloatパースできず
    NaN→定数化するバグの原因)。純粋なbool dtype列(欠損なし。欠損混在だとpandasは
    object dtypeに落ちて元から正しく判定されていたため、このケースを明示的に検証する)
    が2クラスone-hot(class_value="True"/"False")として扱われることを検証する。
    stress_test.csv の 'flag' 列(True/False/未知値'Maybe'/欠損セルを含む、cat1の
    Z/空セルと同じ行に配置)と組み合わせて run_matrix_test.js から検証する。"""
    rng = np.random.RandomState(21)
    n = 260
    x1 = rng.uniform(-3, 3, n)
    flag = rng.choice([True, False], size=n)
    y = 2.0 * x1 + np.where(flag, 5.0, -5.0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x1": x1, "flag": flag, "y": y})
    assert df["flag"].dtype == bool, df["flag"].dtype

    df2, onehot_specs, target_cols, dropped = tb._prepare_categoricals(df, "y")
    assert not target_cols and not dropped, (target_cols, dropped)
    assert len(onehot_specs) == 2, onehot_specs
    assert {s["class_value"] for s in onehot_specs} == {"True", "False"}, onehot_specs
    cat_encoders_all = onehot_specs

    model_dir = _mk_dir()
    r2, feat_list, model_type, preds, info = tb._try_linear(
        df2, None, "y", model_dir, y_transform="none", y_params={},
        df_all=None, use_oof=False, splits=None)
    assert model_type == "linear", model_type

    ok = tb._export_treg("linear", model_dir, "y", y_transform="none", y_params={},
                         smear=1.0, y_clip=(-tb.X_CLIP_SENTINEL, tb.X_CLIP_SENTINEL),
                         round_output=False, x_clip_all={}, derived_recipe=[],
                         cat_encoders_all=cat_encoders_all)
    assert ok
    _write_and_copy(model_dir, "linear_onehot_bool_none_roundFalse")
    return "linear_onehot_bool_none_roundFalse"


def main():
    os.makedirs(MATRIX_DIR, exist_ok=True)
    new_names = []
    new_names.append(gen_linear_onehot())
    new_names.append(gen_lgbm_target_enc())
    new_names.append(gen_blend_cat_mixed())
    new_names.append(gen_bool_onehot())

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for name in new_names:
        if name not in manifest:
            manifest.append(name)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n_manifest.json に {len(new_names)} 件追加(合計 {len(manifest)} 件)")
    print("次に stress_test.csv(cat1/cat2列追加済み)で予測を実行し、"
          "C++参照実装(predict_native_ref)の出力を matrix_cpp_out/ に生成すること。")


if __name__ == "__main__":
    main()
