import sys
import os
import json
import math
import pathlib
import pickle
import numpy as np
import pandas as pd


# ─── Y 逆変換 ─────────────────────────────────────────────────────────────────

def _invert_y(arr: np.ndarray, transform: str, params: dict) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(arr)
    if transform == "yeo_johnson":
        # Yeo-Johnson 逆変換（純数式・依存なし。native C++ の yeo_johnson_inv と同一）
        y = np.asarray(arr, float); lam = float(params.get("lambda", 1.0))
        out = np.empty_like(y); pos = y >= 0
        if abs(lam) < 1e-6:
            out[pos] = np.expm1(y[pos])
        else:
            out[pos] = (lam * y[pos] + 1.0) ** (1.0 / lam) - 1.0
        if abs(lam - 2.0) < 1e-6:
            out[~pos] = 1.0 - np.exp(-y[~pos])
        else:
            out[~pos] = 1.0 - (-(2.0 - lam) * y[~pos] + 1.0) ** (1.0 / (2.0 - lam))
        return out
    return arr


def _round_half_away(arr: np.ndarray) -> np.ndarray:
    """half-away-from-zero 丸め。native exe の std::round と一致させる。"""
    arr = np.asarray(arr, dtype=float)
    return np.copysign(np.floor(np.abs(arr) + 0.5), arr)


def _fill_values(fc, model_medians, impute_medians):
    """モデル自身の学習時 median を優先し、なければ全列 median にフォールバック。"""
    return {c: model_medians.get(c, impute_medians.get(c, 0.0)) for c in fc}


# ─── モデル別予測ヘルパー ──────────────────────────────────────────────────────

def _load_lgbm_meta(model_dir: str):
    p = os.path.join(model_dir, "lgbm_meta.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def _predict_lgbm(df: pd.DataFrame, model_dir: str,
                  feat_cols: list, impute_medians: dict,
                  y_transform: str, y_params: dict) -> np.ndarray:
    import lightgbm as lgb
    bst = lgb.Booster(model_file=os.path.join(model_dir, "lgbm_model.txt"))
    meta = _load_lgbm_meta(model_dir)
    if meta:
        fc = meta.get("feat_cols", feat_cols)
        med = meta.get("medians", {})
    else:
        fc, med = feat_cols, {}
    X = df.reindex(columns=fc).fillna(_fill_values(fc, med, impute_medians)).values.astype(float)
    return _invert_y(bst.predict(X), y_transform, y_params)


def _predict_linear(df: pd.DataFrame, model_dir: str,
                    impute_medians: dict,
                    y_transform: str, y_params: dict) -> np.ndarray:
    with open(os.path.join(model_dir, "linear_model.pkl"), "rb") as f:
        d = pickle.load(f)
    fc = d["feat_cols"]
    med = d.get("medians", {})
    X = df.reindex(columns=fc).fillna(_fill_values(fc, med, impute_medians)).values.astype(float)
    if d.get("use_poly"):
        X_s = d["scaler"].transform(X)
        return _invert_y(d["model"].predict(d["poly"].transform(X_s)), y_transform, y_params)
    X_s = d["scaler"].transform(X)
    return _invert_y(d["model"].predict(X_s), y_transform, y_params)


def _predict_gp(df: pd.DataFrame, model_dir: str,
                impute_medians: dict,
                y_transform: str, y_params: dict) -> np.ndarray:
    with open(os.path.join(model_dir, "gp_model.pkl"), "rb") as f:
        gp_data = pickle.load(f)
    fc = gp_data["feat_cols"]
    med = gp_data.get("medians", {})
    X = df.reindex(columns=fc).fillna(_fill_values(fc, med, impute_medians)).values.astype(float)
    X_s = gp_data["scaler"].transform(X)
    return _invert_y(gp_data["model"].predict(X_s), y_transform, y_params)


def _predict_mlp(df: pd.DataFrame, model_dir: str,
                 impute_medians: dict,
                 y_transform: str, y_params: dict) -> np.ndarray:
    with open(os.path.join(model_dir, "mlp_model.pkl"), "rb") as f:
        d = pickle.load(f)
    fc = d["feat_cols"]
    med = d.get("medians", {})
    X = df.reindex(columns=fc).fillna(_fill_values(fc, med, impute_medians)).values.astype(float)
    return _invert_y(d["pipeline"].predict(X), y_transform, y_params)


def _predict_lgbm_bag(kind: str, df: pd.DataFrame, model_dir: str,
                      impute_medians: dict,
                      y_transform: str, y_params: dict) -> np.ndarray:
    """LightGBM バギング多様化メンバー（RF/XT モード。テキスト形式 + sidecar meta）。"""
    import lightgbm as lgb
    bst = lgb.Booster(model_file=os.path.join(model_dir, f"{kind}_model.txt"))
    meta_p = os.path.join(model_dir, f"{kind}_meta.json")
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            m = json.load(f)
        fc = m.get("feat_cols", []); med = m.get("medians", {})
    else:
        fc, med = [], {}
    X = df.reindex(columns=fc).fillna(_fill_values(fc, med, impute_medians)).values.astype(float)
    return _invert_y(bst.predict(X), y_transform, y_params)


def _predict_by_type(model_type: str, df: pd.DataFrame, model_dir: str,
                     feat_cols_from_meta: list, impute_medians: dict,
                     y_transform: str, y_params: dict) -> np.ndarray:
    if model_type == "lgbm":
        return _predict_lgbm(df, model_dir, feat_cols_from_meta, impute_medians, y_transform, y_params)
    if model_type == "linear":
        return _predict_linear(df, model_dir, impute_medians, y_transform, y_params)
    if model_type == "gp":
        return _predict_gp(df, model_dir, impute_medians, y_transform, y_params)
    if model_type == "mlp":
        return _predict_mlp(df, model_dir, impute_medians, y_transform, y_params)
    if model_type in ("rf", "xt"):
        return _predict_lgbm_bag(model_type, df, model_dir, impute_medians, y_transform, y_params)
    raise ValueError(f"未知のモデル種別: {model_type}")


# candidates displayname → internal model_type
_NAME_TO_TYPE = {
    'Linear (Ridge)':            'linear',
    'LightGBM':                  'lgbm',
    'GaussianProcess (ARD-RBF)': 'gp',
    'MLP':                       'mlp',
    'LGBM-RF':                   'rf',
    'LGBM-XT':                   'xt',
}


# ─── 自動特徴量エンジニアリング（学習時レシピの再計算） ─────────────────────────

def _apply_derived(df: pd.DataFrame, recipe: list) -> pd.DataFrame:
    """学習時の派生特徴レシピを適用する（train_bridge._apply_derived と同一仕様）。
    ソース欠損・非有限は NaN として伝播し、各モデルの median 補完に委ねる。"""
    if not recipe:
        return df
    df = df.copy()
    nan_series = pd.Series(np.nan, index=df.index)
    for r in recipe:
        cols = r.get("cols", [])
        a = pd.to_numeric(df[cols[0]], errors="coerce") if cols and cols[0] in df.columns else nan_series
        if r.get("op") == "mul":
            b = pd.to_numeric(df[cols[1]], errors="coerce") if len(cols) > 1 and cols[1] in df.columns else nan_series
            v = (a * b).values.astype(float)
        elif r.get("op") == "sq":
            v = (a * a).values.astype(float)
        elif r.get("op") == "sign":
            v = np.sign(a.values.astype(float))
        else:
            continue
        df[r["name"]] = np.where(np.isfinite(v), v, np.nan)
    return df


def _predict_blend(df: pd.DataFrame, model_dir: str,
                   feat_cols_from_meta: list, impute_medians: dict,
                   y_transform: str, y_params: dict) -> np.ndarray:
    with open(os.path.join(model_dir, "blend_meta.pkl"), "rb") as f:
        bm = pickle.load(f)
    model_names = bm["models"]
    # v2 以降は NNLS 生重みの内積（正規化しない）。v1 は和で正規化する旧仕様。
    raw_weights = (bm.get("version", 1) >= 2) or (bm.get("normalize", True) is False)

    sub_preds  = []
    used_names = []
    for name in model_names:
        mtype = _NAME_TO_TYPE.get(name)
        if mtype is None:
            raise RuntimeError(f"Blend の未知サブモデル '{name}'")
        # サブモデルが1つでも欠けると学習時に最適化した重み構成が崩れるため、失敗は即エラー
        p = _predict_by_type(mtype, df, model_dir, feat_cols_from_meta,
                             impute_medians, y_transform, y_params)
        sub_preds.append(p)
        used_names.append(name)
        print(f"[Robot] Blend サブモデル '{name}' 完了", flush=True)

    stack_X = np.column_stack(sub_preds)
    weights = bm.get("weights", {})
    w = np.array([weights.get(n, 0.0) for n in used_names], dtype=float)

    if raw_weights:
        return stack_X @ w
    # v1 後方互換: 和で正規化
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.ones(len(used_names))
        w_sum = float(len(used_names))
    return (stack_X * w).sum(axis=1) / w_sum


def _collect_required_cols(model_type: str, model_dir: str, meta: dict) -> list:
    """予測に必要な入力列（モデル実使用列）を集める。blend は全サブモデルの union。"""
    def _pkl_cols(fname):
        p = os.path.join(model_dir, fname)
        if not os.path.exists(p):
            return []
        try:
            with open(p, "rb") as f:
                return pickle.load(f).get("feat_cols", [])
        except Exception:
            return []

    def _json_cols(fname):
        p = os.path.join(model_dir, fname)
        if not os.path.exists(p):
            return []
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f).get("feat_cols", [])
        except Exception:
            return []

    if model_type == "blend":
        cols = []
        try:
            with open(os.path.join(model_dir, "blend_meta.pkl"), "rb") as f:
                names = pickle.load(f).get("models", [])
        except Exception:
            names = []
        for name in names:
            mtype = _NAME_TO_TYPE.get(name)
            if mtype == "lgbm":
                m = _load_lgbm_meta(model_dir)
                cols += (m or {}).get("feat_cols", meta.get("feat_cols", []))
            elif mtype == "linear":
                cols += _pkl_cols("linear_model.pkl")
            elif mtype == "gp":
                cols += _pkl_cols("gp_model.pkl")
            elif mtype == "mlp":
                cols += _pkl_cols("mlp_model.pkl")
            elif mtype in ("rf", "xt"):
                cols += _json_cols(f"{mtype}_meta.json")
        seen = set()
        return [c for c in cols if not (c in seen or seen.add(c))]
    if model_type == "lgbm":
        m = _load_lgbm_meta(model_dir)
        return (m or {}).get("feat_cols", meta.get("feat_cols", []))
    if model_type in ("rf", "xt"):
        cols = _json_cols(f"{model_type}_meta.json")
        return cols if cols else meta.get("feat_cols", [])
    fname = {"linear": "linear_model.pkl", "gp": "gp_model.pkl", "mlp": "mlp_model.pkl"}.get(model_type)
    if fname:
        cols = _pkl_cols(fname)
        return cols if cols else meta.get("feat_cols", [])
    return meta.get("feat_cols", [])


def _to_raw_required(required_cols: list, recipe: list) -> list:
    """必要列のうち派生特徴をそのソース列（CSV に実在すべき列）へ展開する。"""
    by_name = {r["name"]: r for r in recipe}
    raw, seen = [], set()
    for c in required_cols:
        srcs = by_name[c].get("cols", []) if c in by_name else [c]
        for s in srcs:
            if s and s not in seen:
                seen.add(s)
                raw.append(s)
    return raw


# ─── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

    if len(sys.argv) < 2:
        sys.exit(1)

    csv_path  = sys.argv[1]
    print(f"[Robot] CSV 読み込み: {os.path.basename(csv_path)}", flush=True)

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model")
    df        = pd.read_csv(csv_path)

    meta_path = os.path.join(model_dir, "model_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    model_type   = meta["model_type"]
    feat_cols    = meta["feat_cols"]
    target_col   = meta["target_col"]
    y_transform  = meta.get("y_transform", "none")
    y_params     = meta.get("y_params", {})
    cat_encoders = meta.get("cat_encoders", {})
    x_clip       = meta.get("x_clip", {})
    postprocess  = meta.get("postprocess", {})
    smear        = postprocess.get("smear", 1.0)
    y_clip       = postprocess.get("y_clip", [-3.4e38, 3.4e38])
    round_output = postprocess.get("round_output", False)
    print(f"[Robot] モデル: {meta.get('model_label', model_type)}", flush=True)

    # カテゴリカル列エンコーディング
    if cat_encoders:
        for col, classes in cat_encoders.items():
            if col in df.columns:
                class_to_idx = {c: i for i, c in enumerate(classes)}
                df[col] = df[col].fillna('__NaN__').astype(str).map(
                    lambda x, m=class_to_idx: m.get(x, 0))
        print(f"[Robot] カテゴリ列エンコード: {list(cat_encoders.keys())}", flush=True)

    # X クリッピング
    if x_clip:
        for col, bounds in x_clip.items():
            if col in df.columns:
                lo, hi = bounds[0], bounds[1]
                df[col] = df[col].clip(lower=lo, upper=hi)

    # 自動特徴量（学習時レシピの再計算 — clip 後の値から生成、モデル入力専用の複製に追加）
    derived_recipe = meta.get("derived_features", []) or []
    if derived_recipe:
        df_model = _apply_derived(df, derived_recipe)
        print(f"[Robot] 自動特徴量 {len(derived_recipe)} 本を再計算", flush=True)
    else:
        df_model = df

    # 欠損値補完テーブル（後方互換フォールバック — 各モデルの pkl medians を優先）
    impute_medians = {}
    impute_path = os.path.join(model_dir, "impute_medians.json")
    if os.path.exists(impute_path):
        with open(impute_path, encoding="utf-8") as f:
            impute_medians = json.load(f)

    # 学習時の特徴量列が予測CSVに欠けていないか検査（欠けていても median 補完で続行）
    # 派生特徴はソース列（CSV に実在すべき生列）に展開して検査する
    required_cols = _to_raw_required(
        _collect_required_cols(model_type, model_dir, meta), derived_recipe)
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"[Robot] 警告: 学習時の列がCSVにありません → median補完で続行: {missing_cols}", flush=True)

    # 予測実行
    try:
        if model_type == "blend":
            preds_arr = _predict_blend(df_model, model_dir, feat_cols, impute_medians,
                                       y_transform, y_params)
        else:
            preds_arr = _predict_by_type(model_type, df_model, model_dir, feat_cols,
                                         impute_medians, y_transform, y_params)
    except Exception as e:
        print(f"[Robot] 予測エラー: {e}", flush=True)
        sys.exit(1)

    # 予測後処理（学習時と同じ smearing補正 / 観測レンジclip / half-away丸め）
    preds_arr = np.asarray(preds_arr, dtype=float) * smear
    preds_arr = np.clip(preds_arr, y_clip[0], y_clip[1])
    if round_output:
        preds_arr = _round_half_away(preds_arr)

    preds = pd.Series(preds_arr, index=df.index)
    df[target_col] = preds

    in_path  = pathlib.Path(csv_path)
    out_path = in_path.parent / (in_path.stem + '_predicted' + in_path.suffix)
    df.to_csv(str(out_path), index=False, encoding='utf-8')
    print(f"[Robot] 保存完了: {out_path.name}", flush=True)

    preview_df = df.head(500).copy()
    for col in preview_df.select_dtypes(include='number').columns:
        preview_df[col] = preview_df[col].apply(
            lambda v: None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v)
    preview = preview_df.to_dict('records')

    result = {
        "rows":         len(df),
        "mean":         round(float(preds.mean()), 2) if len(preds) > 0 else 0.0,
        "std":          round(float(preds.std()),  2) if len(preds) > 1 else 0.0,
        "target":       target_col,
        "columns":      list(df.columns),
        "preview":      preview,
        "output_path":  str(out_path),
        "output_name":  out_path.name,
        "missing_cols": missing_cols,
    }
    print(f"PREDICT_JSON:{json.dumps(result, ensure_ascii=False, default=str)}", flush=True)
    print("[Robot] 完了。", flush=True)
