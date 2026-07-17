import sys
import os
import json
import math
import pathlib
import pickle
import numpy as np
import pandas as pd


# ─── CSV 読み込み（日本語Excel既定のShift-JISフォールバック） ───────────────────

def _read_csv_with_encoding_fallback(csv_path):
    """まずUTF-8として読み込みを試み、デコードできなければ Shift-JIS(cp932) として読む。
    日本語Excelが既定で書き出すShift-JIS CSVがUTF-8として「�」化けしたまま
    サイレントに予測が完走してしまうのを防ぐ(中-7)。
    低-M21: 以前はUTF-8妥当性の事前検査のためファイル全体を素のバイト列として読み込み
    (`f.read()`でメモリに全展開)、その上でさらにpandasにも読ませていたため、大きな
    CSVで実質2回分のI/O・デコードが走っていた。pd.read_csv自体もutf-8で全文デコード
    するので不正バイトがあれば同じくUnicodeDecodeErrorを送出する。それをそのまま
    フォールバック判定に使えば1回の読み込みで済む。"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"[Robot] CSVがUTF-8として不正 → Shift-JIS(cp932)として読み込みます", flush=True)
        df = pd.read_csv(csv_path, encoding='cp932')
    # 列選択UI（frontend/index.html・lib.rs）はヘッダの前後空白をtrimして送信するため、
    # pandas側の列名もtrimして揃える(中-M7)。
    df.columns = df.columns.str.strip()
    # 低-1: 非有限値(inf/-inf。pandasは"inf"/"Infinity"のような文字列セルも数値列なら
    # そのままfloat('inf')として読み込んでしまう)をNaNに正規化する。C++版は
    # パース直後に!std::isfinite(v)でNaN化、JS版はNumber.isFiniteで弾いており、
    # 3実装で非有限値の扱いを統一する(低-1)。
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _sanitize_json(obj):
    """dict/list を再帰し、非有限 float を None に置換する（train_bridge.py と同一実装）。
    列レベルの apply では NoneがSeries再構成時にNaNへ戻ってしまう（高-M2）ため、
    to_dict('records') 後の生の dict/list に対して再帰的に適用する。"""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


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


def _build_feature_matrix(df, fc, med, impute_medians):
    """指定列を数値行列として取り出す。数値として不正な文字列(例: "12abc")が
    混入したセルは pd.to_numeric(errors="coerce") で NaN 化してから median 補完する
    (欠損セルと同じ経路で扱う)。以前は reindex 後にいきなり .values.astype(float) を
    呼んでおり、そのようなセルが1つでもある列があると ValueError で予測全体が
    クラッシュしていた。native exe は std::stod の部分パース(例: "12abc"→12として
    使ってしまう)、Web版JSは Number() ベースで NaN 化(=本関数と同じ挙動)と、
    実装ごとに挙動が食い違っていた(中-10)。native側もこの関数と同じ「非数値→NaN→
    median補完」に合わせて修正済み(native_predictor/predict_native_v2.cpp)。"""
    sub = df.reindex(columns=fc).apply(pd.to_numeric, errors="coerce")
    return sub.fillna(_fill_values(fc, med, impute_medians)).values.astype(float)


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
    X = _build_feature_matrix(df, fc, med, impute_medians)
    return _invert_y(bst.predict(X), y_transform, y_params)


def _predict_linear(df: pd.DataFrame, model_dir: str,
                    impute_medians: dict,
                    y_transform: str, y_params: dict) -> np.ndarray:
    with open(os.path.join(model_dir, "linear_model.pkl"), "rb") as f:
        d = pickle.load(f)
    fc = d["feat_cols"]
    med = d.get("medians", {})
    X = _build_feature_matrix(df, fc, med, impute_medians)
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
    X = _build_feature_matrix(df, fc, med, impute_medians)
    X_s = gp_data["scaler"].transform(X)
    return _invert_y(gp_data["model"].predict(X_s), y_transform, y_params)


def _predict_mlp(df: pd.DataFrame, model_dir: str,
                 impute_medians: dict,
                 y_transform: str, y_params: dict) -> np.ndarray:
    with open(os.path.join(model_dir, "mlp_model.pkl"), "rb") as f:
        d = pickle.load(f)
    fc = d["feat_cols"]
    med = d.get("medians", {})
    X = _build_feature_matrix(df, fc, med, impute_medians)
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
    X = _build_feature_matrix(df, fc, med, impute_medians)
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


def _to_raw_required(required_cols: list, recipe: list, cat_encoders: list = None) -> list:
    """必要列のうち派生特徴をそのソース列（CSV に実在すべき列）へ展開する。
    精度レバー4: カテゴリエンコーダの生成列(one-hot indicator名やtarget-encoding後の
    元列名)もさらに source_col まで1段解決する(native/JS版 raw_source_for と同一仕様)。"""
    by_name = {r["name"]: r for r in recipe}
    cat_src = {c["feature_name"]: c["source_col"] for c in (cat_encoders or [])}
    raw, seen = [], set()
    for c in required_cols:
        srcs = by_name[c].get("cols", []) if c in by_name else [c]
        for s in srcs:
            s = cat_src.get(s, s)
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
    df        = _read_csv_with_encoding_fallback(csv_path)
    # モデル入力用の複製（カテゴリエンコード/クリップ/派生特徴はここに対して行う）。
    # df 自体はユーザーの原本のまま保ち、出力CSVには原本＋予測列のみを書く(中-2)。
    df_model  = df.copy()

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

    # カテゴリカル列エンコーディング（モデル入力用複製 df_model にのみ適用）。
    # 精度レバー4/.treg v5: cat_encoders は train_bridge._prepare_categoricals /
    # _fit_target_encoders が生成したリスト形式(各要素が {"feature_name","source_col",
    # "method": "onehot"|"target", ...}) に刷新済み。onehot は生成indicator列
    # (feature_name)を新規追加し元列は残す(参照だけなら害はない。feat_colsに
    # 元列名が含まれることはない)、target は元列を数値へ置換する。
    if cat_encoders:
        applied_cols = set()
        for spec in cat_encoders:
            col = spec.get("source_col")
            if col not in df_model.columns:
                continue
            applied_cols.add(col)
            s_filled = df_model[col].fillna('__NaN__').astype(str)
            if spec.get("method") == "onehot":
                df_model[spec["feature_name"]] = (s_filled == spec["class_value"]).astype(float)
            else:  # target encoding
                m, default = spec.get("map", {}), spec.get("default", 0.0)
                df_model[col] = s_filled.map(lambda v, m=m, d=default: m.get(v, d)).astype(float)
        if applied_cols:
            print(f"[Robot] カテゴリ列エンコード: {sorted(applied_cols)}", flush=True)

    # X クリッピング（モデル入力用複製 df_model にのみ適用）
    if x_clip:
        for col, bounds in x_clip.items():
            if col in df_model.columns:
                lo, hi = bounds[0], bounds[1]
                df_model[col] = df_model[col].clip(lower=lo, upper=hi)

    # 自動特徴量（学習時レシピの再計算 — clip 後の値から生成、モデル入力専用の複製に追加）
    derived_recipe = meta.get("derived_features", []) or []
    if derived_recipe:
        df_model = _apply_derived(df_model, derived_recipe)
        print(f"[Robot] 自動特徴量 {len(derived_recipe)} 本を再計算", flush=True)

    # 欠損値補完テーブル（後方互換フォールバック — 各モデルの pkl medians を優先）
    impute_medians = {}
    impute_path = os.path.join(model_dir, "impute_medians.json")
    if os.path.exists(impute_path):
        with open(impute_path, encoding="utf-8") as f:
            impute_medians = json.load(f)

    # 学習時の特徴量列が予測CSVに欠けていないか検査（欠けていても median 補完で続行）
    # 派生特徴はソース列（CSV に実在すべき生列）に展開して検査する
    required_cols = _to_raw_required(
        _collect_required_cols(model_type, model_dir, meta), derived_recipe, cat_encoders)
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
    # 日本語Excelでヘッダ文字化けしないよう BOM付きUTF-8で保存する(中-8)。
    df.to_csv(str(out_path), index=False, encoding='utf-8-sig')
    print(f"[Robot] 保存完了: {out_path.name}", flush=True)

    # NaN/Inf を含みうる object 列も preview に入るため、列dtype単位のapply（列を数値
    # Seriesへ再強制してNoneがNaNへ戻ってしまう）ではなく、辞書化後に_sanitize_jsonで
    # 再帰的に非有限floatをNoneへ置換する（高-M2）。
    preview = df.head(500).to_dict('records')

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
    result = _sanitize_json(result)
    print(f"PREDICT_JSON:{json.dumps(result, ensure_ascii=False, default=str, allow_nan=False)}", flush=True)
    print("[Robot] 完了。", flush=True)
