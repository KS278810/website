import sys
import os

# ── BLAS/OpenMP スレッド数を numpy 等のインポート前に確定させる ──────────────
#    (UI の「CPU並列数」設定を LightGBM だけでなく GP/MLP にも効かせるため)
try:
    _NUM_JOBS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
except (ValueError, IndexError):
    _NUM_JOBS = 4
_NUM_JOBS = max(1, min(_NUM_JOBS, 32))
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env_key] = str(_NUM_JOBS)

import json
import shutil
import math
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from _light import StratifiedKFold, KFold
from _light import r2_score, mean_squared_error, mean_absolute_error

try:
    import threadpoolctl as _threadpoolctl

    def _thread_limit(n):
        return _threadpoolctl.threadpool_limits(limits=max(1, int(n)))
except Exception:
    import contextlib

    def _thread_limit(n):
        return contextlib.nullcontext()

MIN_ROWS_FOR_SPLIT = 10
GP_MAX_TRAIN       = 300   # GP 1フィットあたりの最大学習行数（超過時はランダムサブサンプル）
SCIPY_GP_MIN_ROWS  = 50
MLP_MIN_ROWS       = 30
OUTLIER_IQR_MULT   = 3.0
OUTLIER_IQR_QUICK  = 4.5
SKEW_THRESH        = 0.75
X_CLIP_PCTILE      = (1.0, 99.0)
MAX_MISS_RATE      = 0.70
POLY_MAX_ROWS      = 200   # n ≤ this → polynomial features for linear model
POLY_MAX_FEATS     = 8     # top-K features used for polynomial expansion
BLEND_R2_THRESH    = 0.30  # minimum R² to participate in blend
CONST_STD_EPS      = 1e-12
DUP_CORR_THRESH    = 0.999
SMALL_N_OOF_THRESH = 50    # この行数未満なら quick/thorough を問わず OOF 評価
Y_CLIP_MARGIN_FRAC = 0.05
SMEAR_CLIP_RANGE   = (0.5, 2.0)
LGBM_HALVING_FOLDS = 2      # thorough グリッド予選に使う fold 数
LGBM_SCREEN_MIN_FEATS = 10  # これ以下なら特徴量スクリーニングをスキップ
X_CLIP_SENTINEL    = 3.4e38  # .treg での「クリップなし」境界値 (float32 有限最大近傍)

# LightGBM early stopping 専用 holdout（高-H2: 評価fold自身をESに使わない）
ES_VAL_FRAC        = 0.10   # fold-train内から切り出すES専用valの比率
ES_SPLIT_SEED      = 20260716
ES_MIN_TRAIN_ROWS  = 20     # これ未満ならES分割せず固定本数で学習（小データでのfit不能を回避）

# Hyperparameter search for thorough mode (ランダムサーチ + 予選足切り)
LGBM_FINALISTS     = 3   # 予選(2fold)通過して全foldで本戦する候補数
MLP_N_CANDIDATES   = 8
MLP_FINALISTS      = 2
MLP_HALVING_FOLDS  = 2
LGBM_PARAM_QUICK = dict(num_leaves=31, learning_rate=0.05, n_estimators=500)
MLP_PARAM_QUICK = dict(alpha=1e-4, single_layer=True)

# Blend 採用マージン: OOF で単体最良をこの差以上上回った時のみ Blend を採用
BLEND_MARGIN = 0.005

# 自動特徴量エンジニアリング (thorough のみ)
FE_MIN_ROWS = 50   # これ未満の行数では派生特徴を作らない（過学習リスク）
FE_MAX_RAW  = 12   # ペア生成に使う生特徴の最大数（重要度上位）
FE_TOP_K    = 15   # 採用する派生特徴の最大数


def _lgbm_search_candidates(n_rows, rng_seed=42):
    """LightGBM のランダムサーチ候補。先頭は現行デフォルト（安全な基準線）。
    候補数はデータ量に応じて自動調整する。"""
    n_cand = 24
    if n_rows > 5000:
        n_cand = 14
    if n_rows > 20000:
        n_cand = 8
    rng = np.random.RandomState(rng_seed)
    cands = [dict(num_leaves=31, learning_rate=0.05, n_estimators=3000)]
    while len(cands) < n_cand:
        cands.append(dict(
            num_leaves=int(rng.choice([7, 15, 31, 63, 127])),
            learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.1])),
            n_estimators=3000,
            min_child_samples=int(rng.choice([5, 10, 20, 40])),
            colsample_bytree=float(np.round(rng.uniform(0.5, 1.0), 2)),
            subsample=float(np.round(rng.uniform(0.6, 1.0), 2)),
            reg_alpha=float(np.round(10 ** rng.uniform(-3, 0.5), 4)),
            reg_lambda=float(np.round(10 ** rng.uniform(-2, 1.5), 4)),
        ))
    return cands


def _mlp_search_candidates(rng_seed=42):
    """MLP のランダムサーチ候補。先頭3つは現行グリッド（後方互換の基準線）。"""
    rng = np.random.RandomState(rng_seed)
    cands = [dict(alpha=1e-4),
             dict(alpha=1e-2, single_layer=True),
             dict(alpha=1e-5, extra_layer=True)]
    while len(cands) < MLP_N_CANDIDATES:
        layout = rng.choice(['single', 'double', 'triple'])
        cands.append(dict(
            alpha=float(10 ** rng.uniform(-6, -1)),
            single_layer=(layout == 'single'),
            extra_layer=(layout == 'triple'),
            width=float(rng.choice([0.5, 1.0, 2.0])),
        ))
    return cands

GP_RESTARTS_THOROUGH = 3
GP_RESTARTS_QUICK    = 1


# ─── ユーティリティ ────────────────────────────────────────────────────────────

def _emit_progress(pct, label):
    """UI 進捗バー用のマイルストーン通知。lib.rs が log_data として素通しし、
    フロントが `PROGRESS:` prefix を判定してバー幅とサブラベルを更新する。"""
    pct = int(max(0, min(100, pct)))
    print(f"PROGRESS:{pct}:{label}", flush=True)


def _round_half_away(arr):
    """half-away-from-zero 丸め。C++ 側 std::round と一致させる (np.round は銀行家丸め)。"""
    arr = np.asarray(arr, dtype=float)
    return np.copysign(np.floor(np.abs(arr) + 0.5), arr)


def _sanitize_json(obj):
    """dict/list を再帰し、非有限 float を None に置換する (serde_json は NaN/Inf を拒否する)。"""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _y_true_for(eval_kind, y_raw_all, tr_idx0, va_idx0):
    """winsorize(外れ値クリップ)前の生yから、評価対象行を取り出す。
    以前は winsorize 後の df/df_train/df_val から取っており、学習側で外れ値を
    丸め込んだ後の「甘くなった正解値」に対してR²/RMSE/MAEを計算していたため、
    実データに対する精度より楽観的な数値が出ていた(性能アップ計画Phase2/評価の
    楽観バイアス低減)。y_raw_all は df と同じ行順序を保つ(winsorizeは値の書き換えのみで
    行の並べ替え・削除は行わないため、tr_idx0/va_idx0とそのまま対応する)。"""
    if eval_kind == 'oof':
        return y_raw_all
    if eval_kind == 'val' and va_idx0 is not None and len(va_idx0) > 0:
        return y_raw_all[va_idx0]
    if eval_kind == 'train':
        return y_raw_all[tr_idx0]
    return None


def _eval_metrics(val_preds, eval_kind, y_raw_all, tr_idx0, va_idx0):
    """評価指標（RMSE/MAE/eval_on/eval_rows/y_true）を計算する。eval_kind は明示指定。"""
    y_true = _y_true_for(eval_kind, y_raw_all, tr_idx0, va_idx0)
    if val_preds is None or y_true is None or len(y_true) != len(val_preds):
        return 0.0, 0.0, (eval_kind or 'train'), len(tr_idx0), None
    rmse_val = float(np.sqrt(mean_squared_error(y_true, val_preds)))
    mae_val  = float(mean_absolute_error(y_true, val_preds))
    return rmse_val, mae_val, eval_kind, len(y_true), y_true


def _candidate_r2_std(preds_arr, eval_kind, cv_splits, y_col_values):
    """高-H3緩和: 候補モデルのfold別R²のばらつき(標準偏差)を計算する（過学習/不安定性の
    診断用、candidate_models.r2_std）。OOF評価(eval_kind=='oof')かつfold分割が2以上ある
    場合のみ意味を持つ。非OOF(quickの単一train/val分割)はfold概念が1つしかなく分散を
    計算できないため0.0を返す(=「参考値なし」として扱う)。"""
    if eval_kind != "oof" or cv_splits is None or preds_arr is None:
        return 0.0
    preds_arr = np.asarray(preds_arr, dtype=float)
    if len(preds_arr) != len(y_col_values):
        return 0.0
    fold_r2s = []
    for _, va_idx in cv_splits:
        if len(va_idx) < 2:
            continue
        try:
            fold_r2s.append(float(r2_score(y_col_values[va_idx], preds_arr[va_idx])))
        except Exception:
            continue
    if len(fold_r2s) < 2:
        return 0.0
    return float(np.std(fold_r2s))


def _get_feat_cols(df, target_col):
    return [c for c in df.columns
            if c != target_col
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].isna().mean() < MAX_MISS_RATE]


def _find_constant_and_duplicate_cols(df, feat_cols):
    """分散ゼロの定数列、および相関がほぼ1の重複列を検出する。"""
    const_cols = []
    for c in feat_cols:
        s = df[c]
        if s.nunique(dropna=True) <= 1 or float(s.std(skipna=True) or 0.0) < CONST_STD_EPS:
            const_cols.append(c)
    remaining = [c for c in feat_cols if c not in const_cols]

    dup_cols = []
    kept = []
    filled = {c: df[c].fillna(df[c].median()) for c in remaining}
    for c in remaining:
        is_dup = False
        for k in kept:
            a, b = filled[c], filled[k]
            if a.std() > 0 and b.std() > 0:
                corr = a.corr(b)
                if corr is not None and abs(corr) > DUP_CORR_THRESH:
                    is_dup = True
                    break
        if is_dup:
            dup_cols.append(c)
        else:
            kept.append(c)
    return const_cols, dup_cols


# ─── LightGBM Booster ヘルパー（sklearn 非依存: lgb.train ベース） ─────────────
#     lgb.LGBMRegressor は sklearn を要求するため、native Booster API に統一する。
#     LightGBM は subsample/colsample_bytree/reg_alpha 等の sklearn 風エイリアスを
#     native 側で解決するので、パラメータ dict はほぼそのまま渡せる。

def _split_es_holdout(dtr, seed_offset=0):
    """fold-train（またはquickのdf_train）内をさらに90/10に分割し、10%をLightGBMの
    early stopping専用valとして返す。以前は評価対象のfold(OOF)やdf_val(quick)自身を
    ESのvalとしても使い回しており、LGBMだけが評価データに直接フィットして系統的に
    有利になりOOF/検証R²を楽観化させていた(高-H2)。ここで切り出す10%はbest_iteration
    の決定にのみ使い、oof_preds/最終R²の算出には一切使わない。
    Returns: (fit_df, es_df)。行数不足時は (dtr, None)（=ESなしで固定本数学習にフォールバック）。"""
    n = len(dtr)
    if n < ES_MIN_TRAIN_ROWS:
        return dtr, None
    rng = np.random.RandomState(ES_SPLIT_SEED + seed_offset)
    idx = rng.permutation(n)
    n_es = max(1, int(round(n * ES_VAL_FRAC)))
    if n - n_es < 10:
        return dtr, None
    es_idx, fit_idx = idx[:n_es], idx[n_es:]
    return dtr.iloc[fit_idx], dtr.iloc[es_idx]


def _lgb_fit(sk_params, X, y, X_val=None, y_val=None, early_stopping=0):
    """lgb.train で Booster を学習して返す。n_estimators は num_boost_round に振り替える。"""
    import lightgbm as lgb
    params = dict(sk_params)
    params.setdefault('objective', 'regression')
    n_round = int(params.pop('n_estimators', 100))
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    valid_sets = None
    callbacks = [lgb.log_evaluation(-1)]
    if X_val is not None and early_stopping > 0:
        valid_sets = [lgb.Dataset(X_val, label=y_val, reference=ds)]
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    return lgb.train(params, ds, num_boost_round=n_round,
                     valid_sets=valid_sets, callbacks=callbacks)


def _lgb_importance(bst, n_features):
    """Booster の split 重要度（LGBMRegressor.feature_importances_ 既定と同じ）。"""
    imp = np.asarray(bst.feature_importance(importance_type='split'), dtype=float)
    if len(imp) < n_features:
        imp = np.concatenate([imp, np.zeros(n_features - len(imp))])
    return imp


def _lgbm_feature_screen(df_train, target_col, num_jobs=4):
    """軽量 LightGBM で重要度ゼロの特徴量を除外し、GP/MLP の次元を削減する。"""
    feat_cols = _get_feat_cols(df_train, target_col)
    if len(feat_cols) <= LGBM_SCREEN_MIN_FEATS:
        return feat_cols
    try:
        medians = df_train[feat_cols].median()
        X = df_train[feat_cols].fillna(medians).values
        y = df_train[target_col].values
        bst = _lgb_fit(dict(
            n_estimators=200, num_leaves=31, learning_rate=0.1, verbosity=-1,
            n_jobs=num_jobs, force_col_wise=True,
            min_child_samples=max(3, len(df_train) // 30)), X, y)
        imps = _lgb_importance(bst, len(feat_cols))
        keep = [feat_cols[i] for i in range(len(feat_cols)) if imps[i] > 0]
        if len(keep) < 2:
            return feat_cols
        if len(keep) < len(feat_cols):
            print(f"[Screen] LGBM probe: {len(feat_cols)}→{len(keep)} 列に絞込み (GP/MLP用)", flush=True)
        return keep
    except Exception as e:
        print(f"[Screen] 特徴量スクリーニング失敗 → 全列使用: {e}", flush=True)
        return feat_cols


# ─── 自動特徴量エンジニアリング（thorough のみ） ────────────────────────────────

def _build_derived_recipe(df_train, target_col, num_jobs=4):
    """ペア積・二乗・符号の派生特徴候補を生成し、LGBM 重要度で上位を選抜する。
    選抜は既存の特徴量スクリーニングと同様に fold0 の学習側で行う。
    Returns: [{"name", "op", "cols"}]  (op: mul / sq / sign)"""
    feat_cols = _get_feat_cols(df_train, target_col)
    if len(feat_cols) < 2:
        return []
    try:
        medians = df_train[feat_cols].median()
        X = df_train[feat_cols].fillna(medians)
        y = df_train[target_col].values

        base_cols = feat_cols
        if len(feat_cols) > FE_MAX_RAW:
            probe = _lgb_fit(dict(n_estimators=150, num_leaves=31, learning_rate=0.1,
                                  verbosity=-1, n_jobs=num_jobs, force_col_wise=True), X.values, y)
            order = np.argsort(_lgb_importance(probe, len(feat_cols)))[::-1]
            base_cols = [feat_cols[i] for i in order[:FE_MAX_RAW]]

        existing = set(df_train.columns)
        cand_specs = []
        cand_frames = {}

        def _add(name, op, cols, values):
            if name in existing or name in cand_frames:
                return
            v = np.asarray(values, dtype=float)
            if not np.isfinite(v).all() or v.std() < 1e-12:
                return
            cand_specs.append({"name": name, "op": op, "cols": cols})
            cand_frames[name] = v

        for i, a in enumerate(base_cols):
            va = X[a].values
            for b in base_cols[i + 1:]:
                _add(f"{a}*{b}", "mul", [a, b], va * X[b].values)
            _add(f"{a}^2", "sq", [a], va * va)
            if (va > 0).any() and (va < 0).any():
                _add(f"sign({a})", "sign", [a], np.sign(va))
        if not cand_specs:
            return []

        cand_df = pd.DataFrame(cand_frames, index=df_train.index)
        X_aug = pd.concat([X, cand_df], axis=1)
        scr = _lgb_fit(dict(n_estimators=200, num_leaves=31, learning_rate=0.05,
                            verbosity=-1, n_jobs=num_jobs, force_col_wise=True,
                            min_child_samples=max(3, len(df_train) // 30)), X_aug.values, y)
        imp = pd.Series(_lgb_importance(scr, X_aug.shape[1]), index=list(X_aug.columns))
        cand_imp = imp[[s["name"] for s in cand_specs]].sort_values(ascending=False)
        keep = set(cand_imp[cand_imp > 0].head(FE_TOP_K).index)
        recipe = [s for s in cand_specs if s["name"] in keep]
        if recipe:
            names = [r["name"] for r in recipe]
            print(f"[FE] 自動特徴量 {len(recipe)} 本を採用: "
                  f"{', '.join(names[:5])}{' …' if len(names) > 5 else ''}", flush=True)
        return recipe
    except Exception as e:
        print(f"[FE] 自動特徴量生成失敗 → スキップ: {e}", flush=True)
        return []


def _apply_derived(df, recipe):
    """派生特徴レシピを DataFrame に適用する。
    ソース欠損・非有限は NaN として伝播し、後段の median 補完に委ねる。"""
    if not recipe:
        return df
    df = df.copy()
    nan_series = pd.Series(np.nan, index=df.index)
    for r in recipe:
        cols = r["cols"]
        a = pd.to_numeric(df[cols[0]], errors="coerce") if cols[0] in df.columns else nan_series
        if r["op"] == "mul":
            b = pd.to_numeric(df[cols[1]], errors="coerce") if len(cols) > 1 and cols[1] in df.columns else nan_series
            v = (a * b).values.astype(float)
        elif r["op"] == "sq":
            v = (a * a).values.astype(float)
        elif r["op"] == "sign":
            v = np.sign(a.values.astype(float))
        else:
            continue
        df[r["name"]] = np.where(np.isfinite(v), v, np.nan)
    return df


def _r2_interpretation(r2):
    if r2 >= 0.95:
        return "非常に高精度"
    elif r2 >= 0.85:
        return "高精度（実用レベル）"
    elif r2 >= 0.70:
        return "実用的な精度"
    elif r2 >= 0.50:
        return "精度はやや低め"
    elif r2 >= 0.0:
        return "精度不足"
    else:
        return "モデルが機能していません"


def _clean_model_files(model_dir, keep_type):
    all_files = {
        'linear': ['linear_model.pkl'],
        'lgbm':   ['lgbm_model.txt', 'lgbm_meta.json'],
        'gp':     ['gp_model.pkl'],
        'mlp':    ['mlp_model.pkl'],
        'rf':     ['rf_model.txt', 'rf_meta.json'],
        'xt':     ['xt_model.txt', 'xt_meta.json'],
    }
    for mtype, fnames in all_files.items():
        if mtype != keep_type:
            for fname in fnames:
                p = os.path.join(model_dir, fname)
                if os.path.exists(p):
                    os.remove(p)


# ─── CSV 読み込み（日本語Excel既定のShift-JISフォールバック） ───────────────────

def _read_csv_with_encoding_fallback(csv_path):
    """UTF-8として妥当かをまずバイト列で検査し、無効なら Shift-JIS(cp932) として読む。
    日本語Excelが既定で書き出すShift-JIS CSVがUTF-8として「�」化けしたまま
    サイレントに学習が完走してしまうのを防ぐ(中-7)。"""
    with open(csv_path, 'rb') as f:
        raw = f.read()
    try:
        raw.decode('utf-8')
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"[Python] CSVがUTF-8として不正 → Shift-JIS(cp932)として読み込みます", flush=True)
        df = pd.read_csv(csv_path, encoding='cp932')
    # 列選択UI（frontend/index.html・lib.rs）はヘッダの前後空白をtrimして送信するため、
    # pandas側の列名もtrimして揃える。非対称のままだと「表示された列を選んだのに
    # ターゲット列が存在しません」エラーになる(中-M7)。
    df.columns = df.columns.str.strip()
    return df


# ─── ターゲット列の解決と検証 ──────────────────────────────────────────────────

def _resolve_and_validate_target(df, target_column_arg):
    """target 列の存在・数値性を検証し、NaN 行を除去する。
    Returns: (df, target_column, n_target_na)。エラーは print + exit(1)。"""
    if target_column_arg:
        if target_column_arg not in df.columns:
            print(f"ERROR: 指定されたターゲット列「{target_column_arg}」がCSVに存在しません", flush=True)
            sys.exit(1)
        target_column = target_column_arg
        print(f"[Python] ターゲット: 「{target_column}」", flush=True)
    else:
        target_column = df.columns[-1]
        print(f"[Python] ターゲット自動判定: 「{target_column}」", flush=True)

    col = df[target_column]
    if not pd.api.types.is_numeric_dtype(col):
        coerced = pd.to_numeric(col, errors='coerce')
        # 数値化で新たに NaN になった値 = 非数値の混入
        bad_mask = coerced.isna() & col.notna()
        if bad_mask.any():
            example = str(col[bad_mask].iloc[0])
            print(f"ERROR: ターゲット列「{target_column}」に数値でない値が含まれています（例: {example}）", flush=True)
            sys.exit(1)
        df = df.copy()
        df[target_column] = coerced

    n_target_na = int(df[target_column].isna().sum())
    if n_target_na > 0:
        df = df.dropna(subset=[target_column]).reset_index(drop=True)
        print(f"[Python] ターゲット欠損 {n_target_na} 行を除外", flush=True)
    if len(df) == 0:
        print("ERROR: 目的変数に有効な値がある行がありません", flush=True)
        sys.exit(1)
    return df, target_column, n_target_na


# ─── Y 変換 ──────────────────────────────────────────────────────────────────

def _detect_y_transform(y: np.ndarray, y_full: np.ndarray = None):
    """skew判定は y（fold0-train等の部分集合で可）で行うが、log1p適用可否のmin>=0判定は
    y_full（全行）で行う。負値行が別foldに落ちても log1p(負値)=NaN が学習ターゲットに
    混入するのを防ぐため。y_full省略時はyそのものを使う（後方互換）。"""
    if y_full is None:
        y_full = y
    from _light import skew as scipy_skew
    sk = float(scipy_skew(y))
    print(f"[YTransform] Y skewness={sk:.3f}", flush=True)
    if sk > SKEW_THRESH and float(y_full.min()) >= 0:
        print(f"[YTransform] log1p 変換を適用（skewness={sk:.2f} > {SKEW_THRESH}, min≥0）", flush=True)
        return 'log1p', {}
    if abs(sk) > SKEW_THRESH:
        try:
            from _light import PowerTransformer
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            pt.fit(y.reshape(-1, 1))
            lam = float(pt.lambdas_[0])
            print(f"[YTransform] Yeo-Johnson 変換を適用（skewness={sk:.2f}, λ={lam:.3f}）", flush=True)
            return 'yeo_johnson', {'lambda': lam}
        except Exception as e:
            print(f"[YTransform] Yeo-Johnson 失敗 → 変換なし: {e}", flush=True)
    return 'none', {}


def _apply_y_transform(y: np.ndarray, transform: str, params: dict):
    if transform == 'log1p':
        return np.log1p(y)
    if transform == 'yeo_johnson':
        from _light import PowerTransformer
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        pt.lambdas_ = np.array([params['lambda']])
        return pt.transform(y.reshape(-1, 1)).ravel()
    return y.copy()


def _invert_y_transform(y: np.ndarray, transform: str, params: dict):
    if transform == 'log1p':
        return np.expm1(y)
    if transform == 'yeo_johnson':
        from _light import PowerTransformer
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        pt.lambdas_ = np.array([params['lambda']])
        return pt.inverse_transform(y.reshape(-1, 1)).ravel()
    return y.copy()


# ─── 予測後処理（観測レンジクリップ / log1p smearing 補正 / 整数丸め） ─────────

def _fit_postprocess_params(preds, y_true, y_transform, y_raw_all, target_is_integer):
    """y_raw_all は winsorize 前の生 y（クリップ範囲が学習用加工に狭められないように）。"""
    y_raw_all = np.asarray(y_raw_all, dtype=float)
    y_min, y_max = float(np.min(y_raw_all)), float(np.max(y_raw_all))
    span = y_max - y_min
    margin = span * Y_CLIP_MARGIN_FRAC if span > 0 else max(abs(y_max), 1.0)
    y_clip_lo, y_clip_hi = y_min - margin, y_max + margin

    smear = 1.0
    if y_transform == 'log1p' and preds is not None and y_true is not None:
        preds_a = np.asarray(preds, dtype=float)
        true_a  = np.asarray(y_true, dtype=float)
        if len(preds_a) == len(true_a) and len(preds_a) > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = true_a / np.clip(preds_a, 1e-6, None)
            ratio = ratio[np.isfinite(ratio)]
            if len(ratio) > 0:
                smear = float(np.clip(np.median(ratio), SMEAR_CLIP_RANGE[0], SMEAR_CLIP_RANGE[1]))
    return smear, y_clip_lo, y_clip_hi, bool(target_is_integer)


def _apply_postprocess(preds, smear, y_clip_lo, y_clip_hi, round_output):
    preds = np.asarray(preds, dtype=float) * smear
    preds = np.clip(preds, y_clip_lo, y_clip_hi)
    if round_output:
        preds = _round_half_away(preds)
    return preds


# ─── 外れ値 winsorize（クリップ、行は保持） ────────────────────────────────────

def _fit_y_winsorize_bounds(y_train: np.ndarray, mult: float):
    q1, q3 = np.percentile(y_train, 25), np.percentile(y_train, 75)
    iqr = q3 - q1
    return q1 - mult * iqr, q3 + mult * iqr


def _apply_y_winsorize(df: pd.DataFrame, target_col: str, lo: float, hi: float):
    y = df[target_col].values
    n_clipped = int(((y < lo) | (y > hi)).sum())
    if n_clipped > 0:
        df = df.copy()
        df[target_col] = np.clip(y, lo, hi)
    return df, n_clipped


# ─── カテゴリカル列エンコーディング ──────────────────────────────────────────

def _encode_categoricals(df, target_col):
    cat_cols = [c for c in df.columns
                if c != target_col
                and (df[c].dtype == object or str(df[c].dtype) in ('bool', 'boolean'))]
    encoders = {}
    for col in cat_cols:
        classes = sorted(df[col].fillna('__NaN__').astype(str).unique().tolist())
        encoders[col] = classes
        print(f"[CatEnc] {col}: {len(classes)} クラス", flush=True)
    return encoders


def _apply_cat_encoders(df, encoders):
    df = df.copy()
    for col, classes in encoders.items():
        if col not in df.columns:
            continue
        class_to_idx = {c: i for i, c in enumerate(classes)}
        df[col] = df[col].fillna('__NaN__').astype(str).map(
            lambda x, m=class_to_idx: m.get(x, 0))
    return df


# ─── X クリッピング ────────────────────────────────────────────────────────

def _compute_x_clip(df, feat_cols):
    lo_p, hi_p = X_CLIP_PCTILE
    bounds = {}
    for col in feat_cols:
        lo = float(df[col].quantile(lo_p / 100.0))
        hi = float(df[col].quantile(hi_p / 100.0))
        if lo < hi:
            bounds[col] = [lo, hi]
    return bounds


def _apply_x_clip(df, bounds):
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ─── Stratified K-Fold（実現可能な fold 数に自動キャップ） ─────────────────────

def _make_binned_splits(df: pd.DataFrame, target_col: str, n_splits: int = 5, seed: int = 42):
    n = len(df)
    n_splits = max(2, min(n_splits, n // 2))  # 各foldに最低2行
    n_bins = min(n_splits, max(2, n // 10))
    y = df[target_col].values
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        bins = pd.Series(bins).fillna(0).astype(int).values
    except Exception:
        bins = pd.cut(pd.Series(y), bins=n_bins, labels=False).fillna(0).astype(int).values
    counts = np.bincount(bins)
    min_count = int(counts[counts > 0].min()) if (counts > 0).any() else 0
    eff_splits = min(n_splits, min_count)
    if eff_splits >= 2:
        skf = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(n), bins))
    kf = KFold(n_splits=max(2, min(n_splits, n)), shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(n)))


def _fold_frame(df_full, df_all_per_fold, fold_idx, tr_idx, target_col, base_feat_cols,
                screen_cols_per_fold=None):
    """高-H1: OOF fold毎に「そのfoldの検証行を一切見ずに」fitしたFE/スクリーニング済み
    データと特徴列を返す。df_all_per_fold が None の場合は従来通り共有 df_full/base_feat_cols
    を使う(FE/screeningがそもそも無効な経路、または非OOF経路との後方互換)。
    screen_cols_per_fold が与えられる場合(GP/MLP)は特徴列をそのfold専用のスクリーニング結果に
    差し替える。それ以外(Linear/LGBM/RF/XT)は df_fold から都度 _get_feat_cols で再計算する
    (fold毎にFE由来の派生列セットが異なり得るため)。"""
    if df_all_per_fold is None:
        return df_full, base_feat_cols
    df_fold = df_all_per_fold[fold_idx]
    if screen_cols_per_fold is not None:
        return df_fold, screen_cols_per_fold[fold_idx]
    return df_fold, _get_feat_cols(df_fold.iloc[tr_idx], target_col)


# ─── 1. Linear (Ridge CV + Polynomial for small data) ─────────────────────────
# Returns: (r2, feat_list, 'linear', preds, info)

def _try_linear(df_train, df_val, target_col, model_dir, y_transform='none', y_params={},
                df_all=None, use_oof=False, splits=None, df_all_per_fold=None):
    try:
        import pickle
        from _light import RidgeCV, RobustScaler, PolynomialFeatures

        feat_cols = _get_feat_cols(df_train, target_col)
        if not feat_cols:
            return None, [], None, None, None

        medians = df_train[feat_cols].median()
        use_poly = (len(df_train) <= POLY_MAX_ROWS and len(feat_cols) <= POLY_MAX_FEATS)
        do_kfold = (use_oof and df_all is not None and splits is not None)
        alphas_std  = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        alphas_poly = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]

        # ── K-Fold OOF (非poly) ────────────────────────────────────────────
        if do_kfold and not use_poly:
            df_full = df_all
            n_splits = len(splits)
            oof_preds = np.zeros(len(df_full))
            medians_full = df_full[feat_cols].median()
            print(f"[Linear] K-Fold OOF ({n_splits} fold)...", flush=True)
            for fold, (tr_idx, va_idx) in enumerate(splits):
                # 高-H1: fold毎に(そのfoldの検証行を見ずに)fitしたFE済みデータ・特徴列を使う
                df_fold, feat_cols_f = _fold_frame(df_full, df_all_per_fold, fold, tr_idx, target_col, feat_cols)
                dtr = df_fold.iloc[tr_idx]
                dva = df_fold.iloc[va_idx]
                med_f = dtr[feat_cols_f].median()
                X_f  = dtr[feat_cols_f].fillna(med_f).values
                y_f  = _apply_y_transform(dtr[target_col].values, y_transform, y_params)
                X_v  = dva[feat_cols_f].fillna(med_f).values
                sc = RobustScaler()
                m  = RidgeCV(alphas=alphas_std)
                m.fit(sc.fit_transform(X_f), y_f)
                preds_t = m.predict(sc.transform(X_v))
                oof_preds[va_idx] = _invert_y_transform(preds_t, y_transform, y_params)

            oof_r2 = float(r2_score(df_full[target_col].values, oof_preds))
            print(f"[Linear] OOF R²={oof_r2:.4f}", flush=True)

            X_all = df_full[feat_cols].fillna(medians_full).values
            y_all = _apply_y_transform(df_full[target_col].values, y_transform, y_params)
            final_sc = RobustScaler()
            final_m  = RidgeCV(alphas=alphas_std)
            final_m.fit(final_sc.fit_transform(X_all), y_all)

            # 高-H3緩和: 過学習診断用に、最終モデル(100%データでfit)自身の学習データに対する
            # R²(train_r2)も記録する。OOF R²との乖離が大きいほど過学習が疑われる。
            train_preds_t = final_m.predict(final_sc.transform(X_all))
            train_preds = _invert_y_transform(train_preds_t, y_transform, y_params)
            train_r2 = float(r2_score(df_full[target_col].values, train_preds))

            coefs = np.abs(final_m.coef_)
            total = max(coefs.sum(), 1e-9)
            feat_list = sorted(
                [{"name": feat_cols[i], "pct": round(float(coefs[i] / total * 100), 1)}
                 for i in range(len(feat_cols))],
                key=lambda x: x["pct"], reverse=True)[:10]

            med_dict = {c: float(medians_full[c]) if not np.isnan(float(medians_full[c])) else 0.0
                        for c in feat_cols}
            with open(os.path.join(model_dir, "linear_model.pkl"), "wb") as f:
                pickle.dump({"model": final_m, "scaler": final_sc,
                             "feat_cols": feat_cols, "target_col": target_col,
                             "use_poly": False, "medians": med_dict}, f)
            print(f"[Linear] alpha={final_m.alpha_:.4g}", flush=True)
            info = {"eval_kind": "oof", "used_cols": feat_cols, "medians": med_dict, "exportable": True,
                    "train_r2": round(train_r2, 4)}
            return round(oof_r2, 4), feat_list, "linear", oof_preds, info

        # ── K-Fold OOF (poly): top_feats は df_full 全体で1回固定 ────────────
        #    既知の限界(高-H1の対象外・タスク指示の対象は_build_derived_recipeと
        #    _lgbm_feature_screenのみ): poly-Ridgeの top_feats 選定(分散/相関ベース)は
        #    ここでは fold-local 化していない。FE有効時は特徴数が POLY_MAX_FEATS(8) を
        #    超えて非poly分岐に流れることが大半のため実害は小さいが、raw特徴が極端に
        #    少ないデータでは fold0-train 由来の選定バイアスが残る可能性がある。
        if do_kfold and use_poly:
            df_full = df_all
            n_splits = len(splits)
            medians_full = df_full[feat_cols].median()
            X_imp = df_full[feat_cols].fillna(medians_full).values
            variances = np.nanvar(X_imp, axis=0)
            top_k = min(len(feat_cols), POLY_MAX_FEATS)
            top_idx = np.argsort(variances)[::-1][:top_k]
            top_feats = [feat_cols[i] for i in sorted(top_idx)]
            med_top = df_full[top_feats].median()

            oof_preds = np.zeros(len(df_full))
            print(f"[Linear] poly K-Fold OOF ({n_splits} fold)...", flush=True)
            for fold, (tr_idx, va_idx) in enumerate(splits):
                dtr = df_full.iloc[tr_idx]
                dva = df_full.iloc[va_idx]
                med_f = dtr[top_feats].median()
                X_f = dtr[top_feats].fillna(med_f).values
                y_f = _apply_y_transform(dtr[target_col].values, y_transform, y_params)
                X_v = dva[top_feats].fillna(med_f).values
                sc = RobustScaler()
                po = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
                m  = RidgeCV(alphas=alphas_poly)
                m.fit(po.fit_transform(sc.fit_transform(X_f)), y_f)
                preds_t = m.predict(po.transform(sc.transform(X_v)))
                oof_preds[va_idx] = _invert_y_transform(preds_t, y_transform, y_params)

            oof_r2 = float(r2_score(df_full[target_col].values, oof_preds))
            print(f"[Linear] poly OOF R²={oof_r2:.4f}", flush=True)

            X_all = df_full[top_feats].fillna(med_top).values
            y_all = _apply_y_transform(df_full[target_col].values, y_transform, y_params)
            scaler = RobustScaler()
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            model = RidgeCV(alphas=alphas_poly)
            model.fit(poly.fit_transform(scaler.fit_transform(X_all)), y_all)

            # 高-H3緩和: 過学習診断用のtrain_r2(最終モデル自身の学習データに対するR²)
            train_preds_t = model.predict(poly.transform(scaler.transform(X_all)))
            train_preds = _invert_y_transform(train_preds_t, y_transform, y_params)
            train_r2 = float(r2_score(df_full[target_col].values, train_preds))

            feat_names = poly.get_feature_names_out(top_feats)
            coefs = np.abs(model.coef_)
            total = max(coefs.sum(), 1e-9)
            feat_list = sorted(
                [{"name": str(feat_names[i]), "pct": round(float(coefs[i] / total * 100), 1)}
                 for i in range(len(feat_names))],
                key=lambda x: x["pct"], reverse=True)[:10]

            med_dict = {c: float(med_top[c]) if not np.isnan(float(med_top[c])) else 0.0
                        for c in top_feats}
            with open(os.path.join(model_dir, "linear_model.pkl"), "wb") as f:
                pickle.dump({"model": model, "scaler": scaler, "poly": poly,
                             "feat_cols": top_feats, "target_col": target_col,
                             "use_poly": True, "medians": med_dict}, f)
            print(f"[Linear] poly alpha={model.alpha_:.3g}", flush=True)
            # poly-Ridge も 'linear_poly' 専用フォーマットで .treg 書き出し可能
            info = {"eval_kind": "oof", "used_cols": top_feats, "medians": med_dict, "exportable": True,
                    "train_r2": round(train_r2, 4)}
            return round(oof_r2, 4), feat_list, "linear", oof_preds, info

        # ── 単一 train/val ────────────────────────────────────────────────
        X_tr = df_train[feat_cols].fillna(medians).values
        y_tr = _apply_y_transform(df_train[target_col].values, y_transform, y_params)
        eval_df = df_val if df_val is not None else df_train
        eval_kind = "val" if df_val is not None else "train"

        if use_poly:
            variances = np.nanvar(X_tr, axis=0)
            top_k = min(len(feat_cols), POLY_MAX_FEATS)
            top_idx = np.argsort(variances)[::-1][:top_k]
            top_feats = [feat_cols[i] for i in sorted(top_idx)]
            X_top = df_train[top_feats].fillna(medians[top_feats]).values

            scaler = RobustScaler()
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            X_tr_s = poly.fit_transform(scaler.fit_transform(X_top))
            model = RidgeCV(alphas=alphas_poly)
            model.fit(X_tr_s, y_tr)

            X_ev = eval_df[top_feats].fillna(medians[top_feats]).values
            y_pred_t = model.predict(poly.transform(scaler.transform(X_ev)))
            y_pred = _invert_y_transform(y_pred_t, y_transform, y_params)
            lin_r2 = float(r2_score(eval_df[target_col].values, y_pred))

            # 高-H3緩和: 過学習診断用train_r2(このモデル自身の学習データdf_trainに対するR²)
            if eval_kind == "train":
                train_r2 = lin_r2  # df_val無し: eval_df==df_train のためそのまま流用
            else:
                train_pred_t = model.predict(poly.transform(scaler.transform(X_top)))
                train_r2 = float(r2_score(df_train[target_col].values,
                                          _invert_y_transform(train_pred_t, y_transform, y_params)))

            feat_names = poly.get_feature_names_out(top_feats)
            coefs = np.abs(model.coef_)
            total = max(coefs.sum(), 1e-9)
            feat_list = sorted(
                [{"name": str(feat_names[i]), "pct": round(float(coefs[i] / total * 100), 1)}
                 for i in range(len(feat_names))],
                key=lambda x: x["pct"], reverse=True)[:10]

            med_dict = {c: float(medians[c]) if not np.isnan(float(medians[c])) else 0.0
                        for c in top_feats}
            with open(os.path.join(model_dir, "linear_model.pkl"), "wb") as f:
                pickle.dump({"model": model, "scaler": scaler, "poly": poly,
                             "feat_cols": top_feats, "target_col": target_col,
                             "use_poly": True, "medians": med_dict}, f)
            n_feats_poly = X_tr_s.shape[1]
            print(f"[Linear] poly R²={lin_r2:.4f}  α={model.alpha_:.3g}  poly_feats={n_feats_poly}", flush=True)
            # poly-Ridge も 'linear_poly' 専用フォーマットで .treg 書き出し可能
            info = {"eval_kind": eval_kind, "used_cols": top_feats, "medians": med_dict, "exportable": True,
                    "train_r2": round(train_r2, 4)}
            return round(lin_r2, 4), feat_list, "linear", y_pred, info

        # Standard Ridge (no poly)
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        model = RidgeCV(alphas=alphas_std)
        model.fit(X_tr_s, y_tr)

        X_val_s = scaler.transform(eval_df[feat_cols].fillna(medians).values)
        y_pred_t = model.predict(X_val_s)
        y_pred = _invert_y_transform(y_pred_t, y_transform, y_params)
        lin_r2 = float(r2_score(eval_df[target_col].values, y_pred))

        # 高-H3緩和: 過学習診断用train_r2(このモデル自身の学習データdf_trainに対するR²)
        if eval_kind == "train":
            train_r2 = lin_r2  # df_val無し: eval_df==df_train のためそのまま流用
        else:
            train_pred_t = model.predict(X_tr_s)
            train_r2 = float(r2_score(df_train[target_col].values,
                                      _invert_y_transform(train_pred_t, y_transform, y_params)))

        coefs = np.abs(model.coef_)
        total = max(coefs.sum(), 1e-9)
        feat_list = sorted(
            [{"name": feat_cols[i], "pct": round(float(coefs[i] / total * 100), 1)}
             for i in range(len(feat_cols))],
            key=lambda x: x["pct"], reverse=True)[:10]

        med_dict = {c: float(medians[c]) if not np.isnan(float(medians[c])) else 0.0
                    for c in feat_cols}
        with open(os.path.join(model_dir, "linear_model.pkl"), "wb") as f:
            pickle.dump({"model": model, "scaler": scaler,
                         "feat_cols": feat_cols, "target_col": target_col,
                         "use_poly": False, "medians": med_dict}, f)
        print(f"[Linear] R²={lin_r2:.4f}  alpha={model.alpha_:.4g}", flush=True)
        info = {"eval_kind": eval_kind, "used_cols": feat_cols, "medians": med_dict, "exportable": True,
                "train_r2": round(train_r2, 4)}
        return round(lin_r2, 4), feat_list, "linear", y_pred, info

    except Exception as e:
        print(f"[Linear] 失敗: {e}", flush=True)
        return None, [], None, None, None


# ─── 2. LightGBM ──────────────────────────────────────────────────────────────
# Returns: (r2, feat_list, 'lgbm', preds, info)

def _try_lgbm(df_train, df_val, target_col, model_dir, use_grid=False, use_oof=False,
              y_transform='none', y_params={}, df_all=None, num_jobs=4, splits=None, prog=None,
              df_all_per_fold=None):
    try:
        import lightgbm as lgb

        feat_cols = _get_feat_cols(df_train, target_col)
        if not feat_cols:
            return None, [], None, None, None

        n = len(df_train)
        base_params = dict(
            objective='regression',
            metric='rmse',
            verbosity=-1,
            n_jobs=num_jobs,
            force_col_wise=True,
            deterministic=True,
            min_child_samples= max(3, n // 30),
            subsample        = 0.8,
            subsample_freq   = 1,
            colsample_bytree = 0.8,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
        )

        def _save_sidecar(medians_dict):
            with open(os.path.join(model_dir, "lgbm_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"feat_cols": feat_cols, "medians": medians_dict},
                          f, ensure_ascii=False)

        if use_oof and df_all is not None and splits is not None:
            df_full = df_all
            n_splits = len(splits)
            param_grid = _lgbm_search_candidates(len(df_full)) if use_grid else [LGBM_PARAM_QUICK]

            def _run_folds(param_override, fold_idxs):
                params = dict(base_params)
                params.update(param_override)
                oof_preds_local = np.zeros(len(df_full))
                touched = np.zeros(len(df_full), dtype=bool)
                best_iters_local = []
                for fold_idx in fold_idxs:
                    tr_idx, va_idx = splits[fold_idx]
                    # 高-H1: fold毎に(そのfoldの検証行を見ずに)fitしたFE済みデータ・特徴列を使う
                    df_fold, feat_cols_f = _fold_frame(df_full, df_all_per_fold, fold_idx, tr_idx,
                                                       target_col, feat_cols)
                    dtr = df_fold.iloc[tr_idx]
                    dva = df_fold.iloc[va_idx]
                    med_f = dtr[feat_cols_f].median()
                    # 高-H2: early stoppingは評価fold(dva)自身ではなくfold-train内の
                    # 専用holdout(90/10)で行う。dvaはOOF評価にのみ使い、ESには触れさせない。
                    dtr_fit, dtr_es = _split_es_holdout(dtr, seed_offset=fold_idx)
                    X_f = dtr_fit[feat_cols_f].fillna(med_f).values
                    y_f = _apply_y_transform(dtr_fit[target_col].values, y_transform, y_params)
                    X_v = dva[feat_cols_f].fillna(med_f).values
                    if dtr_es is not None:
                        X_es = dtr_es[feat_cols_f].fillna(med_f).values
                        y_es_t = _apply_y_transform(dtr_es[target_col].values, y_transform, y_params)
                        bst = _lgb_fit(params, X_f, y_f, X_es, y_es_t, early_stopping=50)
                    else:
                        bst = _lgb_fit(params, X_f, y_f)
                    preds_t = bst.predict(X_v)
                    oof_preds_local[va_idx] = _invert_y_transform(preds_t, y_transform, y_params)
                    touched[va_idx] = True
                    best_iters_local.append(bst.best_iteration or int(params.get('n_estimators', 100)))
                return oof_preds_local, touched, best_iters_local

            # ── successive halving: 予選(2fold, 全候補) → 本戦(全fold, 上位のみ) ──
            if use_grid and len(param_grid) > 1 and n_splits > LGBM_HALVING_FOLDS:
                print(f"[LightGBM] ランダムサーチ {len(param_grid)} 候補を予選 ({LGBM_HALVING_FOLDS}fold)...", flush=True)
                prelim_scores = []
                n_cand = len(param_grid)
                for pidx, param_override in enumerate(param_grid, 1):
                    if prog is not None:
                        lo, hi = prog
                        _emit_progress(lo + (hi - lo) * 0.8 * pidx / n_cand,
                                       f"最適なパラメータを探索中 {pidx}/{n_cand}")
                    oof_p, touched, _ = _run_folds(param_override, range(LGBM_HALVING_FOLDS))
                    r2_p = float(r2_score(df_full[target_col].values[touched], oof_p[touched]))
                    print(f"  予選{pidx}/{len(param_grid)}: R²={r2_p:.4f} "
                          f"(leaves={param_override['num_leaves']}, lr={param_override['learning_rate']})", flush=True)
                    prelim_scores.append(r2_p if np.isfinite(r2_p) else -np.inf)
                top_idx = np.argsort(prelim_scores)[::-1][:LGBM_FINALISTS]
                param_grid_final = [param_grid[i] for i in top_idx]
                print(f"[LightGBM] 予選通過 {len(param_grid_final)} 候補 → 全{n_splits}foldで本戦", flush=True)
            else:
                param_grid_final = param_grid

            best_param_r2 = -np.inf
            best_param_set = None
            for pidx, param_override in enumerate(param_grid_final, 1):
                if use_grid:
                    print(f"[LightGBM] 候補{pidx}/{len(param_grid_final)}: num_leaves={param_override['num_leaves']}, "
                          f"lr={param_override['learning_rate']}, n={param_override['n_estimators']}", flush=True)
                oof_preds, _, best_iters = _run_folds(param_override, range(n_splits))
                oof_r2 = float(r2_score(df_full[target_col].values, oof_preds))
                if use_grid:
                    print(f"  OOF R²={oof_r2:.4f}  avg_iter={int(np.mean(best_iters))}", flush=True)
                if oof_r2 > best_param_r2:
                    best_param_r2 = oof_r2
                    best_param_set = (dict(base_params, **param_override), best_iters, oof_preds, oof_r2)

            params, best_iters, oof_preds, oof_r2 = best_param_set
            print(f"[LightGBM] 最良パラメータ R²={oof_r2:.4f}", flush=True)

            best_n = max(50, int(np.mean(best_iters) * 1.05))
            final_p = dict(params)
            final_p['n_estimators'] = best_n
            medians_full = df_full[feat_cols].median()
            X_all = df_full[feat_cols].fillna(medians_full).values
            y_all = _apply_y_transform(df_full[target_col].values, y_transform, y_params)
            final_bst = _lgb_fit(final_p, X_all, y_all)
            final_bst.save_model(os.path.join(model_dir, "lgbm_model.txt"))

            # 高-H3緩和: 過学習診断用train_r2(最終モデル自身の学習データに対するR²)
            train_preds_t = final_bst.predict(X_all)
            train_r2 = float(r2_score(df_full[target_col].values,
                                      _invert_y_transform(train_preds_t, y_transform, y_params)))

            med_dict = {c: (float(medians_full[c]) if not np.isnan(float(medians_full[c])) else 0.0)
                        for c in feat_cols}
            _save_sidecar(med_dict)

            imps = _lgb_importance(final_bst, len(feat_cols))
            total = max(imps.sum(), 1.0)
            feat_list = sorted(
                [{"name": feat_cols[i], "pct": round(float(imps[i] / total * 100), 1)}
                 for i in range(len(feat_cols))],
                key=lambda x: x["pct"], reverse=True)[:10]
            info = {"eval_kind": "oof", "used_cols": feat_cols, "medians": med_dict, "exportable": True,
                    "train_r2": round(train_r2, 4)}
            return round(oof_r2, 4), feat_list, "lgbm", oof_preds, info

        # Quick: single split
        medians = df_train[feat_cols].median()
        X_tr = df_train[feat_cols].fillna(medians).values
        y_tr = _apply_y_transform(df_train[target_col].values, y_transform, y_params)
        params = dict(base_params)
        params.update(LGBM_PARAM_QUICK)
        # 高-H2: early stoppingはdf_val(このfoldの評価データ)ではなくdf_train内の
        # 専用holdout(90/10)で行う。df_valは評価専用に残し、ESには一切使わない。
        dtr_fit, dtr_es = _split_es_holdout(df_train)
        if dtr_es is not None:
            X_f = dtr_fit[feat_cols].fillna(medians).values
            y_f = _apply_y_transform(dtr_fit[target_col].values, y_transform, y_params)
            X_es = dtr_es[feat_cols].fillna(medians).values
            y_es_t = _apply_y_transform(dtr_es[target_col].values, y_transform, y_params)
            model = _lgb_fit(params, X_f, y_f, X_es, y_es_t, early_stopping=30)
        else:
            model = _lgb_fit(params, X_tr, y_tr)

        eval_df = df_val if df_val is not None else df_train
        eval_kind = "val" if df_val is not None else "train"
        X_ev = eval_df[feat_cols].fillna(medians).values
        preds_t = model.predict(X_ev)
        preds = _invert_y_transform(preds_t, y_transform, y_params)
        lgbm_r2 = float(r2_score(eval_df[target_col].values, preds))

        # 高-H3緩和: 過学習診断用train_r2(df_train全体に対するR²)
        if eval_kind == "train":
            train_r2 = lgbm_r2  # df_val無し: eval_df==df_train のためそのまま流用
        else:
            train_preds_t = model.predict(X_tr)
            train_r2 = float(r2_score(df_train[target_col].values,
                                      _invert_y_transform(train_preds_t, y_transform, y_params)))

        imps = _lgb_importance(model, len(feat_cols))
        total = max(imps.sum(), 1.0)
        feat_list = sorted(
            [{"name": feat_cols[i], "pct": round(float(imps[i] / total * 100), 1)}
             for i in range(len(feat_cols))],
            key=lambda x: x["pct"], reverse=True)[:10]

        model.save_model(os.path.join(model_dir, "lgbm_model.txt"))
        med_dict = {c: (float(medians[c]) if not np.isnan(float(medians[c])) else 0.0)
                    for c in feat_cols}
        _save_sidecar(med_dict)
        print(f"[LightGBM] R²={lgbm_r2:.4f}  trees={model.num_trees()}", flush=True)
        info = {"eval_kind": eval_kind, "used_cols": feat_cols, "medians": med_dict, "exportable": True,
                "train_r2": round(train_r2, 4)}
        return round(lgbm_r2, 4), feat_list, "lgbm", preds, info

    except Exception as e:
        print(f"[LightGBM] 失敗: {e}", flush=True)
        return None, [], None, None, None


# ─── 3. Gaussian Process (ARD-RBF) ────────────────────────────────────────────
# Returns: (r2, feat_list, 'gp', preds, info)

def _scipy_optimize_gp_hyperparams(X_tr_s, y_train, d, x0=None, max_iter=200):
    """ARD-RBF + White カーネルの超パラメータを自前 L-BFGS で最尤推定する。
    NLL/解析勾配・オプティマイザとも _light（numpy のみ）を使用（scipy 非依存）。
    Returns: (ls_opt, sv_opt, nv_opt, nll_value)
    """
    from _light import gp_nll_and_grad, minimize_lbfgs
    ym = float(y_train.mean()); ys = max(float(y_train.std()), 1e-6)
    yt = (y_train - ym) / ys

    if x0 is None:
        x0 = np.zeros(d + 2); x0[d + 1] = -3.0
    xbest, fbest = minimize_lbfgs(
        lambda p: gp_nll_and_grad(p, X_tr_s, yt, d), x0, max_iter=max_iter)

    ls_opt = np.exp(xbest[:d]).clip(1e-3, 100.)
    sv_opt = float(np.exp(xbest[d]).clip(1e-4, 1e4))
    nv_opt = float(np.exp(xbest[d + 1]).clip(1e-6, 10.))
    return ls_opt, sv_opt, nv_opt, fbest


def _gp_feature_importance(gp, ls_opt, feat_cols, n_features):
    ard_ls_src = ls_opt
    if ard_ls_src is None:
        try:
            ard_ls_src = np.atleast_1d(np.array(gp.length_scale, dtype=float))
        except Exception:
            ard_ls_src = None
    if ard_ls_src is None or len(ard_ls_src) != n_features:
        return []
    inv_ls = 1.0 / np.clip(ard_ls_src, 1e-9, None)
    importance = inv_ls / inv_ls.sum()
    return sorted(
        [{"name": feat_cols[i], "pct": round(float(importance[i]) * 100, 1)}
         for i in range(n_features)],
        key=lambda x: x["pct"], reverse=True)[:10]


def _try_gp(df_train, df_val, target_col, model_dir, use_grid=False, use_oof=False,
            y_transform='none', y_params={}, df_all=None, feat_cols_override=None, splits=None,
            df_all_per_fold=None, screen_cols_per_fold=None):
    try:
        import pickle
        from _light import StandardScaler, LightGP

        feat_cols = feat_cols_override if feat_cols_override else _get_feat_cols(df_train, target_col)
        if not feat_cols:
            return None, [], None, None, None

        n_features = len(feat_cols)
        n_restarts_opt = GP_RESTARTS_THOROUGH if use_grid else GP_RESTARTS_QUICK

        def _fit_one_gp(dtr, dva_or_none, feat_cols_local=None):
            fc = feat_cols_local if feat_cols_local is not None else feat_cols
            d_local = len(fc)
            medians_l = dtr[fc].median()
            X_full = dtr[fc].fillna(medians_l).values
            y_full_raw = dtr[target_col].values
            n_full = len(X_full)
            if n_full > GP_MAX_TRAIN:
                idx = np.random.RandomState(42).choice(n_full, GP_MAX_TRAIN, replace=False)
                X_tr, y_tr_raw = X_full[idx], y_full_raw[idx]
            else:
                X_tr, y_tr_raw = X_full, y_full_raw
            y_tr = _apply_y_transform(y_tr_raw, y_transform, y_params)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)

            ls_opt = sv_opt = nv_opt = None
            if len(X_tr) >= SCIPY_GP_MIN_ROWS:
                best_nll = float('inf')
                best_params = None
                for ridx in range(n_restarts_opt):
                    x0 = np.zeros(d_local + 2)
                    if ridx > 0:
                        x0[:d_local] = np.random.randn(d_local) * 0.5
                    x0[d_local + 1] = -3.0
                    try:
                        lo_, sv_, nv_, nll_ = _scipy_optimize_gp_hyperparams(
                            X_tr_s, y_tr, d_local, x0=x0)
                        if nll_ < best_nll:
                            best_nll = nll_
                            best_params = (lo_, sv_, nv_)
                    except Exception:
                        pass
                if best_params is not None:
                    ls_opt, sv_opt, nv_opt = best_params

            if ls_opt is not None:
                gp = LightGP(length_scale=ls_opt, sigma_var=sv_opt, noise_var=nv_opt)
            else:
                # 最適化を回さない小データ等: デフォルトハイパラ（ls=1, sv=1, nv=1e-2）
                gp = LightGP(length_scale=np.ones(d_local), sigma_var=1.0, noise_var=1e-2)
            gp.fit(X_tr_s, y_tr)

            preds_local = None
            if dva_or_none is not None:
                X_v = dva_or_none[fc].fillna(medians_l).values
                preds_t = gp.predict(scaler.transform(X_v))
                preds_local = _invert_y_transform(preds_t, y_transform, y_params)
            med_dict = {c: (float(medians_l[c]) if not np.isnan(float(medians_l[c])) else 0.0)
                        for c in fc}
            return gp, scaler, ls_opt, preds_local, med_dict

        if use_oof and df_all is not None and splits is not None:
            df_full = df_all
            n_splits = len(splits)
            oof_preds = np.zeros(len(df_full))
            for fold, (tr_idx, va_idx) in enumerate(splits):
                # 高-H1: fold毎に(そのfoldの検証行を見ずに)fitしたFE済みデータ・
                # スクリーニング済み特徴列を使う
                df_fold, feat_cols_f = _fold_frame(df_full, df_all_per_fold, fold, tr_idx,
                                                   target_col, feat_cols, screen_cols_per_fold)
                dtr = df_fold.iloc[tr_idx]
                dva = df_fold.iloc[va_idx]
                _, _, _, preds_local, _ = _fit_one_gp(dtr, dva, feat_cols_local=feat_cols_f)
                oof_preds[va_idx] = preds_local
                if use_grid:
                    fold_r2 = r2_score(dva[target_col].values, preds_local)
                    print(f"  [GP] fold {fold+1}/{n_splits}: R²={fold_r2:.4f}", flush=True)
            oof_r2 = float(r2_score(df_full[target_col].values, oof_preds))
            print(f"[GP] OOF R²={oof_r2:.4f}", flush=True)

            # 高-H3緩和: train_r2算出のため、最終モデルへの入力にdf_fullを検証データとしても
            # 渡し(train_preds取得)、過学習診断用の在庫データR²を得る。
            gp_final, scaler_final, ls_final, train_preds, med_dict = _fit_one_gp(df_full, df_full)
            train_r2 = (float(r2_score(df_full[target_col].values, train_preds))
                        if train_preds is not None else None)
            feat_list = _gp_feature_importance(gp_final, ls_final, feat_cols, n_features)
            with open(os.path.join(model_dir, "gp_model.pkl"), "wb") as f:
                pickle.dump({"model": gp_final, "scaler": scaler_final,
                             "feat_cols": feat_cols, "target_col": target_col,
                             "medians": med_dict}, f)
            info = {"eval_kind": "oof", "used_cols": feat_cols, "medians": med_dict, "exportable": True,
                    "train_r2": round(train_r2, 4) if train_r2 is not None else None}
            return round(oof_r2, 4), feat_list, "gp", oof_preds, info

        # 非OOF: 単一 train/val
        eval_df = df_val if df_val is not None else df_train
        eval_kind = "val" if df_val is not None else "train"
        gp, scaler, ls_opt, _, med_dict = _fit_one_gp(df_train, None)
        medians = df_train[feat_cols].median()
        X_val = eval_df[feat_cols].fillna(medians).values
        preds_t = gp.predict(scaler.transform(X_val))
        preds = _invert_y_transform(preds_t, y_transform, y_params)
        gp_r2 = float(r2_score(eval_df[target_col].values, preds))

        # 高-H3緩和: 過学習診断用train_r2(df_train全体に対するR²)
        if eval_kind == "train":
            train_r2 = gp_r2  # df_val無し: eval_df==df_train のためそのまま流用
        else:
            X_tr_val = df_train[feat_cols].fillna(medians).values
            train_preds_t = gp.predict(scaler.transform(X_tr_val))
            train_r2 = float(r2_score(df_train[target_col].values,
                                      _invert_y_transform(train_preds_t, y_transform, y_params)))

        feat_list = _gp_feature_importance(gp, ls_opt, feat_cols, n_features)
        with open(os.path.join(model_dir, "gp_model.pkl"), "wb") as f:
            pickle.dump({"model": gp, "scaler": scaler,
                         "feat_cols": feat_cols, "target_col": target_col,
                         "medians": med_dict}, f)

        print(f"[GP] R²={gp_r2:.4f}", flush=True)
        info = {"eval_kind": eval_kind, "used_cols": feat_cols, "medians": med_dict, "exportable": True,
                "train_r2": round(train_r2, 4)}
        return round(gp_r2, 4), feat_list, "gp", preds, info

    except Exception as e:
        print(f"[GP] 失敗: {e}", flush=True)
        return None, [], None, None, None


# ─── 4. MLP (numpy 自前: forward + Adam) ──────────────────────────────────────
# Returns: (r2, feat_list, 'mlp', preds, info)

def _mlp_feature_importance(pipeline, eval_df, feat_cols, target_col, y_transform, y_params):
    from _light import permutation_importance

    medians = eval_df[feat_cols].median()
    X_ev = eval_df[feat_cols].fillna(medians).values
    y_ev = eval_df[target_col].values

    class _WrappedMLP:
        def __init__(self, pipe, t, p):
            self.pipe = pipe; self.t = t; self.p = p
        def fit(self, X, y=None):
            # 既に学習済みのモデルをラップするだけ（sklearn の estimator API 検証を通すためのダミー）
            return self
        def predict(self, X):
            return _invert_y_transform(self.pipe.predict(X), self.t, self.p)
        def score(self, X, y):
            return r2_score(y, self.predict(X))

    wrapped = _WrappedMLP(pipeline, y_transform, y_params)
    X_pi, y_pi = X_ev, y_ev
    if len(X_pi) > 100:
        idx = np.random.RandomState(42).choice(len(X_pi), 100, replace=False)
        X_pi, y_pi = X_pi[idx], y_pi[idx]
    try:
        pi = permutation_importance(wrapped, X_pi, y_pi, n_repeats=5, random_state=42)
        imps = pi.importances_mean
    except Exception:
        imps = np.zeros(len(feat_cols))
    pos_total = max(float(imps[imps > 0].sum()), 1e-9)
    return sorted(
        [{"name": feat_cols[i],
          "pct": round(float(imps[i]) / pos_total * 100, 1) if imps[i] > 0 else 0.0}
         for i in range(len(feat_cols))],
        key=lambda x: x["pct"], reverse=True)[:10]


def _try_mlp(df_train, df_val, target_col, model_dir, use_grid=False, use_oof=False,
             y_transform='none', y_params={}, df_all=None, feat_cols_override=None, splits=None,
             df_all_per_fold=None, screen_cols_per_fold=None):
    try:
        import pickle
        from _light import StandardScaler, LightMLP, LightPipeline

        feat_cols = feat_cols_override if feat_cols_override else _get_feat_cols(df_train, target_col)
        if not feat_cols or len(df_train) < MLP_MIN_ROWS:
            return None, [], None, None, None

        d = len(feat_cols)
        h1 = min(256, max(32, d * 4))
        h2 = min(128, max(16, d * 2))
        param_grid = _mlp_search_candidates() if use_grid else [MLP_PARAM_QUICK]

        def _build_pipeline(param_spec, n_rows_local):
            alpha        = param_spec.get('alpha', 1e-4)
            single_layer = param_spec.get('single_layer', False)
            extra_layer  = param_spec.get('extra_layer', False)
            width        = param_spec.get('width', 1.0)
            h1w = max(8, int(h1 * width))
            h2w = max(4, int(h2 * width))
            if single_layer:
                hidden = (h1w,)
            elif extra_layer:
                hidden = (h1w, h2w, h1w)
            else:
                hidden = (h1w, h2w)
            # numpy 自前 MLP（forward+Adam）。PoC で sklearn(lbfgs) 以上の外部R²を実証。
            max_iter = 1500 if use_grid else 600
            mlp = LightMLP(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                           learning_rate_init=1e-2, random_state=42)
            pipeline = LightPipeline([('scaler', StandardScaler()), ('mlp', mlp)])
            return pipeline, hidden

        def _fit_and_eval(dtr, dva_or_none, param_spec, feat_cols_local=None):
            fc = feat_cols_local if feat_cols_local is not None else feat_cols
            medians_l = dtr[fc].median()
            X_tr = dtr[fc].fillna(medians_l).values
            y_tr = _apply_y_transform(dtr[target_col].values, y_transform, y_params)
            pipeline, hidden = _build_pipeline(param_spec, len(dtr))
            pipeline.fit(X_tr, y_tr)
            preds = None
            if dva_or_none is not None:
                X_v = dva_or_none[fc].fillna(medians_l).values
                preds_t = pipeline.predict(X_v)
                preds = _invert_y_transform(preds_t, y_transform, y_params)
            med_dict = {c: (float(medians_l[c]) if not np.isnan(float(medians_l[c])) else 0.0)
                        for c in fc}
            return pipeline, hidden, preds, med_dict

        if use_oof and df_all is not None and splits is not None:
            df_full = df_all
            n_splits = len(splits)

            # ── 予選足切り: 全候補を先頭2foldで評価し、上位のみ全foldで本戦 ──
            param_list = param_grid
            if use_grid and len(param_grid) > MLP_FINALISTS and n_splits > MLP_HALVING_FOLDS:
                print(f"[MLP] ランダムサーチ {len(param_grid)} 候補を予選 ({MLP_HALVING_FOLDS}fold)...", flush=True)
                prelim_scores = []
                for pidx, param_spec in enumerate(param_grid, 1):
                    preds_p = np.zeros(len(df_full))
                    touched = np.zeros(len(df_full), dtype=bool)
                    try:
                        for h_fold, (tr_idx, va_idx) in enumerate(splits[:MLP_HALVING_FOLDS]):
                            # 高-H1: fold毎に(そのfoldの検証行を見ずに)fitしたFE済みデータ・
                            # スクリーニング済み特徴列を使う
                            df_fold, feat_cols_f = _fold_frame(df_full, df_all_per_fold, h_fold, tr_idx,
                                                               target_col, feat_cols, screen_cols_per_fold)
                            dtr = df_fold.iloc[tr_idx]
                            dva = df_fold.iloc[va_idx]
                            _, _, preds_local, _ = _fit_and_eval(dtr, dva, param_spec, feat_cols_local=feat_cols_f)
                            preds_p[va_idx] = preds_local
                            touched[va_idx] = True
                        r2_p = float(r2_score(df_full[target_col].values[touched], preds_p[touched]))
                    except Exception:
                        r2_p = -np.inf
                    print(f"  予選{pidx}/{len(param_grid)}: R²={r2_p:.4f} (alpha={param_spec.get('alpha'):.2g})", flush=True)
                    prelim_scores.append(r2_p if np.isfinite(r2_p) else -np.inf)
                top_idx = np.argsort(prelim_scores)[::-1][:MLP_FINALISTS]
                param_list = [param_grid[i] for i in top_idx]
                print(f"[MLP] 予選通過 {len(param_list)} 候補 → 全{n_splits}foldで本戦", flush=True)

            best_param_r2 = -np.inf
            best_param_spec = param_list[0]
            best_oof_preds = None
            for pidx, param_spec in enumerate(param_list, 1):
                if use_grid:
                    print(f"[MLP] 候補{pidx}/{len(param_list)}: alpha={param_spec.get('alpha'):.2g}, "
                          f"single_layer={param_spec.get('single_layer', False)}", flush=True)
                oof_preds = np.zeros(len(df_full))
                for fold, (tr_idx, va_idx) in enumerate(splits):
                    # 高-H1: fold毎に(そのfoldの検証行を見ずに)fitしたFE済みデータ・
                    # スクリーニング済み特徴列を使う
                    df_fold, feat_cols_f = _fold_frame(df_full, df_all_per_fold, fold, tr_idx,
                                                       target_col, feat_cols, screen_cols_per_fold)
                    dtr = df_fold.iloc[tr_idx]
                    dva = df_fold.iloc[va_idx]
                    _, _, preds_local, _ = _fit_and_eval(dtr, dva, param_spec, feat_cols_local=feat_cols_f)
                    oof_preds[va_idx] = preds_local
                oof_r2 = float(r2_score(df_full[target_col].values, oof_preds))
                if use_grid:
                    print(f"  OOF R²={oof_r2:.4f}", flush=True)
                if oof_r2 > best_param_r2:
                    best_param_r2 = oof_r2
                    best_param_spec = param_spec
                    best_oof_preds = oof_preds

            # 高-H3緩和: train_r2算出のため、最終モデルへの入力にdf_fullを検証データとしても
            # 渡し(train_preds取得)、過学習診断用の在庫データR²を得る。
            pipeline, hidden, train_preds, med_dict = _fit_and_eval(df_full, df_full, best_param_spec)
            train_r2 = (float(r2_score(df_full[target_col].values, train_preds))
                       if train_preds is not None else None)
            feat_list = _mlp_feature_importance(pipeline, df_full, feat_cols, target_col, y_transform, y_params)
            with open(os.path.join(model_dir, "mlp_model.pkl"), "wb") as f:
                pickle.dump({"pipeline": pipeline, "feat_cols": feat_cols,
                             "target_col": target_col, "medians": med_dict}, f)
            n_iter = getattr(pipeline['mlp'], 'n_iter_', '?')
            print(f"[MLP] OOF R²={best_param_r2:.4f}  hidden={hidden}  iter={n_iter}", flush=True)
            info = {"eval_kind": "oof", "used_cols": feat_cols, "medians": med_dict, "exportable": True,
                    "train_r2": round(train_r2, 4) if train_r2 is not None else None}
            return round(best_param_r2, 4), feat_list, "mlp", best_oof_preds, info

        # 非OOF: 単一 train/val
        eval_df = df_val if df_val is not None else df_train
        eval_kind = "val" if df_val is not None else "train"
        best_pipeline = None
        best_mlp_r2 = -np.inf
        best_hidden = None
        best_preds = None
        best_med = None
        for pidx, param_spec in enumerate(param_grid, 1):
            if use_grid:
                print(f"[MLP] 候補{pidx}/{len(param_grid)}: alpha={param_spec.get('alpha')}, "
                      f"single_layer={param_spec.get('single_layer', False)}", flush=True)
            pipeline, hidden, preds, med_dict = _fit_and_eval(df_train, eval_df, param_spec)
            mlp_r2 = float(r2_score(eval_df[target_col].values, preds))
            if mlp_r2 > best_mlp_r2:
                best_mlp_r2 = mlp_r2
                best_pipeline = pipeline
                best_hidden = hidden
                best_preds = preds
                best_med = med_dict

        if best_pipeline is None:
            return None, [], None, None, None

        # 高-H3緩和: 過学習診断用train_r2(df_train全体に対するR²)
        if eval_kind == "train":
            train_r2 = best_mlp_r2  # df_val無し: eval_df==df_train のためそのまま流用
        else:
            X_tr_ev = df_train[feat_cols].fillna(df_train[feat_cols].median()).values
            train_preds_t = best_pipeline.predict(X_tr_ev)
            train_r2 = float(r2_score(df_train[target_col].values,
                                      _invert_y_transform(train_preds_t, y_transform, y_params)))

        feat_list = _mlp_feature_importance(best_pipeline, eval_df, feat_cols, target_col, y_transform, y_params)
        with open(os.path.join(model_dir, "mlp_model.pkl"), "wb") as f:
            pickle.dump({"pipeline": best_pipeline, "feat_cols": feat_cols,
                         "target_col": target_col, "medians": best_med}, f)
        n_iter = getattr(best_pipeline['mlp'], 'n_iter_', '?')
        print(f"[MLP] R²={best_mlp_r2:.4f}  hidden={best_hidden}  iter={n_iter}", flush=True)
        info = {"eval_kind": eval_kind, "used_cols": feat_cols, "medians": best_med, "exportable": True,
                "train_r2": round(train_r2, 4)}
        return round(best_mlp_r2, 4), feat_list, "mlp", best_preds, info

    except Exception as e:
        print(f"[MLP] 失敗: {e}", flush=True)
        return None, [], None, None, None


# ─── 5. LightGBM バギング多様化メンバー（RF / ExtraTrees モード） ──────────────
# sklearn.ensemble の代替（sklearn 非依存）。ブースティング主軸とは異なる
# 「バギング＋ランダム化木」の予測多様性をブレンドに供給する。in-app 予測専用。
# Returns: (r2, feat_list, 'rf'|'xt', preds, info)

_LGBM_BAG_PARAMS = {
    'rf': dict(boosting_type='rf', num_leaves=63, n_estimators=300,
               bagging_fraction=0.7, bagging_freq=1, feature_fraction=0.7,
               min_child_samples=10),
    'xt': dict(boosting_type='rf', num_leaves=63, n_estimators=300,
               bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.6,
               min_child_samples=5, extra_trees=True),
}
_LGBM_BAG_TAG = {'rf': 'LGBM-RF', 'xt': 'LGBM-XT'}


def _try_sktree(kind, df_train, target_col, model_dir,
                y_transform='none', y_params={}, df_all=None, splits=None, num_jobs=4,
                df_all_per_fold=None):
    try:
        import pickle  # noqa: F401 (後方互換のため保持)
        tag = _LGBM_BAG_TAG.get(kind, kind)
        fname = f"{kind}_model.txt"
        base = dict(objective='regression', metric='rmse', verbosity=-1,
                    n_jobs=num_jobs, force_col_wise=True, deterministic=True, random_state=42)
        base.update(_LGBM_BAG_PARAMS[kind])

        feat_cols = _get_feat_cols(df_train, target_col)
        if not feat_cols or df_all is None or splits is None:
            return None, [], None, None, None

        df_full = df_all
        oof_preds = np.zeros(len(df_full))
        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            # 高-H1: fold毎に(そのfoldの検証行を見ずに)fitしたFE済みデータ・特徴列を使う
            df_fold, feat_cols_f = _fold_frame(df_full, df_all_per_fold, fold_idx, tr_idx,
                                               target_col, feat_cols)
            dtr = df_fold.iloc[tr_idx]
            dva = df_fold.iloc[va_idx]
            med_f = dtr[feat_cols_f].median()
            bst = _lgb_fit(base, dtr[feat_cols_f].fillna(med_f).values,
                           _apply_y_transform(dtr[target_col].values, y_transform, y_params))
            preds_t = bst.predict(dva[feat_cols_f].fillna(med_f).values)
            oof_preds[va_idx] = _invert_y_transform(preds_t, y_transform, y_params)
        oof_r2 = float(r2_score(df_full[target_col].values, oof_preds))
        print(f"[{tag}] OOF R²={oof_r2:.4f}", flush=True)

        medians_full = df_full[feat_cols].median()
        X_all = df_full[feat_cols].fillna(medians_full).values
        final_bst = _lgb_fit(base, X_all,
                             _apply_y_transform(df_full[target_col].values, y_transform, y_params))

        # 高-H3緩和: 過学習診断用train_r2(最終モデル自身の学習データに対するR²)
        train_preds_t = final_bst.predict(X_all)
        train_r2 = float(r2_score(df_full[target_col].values,
                                  _invert_y_transform(train_preds_t, y_transform, y_params)))

        med_dict = {c: (float(medians_full[c]) if not np.isnan(float(medians_full[c])) else 0.0)
                    for c in feat_cols}
        # LightGBM テキスト形式で保存（sidecar に feat_cols/medians）
        final_bst.save_model(os.path.join(model_dir, fname))
        with open(os.path.join(model_dir, f"{kind}_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"feat_cols": feat_cols, "medians": med_dict}, f, ensure_ascii=False)

        imps = _lgb_importance(final_bst, len(feat_cols))
        total = max(imps.sum(), 1.0)
        feat_list = sorted(
            [{"name": feat_cols[i], "pct": round(float(imps[i] / total * 100), 1)}
             for i in range(len(feat_cols))],
            key=lambda x: x["pct"], reverse=True)[:10]

        # LGBM-RF/XT はテキストモデル形式がLightGBMネイティブと同一のため .treg 書き出し可能
        # (_load_export_source が 'lgbm' にエイリアスして _parse_lgbm_to_treg_bytes を再利用)
        info = {"eval_kind": "oof", "used_cols": feat_cols, "medians": med_dict, "exportable": True,
                "train_r2": round(train_r2, 4)}
        return round(oof_r2, 4), feat_list, kind, oof_preds, info

    except Exception as e:
        print(f"[{_LGBM_BAG_TAG.get(kind, kind)}] 失敗: {e}", flush=True)
        return None, [], None, None, None


# ─── Blend (OOF-NNLS) ─────────────────────────────────────────────────────────

def _fit_blend_oof(candidates, y_full):
    """全行OOF予測を持つ候補で NNLS ブレンドを構成する。
    Returns: (blend_r2, names, weights, blend_oof_preds, feat_list) or None"""
    n_rows = len(y_full)
    members = {}
    for name, (c_r2, c_feats, c_type, c_preds, c_info) in candidates.items():
        if c_preds is None or c_r2 is None or c_r2 < BLEND_R2_THRESH:
            continue
        if c_info is None or c_info.get("eval_kind") != "oof":
            continue
        c_preds = np.asarray(c_preds, dtype=float)
        if len(c_preds) != n_rows or not np.isfinite(c_preds).all():
            continue
        members[name] = c_preds
    if len(members) < 2:
        return None

    names = list(members.keys())
    stacked = np.column_stack([members[n] for n in names])
    try:
        from _light import nnls
        weights, _ = nnls(stacked, y_full)
        if weights.sum() <= 1e-9:
            raise ValueError("NNLS returned all-zero weights")
    except Exception:
        # フォールバック: R²比例（和1に正規化 — 生内積でもスケール整合）
        weights = np.clip(np.array([candidates[n][0] for n in names]), 1e-6, None)
        weights = weights / weights.sum()

    blend_oof = stacked @ weights
    blend_r2 = float(r2_score(y_full, blend_oof))
    weight_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights))
    print(f"[Blend] members=[{weight_str}] OOF R²={blend_r2:.4f}", flush=True)

    # 各メンバーの特徴量重要度をブレンド重みで加重平均する
    combined_imp = {}
    for name, w in zip(names, weights):
        if w <= 0:
            continue
        for f in candidates[name][1]:
            combined_imp[f['name']] = combined_imp.get(f['name'], 0.0) + w * f['pct']
    total_imp = sum(combined_imp.values())
    feat_list = sorted(
        [{"name": k, "pct": round(v / total_imp * 100, 1)} for k, v in combined_imp.items()],
        key=lambda x: x["pct"], reverse=True)[:10] if total_imp > 0 else []

    return blend_r2, names, weights, blend_oof, feat_list


# ─── .treg バイナリエクスポート ────────────────────────────────────────────────

def _write_str_treg(f, s):
    b = s.encode('utf-8')
    f.write(len(b).to_bytes(2, 'little'))
    f.write(b)


def _parse_lgbm_to_treg_bytes(model_txt_path):
    """LightGBM v4 テキスト形式 (`Tree=N` ブロック) をパースして .treg のツリーバイト列を返す。
    不整合・非対応形式は ValueError を送出する（黙って壊れた出力を返さない）。"""
    import re, struct
    with open(model_txt_path, encoding='utf-8') as f:
        content = f.read()

    m = re.search(r'^end of trees\s*$', content, re.MULTILINE)
    body = content[:m.start()] if m else content

    # LightGBM の RF モード(boosting_type='rf'、LGBM-RF/LGBM-XT が使用)はヘッダに
    # 'average_output' フラグが立ち、Booster.predict() は「全木の出力の平均」を返す
    # (通常のGBDTは各木の leaf_value に shrinkage が既に畳み込まれているため単純な総和でよいが、
    #  RFモードは畳み込まれていない生の平均のため、木の本数で割る必要がある)。
    # .treg / JS側の predictLgbm は常に「全木の総和」として読むため、ここで
    # leaf_value を木の本数であらかじめ割っておくことで、フォーマット・読み込み側は
    # 一切変更せずに済む。
    average_output = bool(re.search(r'^average_output\s*$', content, re.MULTILINE))

    parts = re.split(r'(?m)^Tree=(\d+)\s*$', body)
    if len(parts) < 3:
        raise ValueError("no 'Tree=N' blocks found (unexpected LightGBM model text format)")

    trees = []
    for i in range(2, len(parts), 2):  # parts[0]=ファイルヘッダ, 奇数=ツリー番号
        kv = {}
        for line in parts[i].splitlines():
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            kv[k] = v

        if int(kv.get('num_cat', '0') or 0) > 0:
            raise ValueError("categorical split (num_cat>0) is not supported by .treg export")

        n_leaves = int(kv['num_leaves'])
        if n_leaves < 1:
            raise ValueError(f"invalid num_leaves={n_leaves}")
        leaf_value = [float(x) for x in kv.get('leaf_value', '').split()]
        if len(leaf_value) != n_leaves:
            raise ValueError(f"leaf_value length {len(leaf_value)} != num_leaves {n_leaves}")

        if n_leaves == 1:
            # 単葉ツリー（定数寄与）: 内部ノード配列なし
            trees.append({'n_leaves': 1, 'split_feature': [], 'threshold': [],
                          'left_child': [], 'right_child': [], 'leaf_value': leaf_value})
            continue

        ni = n_leaves - 1

        def _ints(key):
            a = [int(x) for x in kv.get(key, '').split()]
            if len(a) != ni:
                raise ValueError(f"{key} length {len(a)} != {ni}")
            return a

        def _floats(key):
            a = [float(x) for x in kv.get(key, '').split()]
            if len(a) != ni:
                raise ValueError(f"{key} length {len(a)} != {ni}")
            return a

        trees.append({'n_leaves': n_leaves,
                      'split_feature': _ints('split_feature'),
                      'threshold':     _floats('threshold'),
                      'left_child':    _ints('left_child'),
                      'right_child':   _ints('right_child'),
                      'leaf_value':    leaf_value})

    if not trees:
        raise ValueError("parsed 0 trees")

    if average_output:
        inv_n = 1.0 / len(trees)
        for t in trees:
            t['leaf_value'] = [v * inv_n for v in t['leaf_value']]

    buf = bytearray()
    buf += struct.pack('<II', len(trees), 0)
    for t in trees:
        ni = t['n_leaves'] - 1
        buf += struct.pack('<I', t['n_leaves'])
        for i in range(ni):
            buf += struct.pack('<I', t['split_feature'][i])
        for i in range(ni):
            buf += struct.pack('<f', t['threshold'][i])
        for i in range(ni):
            buf += struct.pack('<i', t['left_child'][i])
        for i in range(ni):
            buf += struct.pack('<i', t['right_child'][i])
        for i in range(t['n_leaves']):
            buf += struct.pack('<f', t['leaf_value'][i])
    return bytes(buf)


def _load_export_source(model_type, model_dir):
    """モデル種別に応じて (feat_cols, medians, payload, export_type) を pkl / sidecar から
    自己取得する。export_type は .treg 上の実書式ファミリ:
      - poly-Ridge は 'linear_poly'（標準化後に多項式展開する専用フォーマット）
      - LGBM-RF/LGBM-XT は木構造・予測規則が LightGBM ネイティブ形式と同一のため
        'lgbm' にエイリアスする（LightGBM は boosting_type='rf' でもテキストモデル形式・
        「木の出力を足し合わせる」推論規則は通常の GBDT と変わらないため）
    未対応の種別（未知の model_type）のみ None を返す。"""
    import pickle
    if model_type == 'linear':
        with open(os.path.join(model_dir, 'linear_model.pkl'), 'rb') as pf:
            d = pickle.load(pf)
        export_type = 'linear_poly' if d.get('use_poly') else 'linear'
        return d['feat_cols'], d.get('medians', {}), d, export_type
    if model_type == 'lgbm':
        with open(os.path.join(model_dir, 'lgbm_meta.json'), encoding='utf-8') as f:
            meta = json.load(f)
        return meta['feat_cols'], meta.get('medians', {}), {'model_txt': 'lgbm_model.txt'}, 'lgbm'
    if model_type in ('rf', 'xt'):
        with open(os.path.join(model_dir, f'{model_type}_meta.json'), encoding='utf-8') as f:
            meta = json.load(f)
        return (meta['feat_cols'], meta.get('medians', {}),
                {'model_txt': f'{model_type}_model.txt'}, 'lgbm')
    if model_type == 'gp':
        with open(os.path.join(model_dir, 'gp_model.pkl'), 'rb') as pf:
            d = pickle.load(pf)
        return d['feat_cols'], d.get('medians', {}), d, 'gp'
    if model_type == 'mlp':
        with open(os.path.join(model_dir, 'mlp_model.pkl'), 'rb') as pf:
            d = pickle.load(pf)
        return d['feat_cols'], d.get('medians', {}), d, 'mlp'
    return None


_TREG_TYPE_MAP = {'linear': 0, 'lgbm': 1, 'gp': 2, 'mlp': 3, 'linear_poly': 4, 'blend': 5}
_TREG_OP_MAP   = {'mul': 0, 'sq': 1, 'sign': 2}


def _write_treg_stream(f, export_type, feat_cols, medians, payload, model_dir,
                       target_col, y_transform, y_params, smear, y_clip, round_output,
                       x_clip_all, derived_recipe):
    """1モデル分の完全な .treg バイト列（'TREG' ヘッダ込み）をファイルライクな f に書く。
    export_type=='blend' の場合、payload['members'] の各要素を「後処理なしの自己完結した
    入れ子 .treg ブロブ」として再帰的に埋め込む（この関数自身を再帰呼び出しする）。
    派生特徴（自動FE）を使うモデルは v4（レシピブロック付き）、それ以外は v3 で書く。"""
    import struct
    n_feat = len(feat_cols)

    def _baked_scale(scale):
        # 中-M3: 読み込み側(C++/JS/predict_template.htmlインライン版)は以前
        # `(x-mean)/(scale+1e-8)` としてεを足していたが、アプリ内Python(_light.py の
        # StandardScaler/RobustScaler.transform)は素の除算で、準定数列(IQR/stdが
        # 1e-12超1e-8未満)でこの2つが大きく乖離していた。書き出し時に
        # scale=max(scale,1e-8)を焼き込み、読み込み側は「.tregの値で割るだけ」に
        # 統一する(3実装の数値挙動を揃えることが最重要)。
        return np.maximum(np.asarray(scale, dtype=np.float64), 1e-8).astype(np.float32)

    # このモデルが実際に使う派生特徴のみ書き出す（ソース列の x_clip 境界も同梱）
    feat_set = set(feat_cols)
    used_derived = [r for r in derived_recipe
                    if r['name'] in feat_set and r['op'] in _TREG_OP_MAP]
    file_version = 4 if used_derived else 3

    f.write(b'TREG')
    f.write(struct.pack('<BB', file_version, _TREG_TYPE_MAP[export_type]))
    f.write(struct.pack('<I', n_feat))

    if file_version >= 4:
        f.write(struct.pack('<I', len(used_derived)))
        for r in used_derived:
            cols = r['cols']
            col_a = cols[0]
            col_b = cols[1] if len(cols) > 1 else ''
            a_lo, a_hi = x_clip_all.get(col_a, (-X_CLIP_SENTINEL, X_CLIP_SENTINEL))
            b_lo, b_hi = (x_clip_all.get(col_b, (-X_CLIP_SENTINEL, X_CLIP_SENTINEL))
                          if col_b else (-X_CLIP_SENTINEL, X_CLIP_SENTINEL))
            f.write(struct.pack('<B', _TREG_OP_MAP[r['op']]))
            _write_str_treg(f, r['name'])
            _write_str_treg(f, col_a)
            f.write(struct.pack('<ff', float(a_lo), float(a_hi)))
            _write_str_treg(f, col_b)
            f.write(struct.pack('<ff', float(b_lo), float(b_hi)))

    if export_type == 'linear':
        d = payload
        f.write(np.array(d['scaler'].center_, dtype=np.float32).tobytes())
        f.write(_baked_scale(d['scaler'].scale_).tobytes())
        f.write(np.array(d['model'].coef_,    dtype=np.float32).tobytes())
        f.write(struct.pack('<f', float(d['model'].intercept_)))

    elif export_type == 'linear_poly':
        # poly-Ridge は RobustScaler で標準化した後の値に PolynomialFeatures(degree=2) を
        # 適用した特徴で学習されている(poly.fit_transform(scaler.fit_transform(X)))。
        # RidgeCV.predict() 自体はセンタリングを intercept_ に畳み込み済みなので、推論側は
        # 「標準化 → (単項 or 標準化後の値どうしの積/二乗) → coef_ 内積 + intercept_」の
        # 1パスで再現できる。既存の DOP_MUL/DOP_SQ 派生特徴(生値ベースで掛け合わせてから
        # モデル側でスケーリングする方式)とは演算順序が異なるため専用フォーマットにする。
        # 項の並びは _light.PolynomialFeatures.transform と同一
        # ([単項(i昇順)] + [i<=jの積(i昇順→j昇順)]、i==jは二乗)にし、model.coef_ の
        # 並びとそのまま対応させる。単項は (i, -1) として区別する。
        d = payload
        scaler = d['scaler']
        model = d['model']
        f.write(np.array(scaler.center_, dtype=np.float32).tobytes())
        f.write(_baked_scale(scaler.scale_).tobytes())
        terms = [(i, -1) for i in range(n_feat)]
        for i in range(n_feat):
            for j in range(i, n_feat):
                terms.append((i, j))
        f.write(struct.pack('<I', len(terms)))
        for (ta, tb) in terms:
            f.write(struct.pack('<ii', ta, tb))
        f.write(np.array(model.coef_, dtype=np.float32).tobytes())
        f.write(struct.pack('<f', float(model.intercept_)))

    elif export_type == 'lgbm':
        # 'lgbm' 本体に加え LGBM-RF/LGBM-XT もここに来る
        # (payload['model_txt'] で実ファイル名を切り替えるだけで、木構造の読み書き規則は
        # 完全に同一のため専用の分岐は不要)。
        lgbm_bytes = _parse_lgbm_to_treg_bytes(
            os.path.join(model_dir, payload['model_txt']))
        f.write(lgbm_bytes)

    elif export_type == 'gp':
        d = payload
        scaler = d['scaler']
        gp     = d['model']   # _light.LightGP
        sv = float(gp.sigma_var)
        ls = np.atleast_1d(np.array(gp.length_scale, dtype=float))
        if len(ls) != n_feat:
            ls = np.full(n_feat, float(ls.mean()) if len(ls) else 1.0)
        f.write(np.array(scaler.mean_,  dtype=np.float32).tobytes())
        f.write(_baked_scale(scaler.scale_).tobytes())
        f.write(ls.astype(np.float32).tobytes())
        f.write(struct.pack('<f', sv))
        y_mean = float(getattr(gp, 'y_mean_', 0.0))
        y_std  = float(getattr(gp, 'y_std_',  1.0))
        f.write(struct.pack('<ff', y_mean, y_std))
        n_train = len(gp.X_train_)
        f.write(struct.pack('<I', n_train))
        f.write(gp.X_train_.astype(np.float32).tobytes())
        f.write(gp.alpha_.astype(np.float32).tobytes())

    elif export_type == 'mlp':
        d = payload
        pipeline = d['pipeline']
        scaler   = pipeline['scaler']
        mlp      = pipeline['mlp']
        f.write(np.array(scaler.mean_,  dtype=np.float32).tobytes())
        f.write(_baked_scale(scaler.scale_).tobytes())
        n_layers = len(mlp.coefs_)
        f.write(struct.pack('<I', n_layers))
        for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            n_in, n_out = W.shape
            act = 1 if i == n_layers - 1 else 0
            f.write(struct.pack('<IIB', n_in, n_out, act))
            f.write(W.astype(np.float32).tobytes())
            f.write(b.astype(np.float32).tobytes())

    elif export_type == 'blend':
        # 各メンバーは「後処理なし(smear=1, y_clip=無制限, round無し)」の自己完結した
        # .treg ブロブとして書き、外側(このブロックの外、末尾の共通post-processingブロック)
        # で加重和にのみ実際の smear/y_clip/round_output を適用する
        # (train_bridge._fit_blend_oof / predict_template.py._predict_blend と同じ2段構成:
        #  各メンバー自身のy_transform逆変換は個別に行うが、最終後処理は合成後に1回だけ)。
        import io
        members = payload['members']
        f.write(struct.pack('<I', len(members)))
        for m in members:
            buf = io.BytesIO()
            _write_treg_stream(buf, m['export_type'], m['feat_cols'], m['medians'], m['payload'],
                               model_dir, target_col, y_transform, y_params,
                               1.0, (-X_CLIP_SENTINEL, X_CLIP_SENTINEL), False,
                               x_clip_all, derived_recipe)
            blob = buf.getvalue()
            f.write(struct.pack('<f', float(m['weight'])))
            f.write(struct.pack('<I', len(blob)))
            f.write(blob)

    else:
        raise ValueError(f"unknown export_type: {export_type}")

    # Y 逆変換情報 (v2)
    # blend の外側トレーラは常に 'none' にする: メンバーは上の再帰呼び出し
    # (_write_treg_stream の 'blend' 分岐内、y_transform=実値のまま)で
    # 個別に実スケールへ逆変換済みのため、外側でもう一度適用すると二重逆変換になる
    # (predict-core.js の predictBlend は加重和後、外側 predictRow が invYTransform を
    #  1回適用する設計 — 外側は smear/y_clip/round のみを担当する)。
    Y_TRANSFORM_MAP = {'none': 0, 'log1p': 1, 'yeo_johnson': 2}
    eff_yt = 'none' if export_type == 'blend' else y_transform
    f.write(struct.pack('<B', Y_TRANSFORM_MAP.get(eff_yt, 0)))
    if eff_yt == 'yeo_johnson':
        f.write(struct.pack('<f', float(y_params.get('lambda', 1.0))))

    # 予測後処理情報 (v3): 整数丸め / smearing補正 / Y観測レンジclip / Xクリップ
    f.write(struct.pack('<B', 1 if round_output else 0))
    f.write(struct.pack('<f', float(smear)))
    f.write(struct.pack('<ff', float(y_clip[0]), float(y_clip[1])))
    f.write(struct.pack('<I', n_feat))
    for col in feat_cols:
        lo, hi = x_clip_all.get(col, (-X_CLIP_SENTINEL, X_CLIP_SENTINEL))
        f.write(struct.pack('<ff', float(lo), float(hi)))

    _write_str_treg(f, target_col)
    f.write(struct.pack('<I', n_feat))
    for col in feat_cols:
        _write_str_treg(f, col)
    f.write(struct.pack('<I', n_feat))
    for col in feat_cols:
        _write_str_treg(f, col)
        f.write(struct.pack('<d', float(medians.get(col, 0.0))))


def _export_treg(model_type, model_dir, target_col, y_transform='none', y_params=None,
                 smear=1.0, y_clip=(-X_CLIP_SENTINEL, X_CLIP_SENTINEL),
                 round_output=False, x_clip_all=None, derived_recipe=None):
    """モデル pkl / sidecar から実使用列・median を自己取得し、次元整合した .treg を書く。
    linear(poly含む)/lgbm/gp/mlp/rf/xt 全種別に対応(blend は _export_treg_blend を使う)。"""
    y_params = y_params or {}
    x_clip_all = x_clip_all or {}
    derived_recipe = derived_recipe or []
    out_path = os.path.join(model_dir, "model.treg")
    tmp_path = out_path + ".tmp"

    try:
        src = _load_export_source(model_type, model_dir)
        if src is None:
            print(f"[TREG] {model_type} は非対応のためスキップ", flush=True)
            return False
        feat_cols, medians, payload, export_type = src

        with open(tmp_path, 'wb') as f:
            _write_treg_stream(f, export_type, feat_cols, medians, payload, model_dir,
                               target_col, y_transform, y_params, smear, y_clip, round_output,
                               x_clip_all, derived_recipe)

        os.replace(tmp_path, out_path)
        size_kb = os.path.getsize(out_path) // 1024
        print(f"[TREG] {model_type} → model.treg ({size_kb} KB, {len(feat_cols)}特徴量)", flush=True)
        return True
    except Exception as e:
        print(f"[TREG] エクスポート失敗 ({model_type}): {e}", flush=True)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def _export_treg_blend(model_dir, target_col, candidates, y_transform='none', y_params=None,
                       smear=1.0, y_clip=(-X_CLIP_SENTINEL, X_CLIP_SENTINEL),
                       round_output=False, x_clip_all=None, derived_recipe=None):
    """blend_meta.pkl のメンバー構成をもとに、各メンバーを自己完結した入れ子 .treg として
    埋め込んだアンサンブル用 .treg を書く。1メンバーでも書き出せなければ全体を失敗とする
    （部分的なブレンドは学習時に最適化した重み構成と乖離するため中途半端な出力を避ける）。"""
    import pickle
    y_params = y_params or {}
    x_clip_all = x_clip_all or {}
    derived_recipe = derived_recipe or []
    out_path = os.path.join(model_dir, "model.treg")
    tmp_path = out_path + ".tmp"

    try:
        with open(os.path.join(model_dir, "blend_meta.pkl"), "rb") as f:
            bm = pickle.load(f)
        names = bm["models"]
        weights_map = bm.get("weights", {})

        members = []
        for name in names:
            if name not in candidates:
                raise ValueError(f"Blend サブモデル '{name}' が候補に見つかりません")
            member_model_type = candidates[name][2]
            src = _load_export_source(member_model_type, model_dir)
            if src is None:
                raise ValueError(f"Blend サブモデル '{name}' ({member_model_type}) を書き出せません")
            m_feat_cols, m_medians, m_payload, m_export_type = src
            members.append({
                "export_type": m_export_type, "feat_cols": m_feat_cols,
                "medians": m_medians, "payload": m_payload,
                "weight": float(weights_map.get(name, 0.0)),
            })
        if len(members) < 2:
            raise ValueError("Blend の書き出し可能サブモデルが2未満です")

        with open(tmp_path, 'wb') as f:
            _write_treg_stream(f, 'blend', [], {}, {"members": members}, model_dir,
                               target_col, y_transform, y_params, smear, y_clip, round_output,
                               x_clip_all, derived_recipe)

        os.replace(tmp_path, out_path)
        size_kb = os.path.getsize(out_path) // 1024
        print(f"[TREG] blend({len(members)}メンバー) → model.treg ({size_kb} KB)", flush=True)
        return True
    except Exception as e:
        print(f"[TREG] エクスポート失敗 (blend): {e}", flush=True)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


# ─── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

    if len(sys.argv) < 2:
        print("ERROR: CSVパスが指定されていません", flush=True)
        sys.exit(1)

    csv_path      = sys.argv[1]
    target_column = sys.argv[2] if len(sys.argv) > 2 else None
    strategy      = sys.argv[4] if len(sys.argv) > 4 else 'quick'
    num_jobs      = _NUM_JOBS
    thorough      = (strategy == 'thorough')

    print(f"[Python] CSV を解析中... {csv_path}", flush=True)
    _emit_progress(3, "CSVを読み込み中")
    df = _read_csv_with_encoding_fallback(csv_path)

    # ── ターゲット解決・検証（カテゴリエンコードより前） ─────────────────────
    df, target_column, n_target_na = _resolve_and_validate_target(df, target_column)

    cat_encoders = _encode_categoricals(df, target_column)
    if cat_encoders:
        print(f"[Python] カテゴリ列エンコード: {list(cat_encoders.keys())}", flush=True)
        df = _apply_cat_encoders(df, cat_encoders)

    n_rows = len(df)
    print(f"[Python] {n_rows} 行 / {df.shape[1]} 列 / モード: {'じっくり' if thorough else 'お急ぎ'} / CPU並列: {num_jobs}", flush=True)

    target_is_integer = bool(np.all(np.mod(df[target_column].dropna().values, 1.0) == 0.0))
    y_raw_all = df[target_column].values.copy()  # winsorize 前の生 y（y_clip 用）

    # 定数列・重複列の除去
    feat_probe = _get_feat_cols(df, target_column)
    const_cols, dup_cols = _find_constant_and_duplicate_cols(df, feat_probe)
    drop_cols = const_cols + dup_cols
    if drop_cols:
        print(f"[Python] 定数/重複列を除去: {drop_cols}", flush=True)
        df = df.drop(columns=drop_cols)

    # ── fold 分割の一元計算（全モデルで共有） ─────────────────────────────────
    use_oof    = thorough or (n_rows < SMALL_N_OOF_THRESH)
    have_split = n_rows >= MIN_ROWS_FOR_SPLIT
    if have_split:
        n_splits_req = (10 if n_rows < 100 else 5) if use_oof else 5
        cv_splits = _make_binned_splits(df, target_column, n_splits=n_splits_req)
        tr_idx0, va_idx0 = cv_splits[0]
    else:
        cv_splits = None
        tr_idx0 = np.arange(n_rows)
        va_idx0 = np.array([], dtype=int)

    # Y 外れ値 winsorize（学習側のみで境界を決定 → リーク防止、行は保持）
    n_outliers = 0
    if len(tr_idx0) >= 20:
        iqr_mult = OUTLIER_IQR_MULT if thorough else OUTLIER_IQR_QUICK
        lo, hi = _fit_y_winsorize_bounds(df[target_column].values[tr_idx0], iqr_mult)
        df, n_outliers = _apply_y_winsorize(df, target_column, lo, hi)

    # Y 変換（学習側のみで検出・fit → リーク防止）
    y_transform = 'none'
    y_params    = {}
    try:
        y_transform, y_params = _detect_y_transform(df[target_column].values[tr_idx0],
                                                     df[target_column].values)
    except Exception as e:
        print(f"[YTransform] 検出失敗 → 変換スキップ: {e}", flush=True)

    if have_split:
        df_train = df.iloc[tr_idx0].reset_index(drop=True)
        df_val   = df.iloc[va_idx0].reset_index(drop=True)
        print(f"[Python] 層化分割: 学習={len(df_train)} 行 / 検証={len(df_val)} 行", flush=True)
    else:
        df_train, df_val = df, None
        print(f"[Python] データが少ないため全データを学習に使用", flush=True)

    # X クリッピング（学習側のみで fit）
    feat_cols_for_clip = _get_feat_cols(df_train, target_column)
    x_clip_bounds = _compute_x_clip(df_train, feat_cols_for_clip)
    if x_clip_bounds:
        print(f"[Python] X クリッピング ({X_CLIP_PCTILE[0]}%–{X_CLIP_PCTILE[1]}%): {len(x_clip_bounds)} 列", flush=True)
        df_train = _apply_x_clip(df_train, x_clip_bounds)
        if df_val is not None:
            df_val = _apply_x_clip(df_val, x_clip_bounds)
    df_clipped = _apply_x_clip(df, x_clip_bounds) if x_clip_bounds else df

    # ── 自動特徴量エンジニアリング（最終書き出しモデル用、thorough のみ、x_clip 後の値から生成） ──
    #    高-H1: この derived_recipe は df_train(=fold0-train)でfitする。最終書き出しモデルの
    #    学習にのみ使う分には評価対象ではないためリークにならないが、以前はこれをそのまま
    #    OOF全fold共有の df_all_for_models に適用しており、fold1以降の検証行がfold0-trainの
    #    recipe選定に混入してOOF R²が選択バイアスで上振れしていた。OOF評価用の特徴選定は
    #    直後の fold-local ブロックで fold 毎に(そのfoldの検証行を一切見ずに)作り直す。
    df_clipped_base = df_clipped  # FE適用前（fold-local FE/screeningの共有ソース）
    derived_recipe = []
    if thorough and n_rows >= FE_MIN_ROWS:
        derived_recipe = _build_derived_recipe(df_train, target_column, num_jobs=num_jobs)
        if derived_recipe:
            df_train   = _apply_derived(df_train, derived_recipe)
            df_clipped = _apply_derived(df_clipped, derived_recipe)
            if df_val is not None:
                df_val = _apply_derived(df_val, derived_recipe)

    df_all_for_models = df_clipped if (use_oof and have_split) else None
    splits_for_models = cv_splits if (use_oof and have_split) else None

    # ── fold-local FE・特徴量スクリーニング（高-H1: OOF評価の選択バイアス排除） ──────
    #    以前は全fold共通(fold0-trainでfitしたrecipe/screen_cols)をOOF全体に適用しており、
    #    fold1以降の検証行がfold0-trainのFE・スクリーニング選定に混入してOOF R²が上振れ
    #    していた。ここでは各foldの学習側(そのfoldのva_idxを一切含まない)のみで
    #    recipe/screeningを作り直す(LGBM probeのコスト増はfold数倍になるが許容する)。
    df_all_per_fold = None
    screen_cols_per_fold = None
    if use_oof and have_split:
        _emit_progress(8, "fold毎の特徴量選定を実行中")
        df_all_per_fold = []
        screen_cols_per_fold = []
        for tr_idx, va_idx in cv_splits:
            dtr_base = df_clipped_base.iloc[tr_idx]
            recipe_fold = []
            if thorough and len(tr_idx) >= FE_MIN_ROWS:
                recipe_fold = _build_derived_recipe(dtr_base, target_column, num_jobs=num_jobs)
            df_fold = _apply_derived(df_clipped_base, recipe_fold) if recipe_fold else df_clipped_base
            df_all_per_fold.append(df_fold)
            screen_cols_per_fold.append(
                _lgbm_feature_screen(df_fold.iloc[tr_idx], target_column, num_jobs=num_jobs))

    # ── 学習は tmp ディレクトリに書き、成功時にアトミック入替 ─────────────────
    #    (キャンセル時に旧モデル一式が無傷で残る)
    script_dir      = os.path.dirname(os.path.abspath(__file__))
    final_model_dir = os.path.join(script_dir, "trained_model")
    model_dir       = os.path.join(script_dir, "trained_model_tmp")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # 特徴量スクリーニング（最終書き出しモデル用。軽量LGBMで重要度ゼロを除外 → GP/MLPの次元削減）
    _emit_progress(12, "前処理が完了しました")
    screen_cols = _lgbm_feature_screen(df_train, target_column, num_jobs=num_jobs)

    # ── モデル訓練 ────────────────────────────────────────────────────────────
    # candidates: {name: (r2, feat_list, model_type, preds, info)}
    candidates = {}

    def _run_linear():
        return _try_linear(df_train, df_val, target_column, model_dir, y_transform, y_params,
                            df_all=df_all_for_models, use_oof=use_oof, splits=splits_for_models,
                            df_all_per_fold=df_all_per_fold)

    def _run_lgbm():
        return _try_lgbm(df_train, df_val, target_column, model_dir, use_grid=thorough,
                          use_oof=use_oof, y_transform=y_transform, y_params=y_params,
                          df_all=df_all_for_models, num_jobs=num_jobs, splits=splits_for_models,
                          prog=(20, 55) if thorough else None, df_all_per_fold=df_all_per_fold)

    def _run_gp():
        return _try_gp(df_train, df_val, target_column, model_dir, use_grid=thorough,
                       use_oof=use_oof, y_transform=y_transform, y_params=y_params,
                       df_all=df_all_for_models, feat_cols_override=screen_cols,
                       splits=splits_for_models, df_all_per_fold=df_all_per_fold,
                       screen_cols_per_fold=screen_cols_per_fold)

    def _run_mlp():
        if len(df_train) < MLP_MIN_ROWS:
            return None, [], None, None, None
        return _try_mlp(df_train, df_val, target_column, model_dir, use_grid=thorough,
                        use_oof=use_oof, y_transform=y_transform, y_params=y_params,
                        df_all=df_all_for_models, feat_cols_override=screen_cols,
                        splits=splits_for_models, df_all_per_fold=df_all_per_fold,
                        screen_cols_per_fold=screen_cols_per_fold)

    if thorough:
        # じっくりモード: LightGBMは予選→本戦で詳細な進捗ログを出すため逐次実行を維持
        _emit_progress(15, "線形モデルを学習中")
        lin = _run_linear()

        _emit_progress(20, "LightGBMを学習中")
        lgb_res = _run_lgbm()

        _emit_progress(58, "ニューラルネット・ガウス過程を学習中")
        with _thread_limit(max(1, num_jobs // 2)):
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_gp  = ex.submit(_run_gp)
                fut_mlp = ex.submit(_run_mlp)
                gp_res  = fut_gp.result()
                mlp_res = fut_mlp.result()
    else:
        # お急ぎモード: ハイパラ探索なしの一発勝負×4モデルは互いに独立なので
        # 全て並列実行して待ち時間（逐次の合計）をなくす。
        _emit_progress(20, "4種類のモデルを並列学習中")
        with _thread_limit(max(1, num_jobs // 2)):
            with ThreadPoolExecutor(max_workers=4) as ex:
                fut_lin = ex.submit(_run_linear)
                fut_lgb = ex.submit(_run_lgbm)
                fut_gp  = ex.submit(_run_gp)
                fut_mlp = ex.submit(_run_mlp)
                lin     = fut_lin.result()
                lgb_res = fut_lgb.result()
                gp_res  = fut_gp.result()
                mlp_res = fut_mlp.result()

    if lin[0] is not None and np.isfinite(lin[0]):
        candidates['Linear (Ridge)'] = lin
    if lgb_res[0] is not None and np.isfinite(lgb_res[0]):
        candidates['LightGBM'] = lgb_res
    if gp_res[0] is not None and np.isfinite(gp_res[0]):
        candidates['GaussianProcess (ARD-RBF)'] = gp_res
    if mlp_res[0] is not None and np.isfinite(mlp_res[0]):
        candidates['MLP'] = mlp_res

    # ── LightGBM バギング多様化メンバー（RF / ExtraTrees モード、thorough のみ） ──
    #    TREG_NO_SKTREE=1 で無効化（去就の実測比較用）。
    if thorough and have_split and df_all_for_models is not None and os.environ.get("TREG_NO_SKTREE") != "1":
        _emit_progress(78, "追加のモデルを学習中")
        rf_res = _try_sktree('rf', df_train, target_column, model_dir,
                             y_transform=y_transform, y_params=y_params,
                             df_all=df_all_for_models, splits=splits_for_models, num_jobs=num_jobs,
                             df_all_per_fold=df_all_per_fold)
        if rf_res[0] is not None and np.isfinite(rf_res[0]):
            candidates['LGBM-RF'] = rf_res
        xt_res = _try_sktree('xt', df_train, target_column, model_dir,
                             y_transform=y_transform, y_params=y_params,
                             df_all=df_all_for_models, splits=splits_for_models, num_jobs=num_jobs,
                             df_all_per_fold=df_all_per_fold)
        if xt_res[0] is not None and np.isfinite(xt_res[0]):
            candidates['LGBM-XT'] = xt_res

    # ── Blend（OOF-NNLS: 重みfitと評価を全行OOFで行う） ───────────────────────
    if thorough and have_split and len(candidates) >= 2:
        _emit_progress(88, "モデルを組み合わせて最適化中")
        blend_result = _fit_blend_oof(candidates, df[target_column].values)
        if blend_result is not None:
            blend_r2, blend_names, blend_weights, blend_oof, blend_feats = blend_result
            # Blend も各メンバーを入れ子.tregとして埋め込む _export_treg_blend で書き出し可能
            if np.isfinite(blend_r2):
                # 高-H3緩和: blendにはメンバー毎の「最終モデル自身の学習データに対するR²」を
                # 直接計算する手段がない(重み付き線形結合はR²の線形結合と一致しないため厳密値
                # ではないが)ため、参考値としてメンバーのtrain_r2をblend重みで加重平均する。
                member_train_r2s = [candidates[n][4].get("train_r2") for n in blend_names
                                    if candidates[n][4] is not None]
                blend_train_r2 = (float(np.average(member_train_r2s, weights=blend_weights))
                                  if len(member_train_r2s) == len(blend_names)
                                  and all(t is not None for t in member_train_r2s) else None)
                blend_info = {"eval_kind": "oof", "used_cols": [], "medians": {}, "exportable": True,
                              "train_r2": round(blend_train_r2, 4) if blend_train_r2 is not None else None}
                candidates['Blend (Ensemble)'] = (round(blend_r2, 4), blend_feats, 'blend',
                                                  blend_oof, blend_info)
            import pickle as _pkl
            with open(os.path.join(model_dir, "blend_meta.pkl"), "wb") as f:
                _pkl.dump({"models": blend_names,
                           "weights": {n: float(w) for n, w in zip(blend_names, blend_weights)},
                           "normalize": False,
                           "version": 2}, f)

    if not candidates:
        print("ERROR: 有効なモデルが1つも訓練できませんでした", flush=True)
        sys.exit(1)

    best_name = max(candidates, key=lambda k: candidates[k][0] if np.isfinite(candidates[k][0]) else -np.inf)

    # ── Blend 採用マージン: 単体最良を BLEND_MARGIN 以上上回った時のみ採用 ──────
    #    (OOF で僅差勝ちしても未知データでは同等以下になりやすいため)
    if best_name == 'Blend (Ensemble)':
        singles = {k: v for k, v in candidates.items() if k != 'Blend (Ensemble)'}
        if singles:
            best_single = max(singles, key=lambda k: singles[k][0] if np.isfinite(singles[k][0]) else -np.inf)
            diff = candidates[best_name][0] - singles[best_single][0]
            if diff < BLEND_MARGIN:
                print(f"[Blend] 単体最良 ({best_single}) とのOOF差 {diff:+.4f} < {BLEND_MARGIN} "
                      f"→ 安定性を優先し単体モデルを採用", flush=True)
                best_name = best_single

    r2_raw, feat_list, model_type, best_preds, best_info = candidates[best_name]
    print(f"[Python] 最良モデル: {best_name} (R²={r2_raw:.4f})", flush=True)

    # in-app 予測の後方互換フォールバック用（各モデルは自身の pkl の medians を優先使用）
    feat_cols_all  = _get_feat_cols(df_train, target_column)
    medians_raw    = df_train[feat_cols_all].median().to_dict()
    impute_medians = {k: (float(v) if not np.isnan(float(v)) else 0.0)
                     for k, v in medians_raw.items()}
    with open(os.path.join(model_dir, "impute_medians.json"), "w", encoding="utf-8") as f:
        json.dump(impute_medians, f, ensure_ascii=False)

    # ── 評価 + 予測後処理（表示 R² は後処理適用後の予測で再計算） ──────────────
    eval_kind = best_info.get("eval_kind", "train") if best_info else "train"
    y_true_eval = _y_true_for(eval_kind, y_raw_all, tr_idx0, va_idx0)

    smear, y_clip_lo, y_clip_hi, round_output = _fit_postprocess_params(
        best_preds, y_true_eval, y_transform, y_raw_all, target_is_integer)

    r2_report = r2_raw
    scatter_data = None  # 予測 vs 実測プロット用（後処理適用後の予測で表示R²と整合）
    if best_preds is not None and y_true_eval is not None and len(y_true_eval) == len(best_preds):
        corrected = _apply_postprocess(best_preds, smear, y_clip_lo, y_clip_hi, round_output)
        rmse_val, mae_val, eval_on_str, eval_rows, _ = _eval_metrics(
            corrected, eval_kind, y_raw_all, tr_idx0, va_idx0)
        r2_report = round(float(r2_score(y_true_eval, corrected)), 4)

        yt = np.asarray(y_true_eval, dtype=float)
        yp = np.asarray(corrected, dtype=float)
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[finite], yp[finite]
        if len(yt) > 0:
            SCATTER_MAX = 250
            if len(yt) > SCATTER_MAX:
                sidx = np.random.RandomState(42).choice(len(yt), SCATTER_MAX, replace=False)
                yt, yp = yt[sidx], yp[sidx]
            scatter_data = {"true": [round(float(v), 4) for v in yt],
                            "pred": [round(float(v), 4) for v in yp]}
    else:
        rmse_val, mae_val, eval_on_str, eval_rows, _ = _eval_metrics(
            best_preds, eval_kind, y_raw_all, tr_idx0, va_idx0)

    meta_payload = _sanitize_json({
        "model_type":      model_type,
        "feat_cols":       feat_cols_all,
        "model_feat_cols": best_info.get("used_cols", feat_cols_all) if best_info else feat_cols_all,
        "target_col":      target_column,
        "r2":              r2_report,
        "model_label":     best_name,
        "y_transform":     y_transform,
        "y_params":        y_params,
        "cat_encoders":    cat_encoders,
        "x_clip":          x_clip_bounds,
        "derived_features": derived_recipe,
        "postprocess":     {"smear": smear, "y_clip": [y_clip_lo, y_clip_hi], "round_output": round_output},
    })
    with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False)

    # ── データ品質警告 ────────────────────────────────────────────────────────
    n_val = len(df_val) if df_val is not None else 0
    data_warning = None
    if n_rows < MIN_ROWS_FOR_SPLIT:
        data_warning = f"データが {n_rows} 行のため全データで学習。R² は訓練スコアのため楽観的な値です。"
    elif eval_on_str == "oof":
        data_warning = f"OOF (交差検証) で評価。"
        if n_rows < SMALL_N_OOF_THRESH:
            data_warning = f"少ないデータ（{n_rows} 行）— " + data_warning
    elif n_val < 20:
        data_warning = f"検証セットが {n_val} 行と少なく R² が不安定な場合があります。100行以上推奨。"
    if n_target_na > 0:
        tw = f"目的変数が欠損している {n_target_na} 行を学習から除外しました。"
        data_warning = (data_warning + " " + tw) if data_warning else tw
    if n_outliers > 0:
        iqr_mult_used = OUTLIER_IQR_MULT if thorough else OUTLIER_IQR_QUICK
        ow = f"Y 外れ値 {n_outliers} 行を許容範囲内に補正しました（IQR×{iqr_mult_used}）。"
        data_warning = (data_warning + " " + ow) if data_warning else ow

    result = {
        "r2":                 r2_report,
        "r2_raw":             r2_raw,
        "rmse":               round(rmse_val, 4),
        "mae":                round(mae_val, 4),
        "best_model":         best_name,
        "model_type":         model_type,
        "feature_importance": feat_list,
        "eval_on":            eval_on_str,
        "train_rows":         len(df_train),
        "val_rows":           eval_rows,
        "target":             target_column,
        "preset":             strategy,
        "data_warning":       data_warning,
        "r2_interpretation":  _r2_interpretation(r2_report),
        "use_gp":             (model_type == 'gp'),
        "gp_format":          "pkl" if model_type == 'gp' else None,
        "scatter":            scatter_data,
        "y_range":            [round(float(np.min(y_raw_all)), 4), round(float(np.max(y_raw_all)), 4)],
        # 配布ファイル（HTML/native exe）は .treg にカテゴリエンコーダを持たないため
        # カテゴリ列を実質使えない（高-M1）。UI側でデプロイ前後に警告を出すためのフラグ。
        "cat_columns":        list(cat_encoders.keys()) if cat_encoders else [],
        # 学習時に比較した候補モデル一覧（R²降順）。以前はbest_modelしかUIに出せず、
        # 「なぜこのモデルが選ばれたか」がユーザーから見えなかった(評価レポート5視点の
        # 提案項目)。r2はcandidates内のOOF/検証スコア(=表示R²と評価データが異なる場合が
        # あるため参考値。厳密な比較はr2_rawとは別軸)。
        # 高-H3緩和(winner's curse対策): r2_std はfold別R²の標準偏差(値が大きいほど
        # fold間でスコアが不安定=選択バイアスの影響を受けやすい)。train_r2は最終モデル
        # 自身の学習データに対するR²(過学習診断用。r2との乖離が大きいほど過学習を疑う)。
        # いずれも参考値であり、フロント表示は本タスクでは対象外(フィールド追加のみ)。
        "candidate_models": [
            {
                "name": k,
                "r2": round(float(v[0]), 4) if np.isfinite(v[0]) else None,
                "model_type": v[2],
                "is_best": (k == best_name),
                "r2_std": round(_candidate_r2_std(
                    v[3], v[4].get("eval_kind") if v[4] else None,
                    cv_splits, df[target_column].values), 4),
                "train_r2": (round(float(v[4].get("train_r2")), 4)
                            if v[4] and v[4].get("train_r2") is not None else None),
            }
            for k, v in sorted(
                candidates.items(),
                key=lambda kv: kv[1][0] if np.isfinite(kv[1][0]) else -np.inf,
                reverse=True,
            )
        ],
    }

    # ── デプロイ用 .treg: 表示モデル(best_name)を最優先で書き出す ──────────────
    #    全モデル種別が .treg 対応済みのため通常は best_name がそのまま書き出せる。
    #    まれに技術的な失敗（壊れたsidecarファイル等）があった場合のみ、次善の
    #    候補（R²降順）へフォールバックする。単純にR²降順で探すと、Blend採用
    #    マージン（BLEND_MARGIN）によって表示上は単体モデルへ格下げされたのに
    #    デプロイだけBlendになる、という安全策と矛盾する逆転が起こり得るため、
    #    best_name を必ず先頭にする。
    #    smear はそのデプロイモデル自身の検証予測から再フィットする。
    _emit_progress(96, "モデルを保存中")
    exportable_candidates = {n: v for n, v in candidates.items()
                             if v[4] is not None and v[4].get("exportable", False)}
    deploy_order = [best_name] + sorted(
        [n for n in exportable_candidates if n != best_name],
        key=lambda n: exportable_candidates[n][0], reverse=True)

    exported = False
    dep_name, dep_r2 = None, None  # exportable_candidates が空の場合でも未定義参照にならないよう初期化
    for dep_name in deploy_order:
        if dep_name not in exportable_candidates:
            continue
        dep_r2, _, dep_type, dep_preds, dep_info = exportable_candidates[dep_name]
        dep_y_true = _y_true_for(dep_info.get("eval_kind", "train"), y_raw_all, tr_idx0, va_idx0)
        dep_smear, _, _, _ = _fit_postprocess_params(
            dep_preds, dep_y_true, y_transform, y_raw_all, target_is_integer)
        if dep_type == 'blend':
            ok = _export_treg_blend(model_dir, target_column, candidates, y_transform, y_params,
                                    dep_smear, (y_clip_lo, y_clip_hi), round_output, x_clip_bounds,
                                    derived_recipe=derived_recipe)
        else:
            ok = _export_treg(dep_type, model_dir, target_column, y_transform, y_params,
                              dep_smear, (y_clip_lo, y_clip_hi), round_output, x_clip_bounds,
                              derived_recipe=derived_recipe)
        if ok:
            exported = True
            if dep_name != best_name:
                print(f"[TREG] 表示モデル({best_name})の書き出しに失敗 → "
                      f"デプロイは {dep_name} (R²={dep_r2:.4f}) を使用", flush=True)
            break
    if not exported:
        print("[TREG] WARNING: デプロイ可能なモデルが存在しません", flush=True)

    # UI側(フロントエンド)が「画面の精度」と「配布ファイルの精度」が食い違う場合に
    # ユーザーへ明示できるよう、置換の有無を result に含める(旧: コンソールログのみ)。
    result["export_available"] = exported
    result["deployed_model"] = dep_name if exported else None
    result["deployed_r2"] = round(float(dep_r2), 4) if exported and dep_r2 is not None else None
    result["deploy_substituted"] = bool(exported and dep_name != best_name)

    # ── 後片付け ─────────────────────────────────────────────────────────────
    if model_type == 'blend':
        # in-app PREDICT の _predict_blend が全サブモデルを必要とするため削除しない
        print("[Python] Blend が最良のため、全サブモデルファイルを保持します", flush=True)
    else:
        _clean_model_files(model_dir, model_type)
        blend_meta_path = os.path.join(model_dir, "blend_meta.pkl")
        if os.path.exists(blend_meta_path):
            os.remove(blend_meta_path)

    # ── アトミック入替: tmp → trained_model ──────────────────────────────────
    try:
        if os.path.exists(final_model_dir):
            shutil.rmtree(final_model_dir)
        os.replace(model_dir, final_model_dir)
    except Exception as e:
        print(f"ERROR: 学習結果の配置に失敗しました（他のプロセスが使用中の可能性）: {e}", flush=True)
        sys.exit(1)

    _emit_progress(100, "完了")
    result = _sanitize_json(result)
    print(f"RESULT_JSON:{json.dumps(result, ensure_ascii=False)}", flush=True)
    print("[Python] 学習完了。", flush=True)
