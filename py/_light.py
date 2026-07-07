# _light.py — sklearn / scipy の軽量自前代替（numpy のみ依存）
#
# 依存スリム化(Phase3+)のため、metrics / CV / 前処理 / RidgeCV /
# PowerTransformer / permutation / nnls / skew を numpy だけで実装する。
# API（メソッド名・属性名）は sklearn/scipy と互換にし、呼び出し側と
# .treg エクスポート（scaler.center_/scale_, model.coef_ 等）を不変に保つ。
#
# 数値挙動が sklearn/scipy と一致することは tests/test_light_parity.py で検証する。
import numpy as np
from numpy.linalg import solve, lstsq, cholesky  # noqa: F401 (choleskyはGP自前化で使用)


# ═══════════ metrics ═══════════
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mu = float(y_true.mean())
    ss_tot = float(np.sum((y_true - mu) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


# ═══════════ 交差検証 (split インターフェースのみ) ═══════════
class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits; self.shuffle = shuffle; self.random_state = random_state

    def split(self, X, y=None):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            np.random.RandomState(self.random_state).shuffle(idx)
        folds = np.array_split(idx, self.n_splits)
        for i in range(self.n_splits):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield train, test


class StratifiedKFold:
    """各クラス(bin)を n_splits に巡回配分する層化 K-Fold。
    sklearn とビット同一ではないが「各 fold に各クラスが均等に入る」層化目的は満たす。"""
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits; self.shuffle = shuffle; self.random_state = random_state

    def split(self, X, y):
        y = np.asarray(y); n = len(y)
        rng = np.random.RandomState(self.random_state)
        fold_id = np.empty(n, dtype=int)
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            if self.shuffle:
                rng.shuffle(ci)
            # クラス内で 0..n_splits-1 を巡回付与
            fold_id[ci] = np.arange(len(ci)) % self.n_splits
        for f in range(self.n_splits):
            test = np.where(fold_id == f)[0]
            train = np.where(fold_id != f)[0]
            yield train, test


# ═══════════ スケーラ ═══════════
class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        s = X.std(axis=0)
        self.scale_ = np.where(s < 1e-12, 1.0, s)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScaler:
    """median 中心・IQR スケール（sklearn.RobustScaler 既定と同じ 25-75 パーセンタイル）。"""
    def fit(self, X):
        X = np.asarray(X, float)
        self.center_ = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.scale_ = np.where(iqr < 1e-12, 1.0, iqr)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.center_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ═══════════ 多項式特徴 (degree=2) ═══════════
class PolynomialFeatures:
    """degree=2, include_bias=False。列順を sklearn と一致させる:
    [x_i (単項, i昇順)] + [x_i*x_j (i<=j, (i,j)昇順)]。"""
    def __init__(self, degree=2, include_bias=False, interaction_only=False):
        assert degree == 2 and not include_bias
        self.interaction_only = interaction_only

    def fit(self, X):
        self.n_input_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, float); n, d = X.shape
        cols = [X[:, i] for i in range(d)]  # 単項
        for i in range(d):                  # 2次（i<=j）
            jstart = i + 1 if self.interaction_only else i
            for j in range(jstart, d):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, names):
        names = list(names); d = len(names); out = list(names)
        for i in range(d):
            jstart = i + 1 if self.interaction_only else i
            for j in range(jstart, d):
                out.append(f"{names[i]} {names[j]}" if i != j else f"{names[i]}^2")
        return np.array(out, dtype=object)


# ═══════════ RidgeCV (αグリッド + K-Fold CV) ═══════════
class RidgeCV:
    """中心化 Ridge を αグリッド×K-Fold で選択（切片は平均差で復元）。
    sklearn.RidgeCV と外部テスト R² が一致することを PoC で確認済み。"""
    def __init__(self, alphas, cv=5):
        self.alphas = list(alphas); self.cv = cv

    def _fit_alpha(self, X, y, a):
        mx = X.mean(0); my = float(y.mean())
        Xc = X - mx; yc = y - my
        d = X.shape[1]
        W = solve(Xc.T @ Xc + a * np.eye(d), Xc.T @ yc)
        return W, mx, my

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        n = len(y)
        k = min(self.cv, n)
        idx = np.random.RandomState(42).permutation(n)
        folds = np.array_split(idx, k)
        best_a, best_mse = self.alphas[0], np.inf
        for a in self.alphas:
            mse = 0.0
            for i in range(k):
                te = folds[i]
                tr = np.concatenate([folds[j] for j in range(k) if j != i])
                W, mx, my = self._fit_alpha(X[tr], y[tr], a)
                pred = (X[te] - mx) @ W + my
                mse += float(np.sum((y[te] - pred) ** 2))
            if mse < best_mse:
                best_mse, best_a = mse, a
        W, mx, my = self._fit_alpha(X, y, best_a)
        self.coef_ = W
        self.intercept_ = float(my - mx @ W)
        self.alpha_ = float(best_a)
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


# ═══════════ PowerTransformer (Yeo-Johnson, standardize=False) ═══════════
class PowerTransformer:
    def __init__(self, method='yeo-johnson', standardize=False):
        assert method == 'yeo-johnson'
        self.standardize = standardize
        self.lambdas_ = None

    @staticmethod
    def _yj(x, lam):
        x = np.asarray(x, float); out = np.empty_like(x); pos = x >= 0
        if abs(lam) < 1e-6:
            out[pos] = np.log1p(x[pos])
        else:
            out[pos] = ((x[pos] + 1.0) ** lam - 1.0) / lam
        if abs(lam - 2.0) < 1e-6:
            out[~pos] = -np.log1p(-x[~pos])
        else:
            out[~pos] = -(((-x[~pos] + 1.0) ** (2.0 - lam)) - 1.0) / (2.0 - lam)
        return out

    def _neg_llf(self, lam, x):
        n = len(x); xt = self._yj(x, lam); var = float(xt.var())
        if var < 1e-12:
            return 1e10
        llf = -0.5 * n * np.log(var) + (lam - 1.0) * float(np.sum(np.sign(x) * np.log1p(np.abs(x))))
        return -llf

    def _optimize_lambda(self, x):
        # 黄金分割探索 [-2, 2]（sklearn は scipy.brent。結果は近接）
        lo, hi = -2.0, 2.0
        gr = (np.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
        fc = self._neg_llf(c, x); fd = self._neg_llf(d, x)
        for _ in range(60):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo); fc = self._neg_llf(c, x)
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo); fd = self._neg_llf(d, x)
            if abs(hi - lo) < 1e-6:
                break
        return (lo + hi) / 2

    def fit(self, X):
        X = np.asarray(X, float)
        self.lambdas_ = np.array([self._optimize_lambda(col) for col in X.T])
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return np.column_stack([self._yj(X[:, i], self.lambdas_[i]) for i in range(X.shape[1])])

    @staticmethod
    def _yj_inv(y, lam):
        y = np.asarray(y, float); out = np.empty_like(y); pos = y >= 0
        if abs(lam) < 1e-6:
            out[pos] = np.expm1(y[pos])
        else:
            out[pos] = (lam * y[pos] + 1.0) ** (1.0 / lam) - 1.0
        if abs(lam - 2.0) < 1e-6:
            out[~pos] = 1.0 - np.exp(-y[~pos])
        else:
            out[~pos] = 1.0 - (-(2.0 - lam) * y[~pos] + 1.0) ** (1.0 / (2.0 - lam))
        return out

    def inverse_transform(self, X):
        X = np.asarray(X, float)
        return np.column_stack([self._yj_inv(X[:, i], self.lambdas_[i]) for i in range(X.shape[1])])


# ═══════════ permutation importance ═══════════
def permutation_importance(estimator, X, y, n_repeats=5, random_state=42):
    X = np.asarray(X, float); y = np.asarray(y, float)
    rng = np.random.RandomState(random_state)
    base = r2_score(y, estimator.predict(X))
    n, d = X.shape
    imps = np.zeros(d)
    for j in range(d):
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(n), j]
            drops[r] = base - r2_score(y, estimator.predict(Xp))
        imps[j] = drops.mean()

    class _Result:
        pass
    res = _Result(); res.importances_mean = imps
    return res


# ═══════════ scipy 代替 ═══════════
def skew(x):
    x = np.asarray(x, float); m = x.mean(); s = x.std()
    return 0.0 if s < 1e-12 else float(np.mean(((x - m) / s) ** 3))


# ═══════════ MLP (numpy forward + Adam, sklearn 非依存) ═══════════
class LightMLP:
    """全結合 NN（relu 隠れ層 + 線形出力）を numpy + Adam で学習。
    学習安定化のため内部で y を標準化し、最終層の重み・バイアスに畳み込んで
    predict は変換後 y スケールを直接返す（native C++ predict_mlp と互換）。
    sklearn.MLPRegressor 互換に coefs_(list of (n_in,n_out)) / intercepts_ / n_iter_ を公開。"""
    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=1500,
                 learning_rate_init=1e-2, random_state=42):
        self.hidden = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.lr = float(learning_rate_init)
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, float); y = np.asarray(y, float)
        ym = float(y.mean()); ys = float(y.std())
        if ys < 1e-12:
            ys = 1.0
        yn = (y - ym) / ys
        dims = [X.shape[1]] + list(self.hidden) + [1]
        Ws = [rng.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i]) for i in range(len(dims) - 1)]
        bs = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        mW = [np.zeros_like(w) for w in Ws]; vW = [np.zeros_like(w) for w in Ws]
        mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]
        b1, b2, eps = 0.9, 0.999, 1e-8
        N = len(X); lam = self.alpha; nL = len(Ws)
        it = 0
        best_loss = np.inf; no_improve = 0
        patience, tol = 15, 1e-4
        for t in range(1, self.max_iter + 1):
            it = t
            a = [X]; z = []
            for i in range(nL):
                zz = a[-1] @ Ws[i] + bs[i]; z.append(zz)
                a.append(np.maximum(0.0, zz) if i < nL - 1 else zz)
            pred = a[-1].ravel()
            # 収束済みなら早期終了（最大反復まで無駄に回さない。quick/thorough共通で有効）
            loss = float(np.mean((pred - yn) ** 2))
            if loss < best_loss - tol:
                best_loss = loss; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
            delta = (2.0 / N) * (pred - yn)[:, None]
            gW = [None] * nL; gb = [None] * nL
            for i in reversed(range(nL)):
                gW[i] = a[i].T @ delta + lam * Ws[i]
                gb[i] = delta.sum(0)
                if i > 0:
                    delta = (delta @ Ws[i].T) * (z[i - 1] > 0)
            for i in range(nL):
                mW[i] = b1 * mW[i] + (1 - b1) * gW[i]; vW[i] = b2 * vW[i] + (1 - b2) * gW[i] ** 2
                Ws[i] -= self.lr * (mW[i] / (1 - b1 ** t)) / (np.sqrt(vW[i] / (1 - b2 ** t)) + eps)
                mb[i] = b1 * mb[i] + (1 - b1) * gb[i]; vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
                bs[i] -= self.lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)
        # y 標準化を最終層に畳み込む: raw = norm*ys + ym
        Ws[-1] = Ws[-1] * ys
        bs[-1] = bs[-1] * ys + ym
        self.coefs_ = Ws
        self.intercepts_ = bs
        self.n_iter_ = it
        return self

    def predict(self, X):
        a = np.asarray(X, float)
        nL = len(self.coefs_)
        for i in range(nL):
            a = a @ self.coefs_[i] + self.intercepts_[i]
            if i < nL - 1:
                a = np.maximum(0.0, a)
        return a.ravel()


class LightPipeline:
    """sklearn.Pipeline の最小代替。steps=[(name, step), ...]。最終ステップ以外は
    fit_transform/transform、最終ステップは fit/predict。pipeline['name'] でアクセス。"""
    def __init__(self, steps):
        self.steps = list(steps)
        self._d = dict(steps)

    def __getitem__(self, key):
        return self._d[key]

    def fit(self, X, y=None):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt)
        self.steps[-1][1].fit(Xt, y)
        return self

    def predict(self, X):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return self.steps[-1][1].predict(Xt)


# ═══════════ Gaussian Process (ARD-RBF, scipy/sklearn 非依存) ═══════════
def _gp_rbf(Xa, Xb, ls):
    """ARD-RBF カーネル行列 K[i,j] = exp(-0.5 * Σ_k ((Xa_ik - Xb_jk)/ls_k)^2)。"""
    d = (Xa[:, None, :] - Xb[None, :, :]) / ls
    return np.exp(-0.5 * (d ** 2).sum(-1))


def gp_nll_and_grad(params, Xs, yt, d):
    """ARD-RBF + White の負対数周辺尤度と解析勾配。
    params = [log ls(d), log sigma_var, log noise_var]。Xs:標準化済み, yt:正規化済み y。"""
    n = len(Xs)
    ls = np.exp(params[:d]).clip(1e-3, 100.)
    sv = float(np.exp(params[d]).clip(1e-4, 1e4))
    nv = float(np.exp(params[d + 1]).clip(1e-6, 10.))
    diff = Xs[:, None, :] - Xs[None, :, :]
    scaled = (diff / ls) ** 2
    rbf = np.exp(-0.5 * scaled.sum(-1))
    K = sv * rbf + (nv + 1e-4) * np.eye(n)
    try:
        L = cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros(d + 2)
    a = solve(L.T, solve(L, yt))
    logdet = 2.0 * np.log(np.diag(L)).sum()
    nll = float(0.5 * (yt @ a + logdet))
    Kinv = solve(L.T, solve(L, np.eye(n)))
    W = Kinv - np.outer(a, a)
    Wr = W * (sv * rbf)
    g = np.empty(d + 2)
    g[:d] = 0.5 * np.einsum('ij,ijk->k', Wr, scaled)
    g[d] = 0.5 * np.sum(Wr)
    g[d + 1] = 0.5 * nv * np.trace(W)
    return nll, g


def minimize_lbfgs(func_grad, x0, max_iter=200, m=10, tol=1e-5):
    """L-BFGS（two-loop recursion + Armijo 線探索）。scipy L-BFGS-B の軽量代替。
    func_grad(x)->(f,grad)。無制約（境界は func 側 clip で担保）。"""
    x = np.asarray(x0, float).copy()
    f, g = func_grad(x)
    s_list, y_list, rho_list = [], [], []
    for _ in range(max_iter):
        if float(np.linalg.norm(g)) < tol:
            break
        q = g.copy(); alphas = []
        for s, yy, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a_i = rho * float(s @ q); alphas.append(a_i); q = q - a_i * yy
        if y_list:
            gamma = float(s_list[-1] @ y_list[-1]) / float(y_list[-1] @ y_list[-1])
            q = q * gamma
        for s, yy, rho, a_i in zip(s_list, y_list, rho_list, reversed(alphas)):
            b_i = rho * float(yy @ q); q = q + (a_i - b_i) * s
        dvec = -q
        gd = float(g @ dvec)
        if gd >= 0:
            dvec = -g; gd = float(g @ dvec)
        t = 1.0; c = 1e-4; xnew = x; fnew = f; gnew = g; ok = False
        for _ls in range(40):
            xnew = x + t * dvec
            fnew, gnew = func_grad(xnew)
            if fnew <= f + c * t * gd:
                ok = True; break
            t *= 0.5
        if not ok:
            break
        s = xnew - x; yv = gnew - g; sy = float(s @ yv)
        if sy > 1e-10:
            s_list.append(s); y_list.append(yv); rho_list.append(1.0 / sy)
            if len(s_list) > m:
                s_list.pop(0); y_list.pop(0); rho_list.pop(0)
        x, f, g = xnew, fnew, gnew
    return x, f


class LightGP:
    """ARD-RBF ガウス過程回帰（cholesky ソルバ、sklearn 非依存）。
    ハイパラ (length_scale, sigma_var, noise_var) は外部で最適化して与える。
    .treg エクスポート用に length_scale/sigma_var/X_train_/alpha_/y_mean_/y_std_ を公開。"""
    def __init__(self, length_scale, sigma_var, noise_var):
        self.length_scale = np.atleast_1d(np.asarray(length_scale, float))
        self.sigma_var = float(sigma_var)
        self.noise_var = float(noise_var)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.y_mean_ = float(y.mean())
        ys = float(y.std())
        self.y_std_ = ys if ys > 1e-12 else 1.0
        yn = (y - self.y_mean_) / self.y_std_
        n = len(X)
        K = self.sigma_var * _gp_rbf(X, X, self.length_scale) + (self.noise_var + 1e-4) * np.eye(n)
        self.L_ = cholesky(K)
        self.alpha_ = solve(self.L_.T, solve(self.L_, yn))
        self.X_train_ = X
        return self

    def predict(self, X):
        Ks = self.sigma_var * _gp_rbf(np.asarray(X, float), self.X_train_, self.length_scale)
        return (Ks @ self.alpha_) * self.y_std_ + self.y_mean_


def nnls(A, b, max_iter=None):
    """Lawson-Hanson active-set NNLS。戻り値は (x, rnorm) で scipy.optimize.nnls 互換。"""
    A = np.asarray(A, float); b = np.asarray(b, float)
    m, n = A.shape
    if max_iter is None:
        max_iter = 3 * n
    x = np.zeros(n); P = np.zeros(n, bool)
    w = A.T @ (b - A @ x)
    it = 0
    while (not P.all()) and ((~P).any() and w[~P].max() > 1e-10):
        it += 1
        if it > max_iter:
            break
        idx = np.where(~P)[0]; j = idx[np.argmax(w[idx])]; P[j] = True
        while True:
            Ap = A[:, P]
            s_p = lstsq(Ap, b, rcond=None)[0]
            if (s_p > 0).all():
                x[P] = s_p; break
            sp_full = np.zeros(n); sp_full[P] = s_p
            mask = P & (sp_full <= 0)
            alpha = (x[mask] / (x[mask] - sp_full[mask])).min()
            x = x + alpha * (sp_full - x)
            P[x <= 1e-12] = False
        w = A.T @ (b - A @ x)
    rnorm = float(np.linalg.norm(b - A @ x))
    return x, rnorm
