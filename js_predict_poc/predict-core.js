// predict-core.js — T-regressor .treg バイナリの JS 版推論エンジン. See README.md for details.
const MT_LINEAR = 0, MT_LGBM = 1, MT_GP = 2, MT_MLP = 3, MT_LINEAR_POLY = 4, MT_BLEND = 5;
const YT_NONE = 0, YT_LOG1P = 1, YT_YEO_JOHNSON = 2;
const DOP_MUL = 0, DOP_SQ = 1, DOP_SIGN = 2;

class Reader {
    constructor(buf) {
        this.dv = new DataView(buf);
        this.pos = 0;
        this.size = buf.byteLength;
        this.fail = false;
    }
    ok() { return !this.fail && this.pos <= this.size; }
    u8()  { return this._num(1, (o) => this.dv.getUint8(o)); }
    i32() { return this._num(4, (o) => this.dv.getInt32(o, true)); }
    u32() { return this._num(4, (o) => this.dv.getUint32(o, true)); }
    f32() { return this._num(4, (o) => this.dv.getFloat32(o, true)); }
    f64() { return this._num(8, (o) => this.dv.getFloat64(o, true)); }
    _num(n, fn) {
        if (this.fail || this.pos + n > this.size) { this.fail = true; return 0; }
        const v = fn(this.pos);
        this.pos += n;
        return v;
    }
    str() {
        const len = this._num(2, (o) => this.dv.getUint16(o, true));
        if (this.fail || this.pos + len > this.size) { this.fail = true; return ""; }
        const bytes = new Uint8Array(this.dv.buffer, this.dv.byteOffset + this.pos, len);
        this.pos += len;
        return new TextDecoder("utf-8").decode(bytes);
    }
    floats(n) {
        if (this.fail || n > 1e8 || this.pos + n * 4 > this.size) { this.fail = true; return new Float32Array(0); }
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = this.dv.getFloat32(this.pos + i * 4, true);
        this.pos += n * 4;
        return out;
    }
    // blend(アンサンブル)の各メンバーは自己完結した入れ子 .treg ブロブとして埋め込まれて
    // いるため、そのバイト範囲を独立した ArrayBuffer としてコピーし、loadTreg に
    // 再帰的に渡せるようにする。
    sliceBuffer(n) {
        if (this.fail || n < 0 || this.pos + n > this.size) { this.fail = true; return new ArrayBuffer(0); }
        const start = this.dv.byteOffset + this.pos;
        const out = this.dv.buffer.slice(start, start + n);
        this.pos += n;
        return out;
    }
}

function loadTreg(buf) {
    const r = new Reader(buf);
    if (r.size < 6) throw new Error("treg too small");
    const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
    r.pos = 4;
    if (magic !== "TREG") throw new Error("bad magic (not a .treg file)");

    const model = {
        file_version: r.u8(),
        type: r.u8(),
        n_feat: r.u32(),
        derived: [],
        derived_idx: new Map(),
        y_transform: YT_NONE, yeo_lambda: 1.0,
        round_output: 0, smear: 1.0,
        y_clip_lo: -3.4e38, y_clip_hi: 3.4e38,
        x_clip_lo: [], x_clip_hi: [],
        target_col: "", feat_cols: [], medians: new Map(),
        linear: null, gp: null, mlp: null, lgbm: null,
    };
    const d = model.n_feat;
    if (model.file_version > 4) throw new Error("unsupported (future) .treg version");
    // blend(アンサンブル)は自身の直接の特徴ベクトルを持たない(各メンバーが個別に持つ)ため
    // n_feat=0 が正当な値になる。それ以外の型は従来通り 1 以上を要求する。
    if (d > 100000 || (model.type !== MT_BLEND && d < 1)) throw new Error("bad n_feat");

    if (model.file_version >= 4) {
        const nDerived = r.u32();
        for (let i = 0; i < nDerived; i++) {
            const df = {
                op: r.u8(), name: r.str(),
                col_a: r.str(), a_lo: r.f32(), a_hi: r.f32(),
                col_b: r.str(), b_lo: r.f32(), b_hi: r.f32(),
            };
            if (r.fail || df.op > 2) throw new Error("bad derived feature block");
            model.derived.push(df);
            model.derived_idx.set(df.name, model.derived.length - 1);
        }
    }

    if (model.type === MT_LGBM) {
        const nTrees = r.u32();
        r.u32();
        if (r.fail || nTrees > 100000) throw new Error("bad n_trees");
        const trees = [];
        for (let t = 0; t < nTrees; t++) {
            const n_leaves = r.u32();
            if (r.fail || n_leaves < 1 || n_leaves > (1 << 20)) throw new Error("bad n_leaves");
            const ni = n_leaves - 1;
            const split_feature = new Uint32Array(ni);
            for (let i = 0; i < ni; i++) split_feature[i] = r.u32();
            const threshold = new Float32Array(ni);
            for (let i = 0; i < ni; i++) threshold[i] = r.f32();
            const left_child = new Int32Array(ni);
            for (let i = 0; i < ni; i++) left_child[i] = r.i32();
            const right_child = new Int32Array(ni);
            for (let i = 0; i < ni; i++) right_child[i] = r.i32();
            const leaf_value = new Float32Array(n_leaves);
            for (let i = 0; i < n_leaves; i++) leaf_value[i] = r.f32();
            if (r.fail) throw new Error("truncated lgbm tree data");
            trees.push({ n_leaves, split_feature, threshold, left_child, right_child, leaf_value });
        }
        model.lgbm = { trees };
    } else if (model.type === MT_LINEAR) {
        model.linear = { mean: r.floats(d), scale: r.floats(d), coef: r.floats(d), intercept: r.f32() };
    } else if (model.type === MT_GP) {
        const mean = r.floats(d);
        const scale = r.floats(d);
        const ls = r.floats(d);
        const sv = r.f32();
        const y_mean = r.f32();
        const y_std = r.f32();
        const n_train = r.u32();
        if (r.fail || n_train < 1 || n_train * d > 1e8) throw new Error("bad gp n_train");
        const X_train = r.floats(n_train * d);
        const alpha = r.floats(n_train);
        model.gp = { n_feat: d, mean, scale, ls, sv, y_mean, y_std, n_train, X_train, alpha };
    } else if (model.type === MT_MLP) {
        const mean = r.floats(d);
        const scale = r.floats(d);
        const n_layers = r.u32();
        if (r.fail || n_layers < 1 || n_layers > 64) throw new Error("bad mlp n_layers");
        const layers = [];
        for (let i = 0; i < n_layers; i++) {
            const n_in = r.u32();
            const n_out = r.u32();
            const act = r.u8();
            if (r.fail || n_in < 1 || n_out < 1 || n_in * n_out > 1e8) throw new Error("bad mlp layer dims");
            const W = r.floats(n_in * n_out);
            const b = r.floats(n_out);
            layers.push({ n_in, n_out, act, W, b });
        }
        model.mlp = { mean, scale, layers };
    } else if (model.type === MT_LINEAR_POLY) {
        // poly-Ridge: RobustScaler(center/scale) → 多項式項(単項 or 標準化後の値どうしの
        // 積/二乗) → coef_ 内積 + intercept_。項の並びは train_bridge._light.PolynomialFeatures
        // と同一([単項(i昇順)]+[i<=jの積])で、書き出し時と同じ順に (idx_a, idx_b) を読む
        // (idx_b<0 は「単項(s[idx_a]そのまま)」を表す)。
        const center = r.floats(d);
        const scale = r.floats(d);
        const nTerms = r.u32();
        if (r.fail || nTerms > 1000000) throw new Error("bad linear_poly n_terms");
        const termA = new Int32Array(nTerms);
        const termB = new Int32Array(nTerms);
        for (let i = 0; i < nTerms; i++) {
            termA[i] = r.i32();
            termB[i] = r.i32();
        }
        const coef = r.floats(nTerms);
        const intercept = r.f32();
        if (r.fail) throw new Error("truncated linear_poly payload");
        model.linearPoly = { center, scale, termA, termB, coef, intercept };
    } else if (model.type === MT_BLEND) {
        // 各メンバーは後処理なし(smear=1, y_clip=無制限, round無し)の自己完結した
        // 入れ子 .treg ブロブ。loadTreg を再帰的に呼んでそれぞれ独立にパースする。
        const nMembers = r.u32();
        if (r.fail || nMembers > 1000) throw new Error("bad blend n_members");
        const members = [];
        for (let i = 0; i < nMembers; i++) {
            const weight = r.f32();
            const blobLen = r.u32();
            const blobBuf = r.sliceBuffer(blobLen);
            if (r.fail) throw new Error("truncated blend member blob");
            members.push({ weight, model: loadTreg(blobBuf) });
        }
        if (members.length < 2) throw new Error("blend must have >=2 members");
        model.blend = { members };
    } else {
        throw new Error(`unknown model type ${model.type}`);
    }
    if (r.fail) throw new Error("truncated model payload");

    if (model.file_version >= 2) {
        model.y_transform = r.u8();
        if (model.y_transform === YT_YEO_JOHNSON) model.yeo_lambda = r.f32();
    }
    if (model.file_version >= 3) {
        model.round_output = r.u8();
        model.smear = r.f32();
        model.y_clip_lo = r.f32();
        model.y_clip_hi = r.f32();
        const nClip = r.u32();
        for (let i = 0; i < nClip; i++) {
            model.x_clip_lo.push(r.f32());
            model.x_clip_hi.push(r.f32());
        }
    }

    model.target_col = r.str();
    const nFc = r.u32();
    if (r.fail || nFc > 100000) throw new Error("bad feat_cols count");
    for (let i = 0; i < nFc; i++) model.feat_cols.push(r.str());
    const nMed = r.u32();
    if (r.fail || nMed > 100000) throw new Error("bad medians count");
    for (let i = 0; i < nMed; i++) {
        const col = r.str();
        if (r.fail || r.pos + 8 > r.size) throw new Error("truncated medians");
        model.medians.set(col, r.f64());
    }
    if (!r.ok()) throw new Error("treg trailing data corrupt");
    return model;
}

function clippedSource(row, col, lo, hi) {
    const v = row[col];
    if (v === undefined || v === null || Number.isNaN(v)) return NaN;
    return Math.min(Math.max(v, lo), hi);
}

function computeDerived(df, row) {
    const a = clippedSource(row, df.col_a, df.a_lo, df.a_hi);
    if (Number.isNaN(a)) return NaN;
    let v;
    if (df.op === DOP_MUL) {
        const b = clippedSource(row, df.col_b, df.b_lo, df.b_hi);
        if (Number.isNaN(b)) return NaN;
        v = a * b;
    } else if (df.op === DOP_SQ) {
        v = a * a;
    } else if (df.op === DOP_SIGN) {
        v = Math.sign(a);
    } else {
        return NaN;
    }
    return Number.isFinite(v) ? v : NaN;
}

function predictLgbm(lgbmModel, x) {
    let sum = 0.0;
    for (const tree of lgbmModel.trees) {
        if (tree.n_leaves === 1) { sum += tree.leaf_value[0]; continue; }
        let node = 0;
        for (;;) {
            const feat = tree.split_feature[node];
            const thr = tree.threshold[node];
            const next = (x[feat] <= thr) ? tree.left_child[node] : tree.right_child[node];
            if (next < 0) { sum += tree.leaf_value[-(next + 1)]; break; }
            node = next;
        }
    }
    return sum;
}

function predictLinear(m, x, d) {
    let s = m.intercept;
    for (let i = 0; i < d; i++) {
        const sc = (x[i] - m.mean[i]) / (m.scale[i] + 1e-8);
        s += m.coef[i] * sc;
    }
    return s;
}

function predictGp(m, x) {
    const d = m.n_feat;
    const xs = new Float32Array(d);
    for (let i = 0; i < d; i++) xs[i] = (x[i] - m.mean[i]) / (m.scale[i] + 1e-8);
    let yNorm = 0.0;
    for (let i = 0; i < m.n_train; i++) {
        let sq = 0.0;
        for (let j = 0; j < d; j++) {
            const diff = xs[j] - m.X_train[i * d + j];
            const lsJ = m.ls[j];
            sq += (diff / lsJ) * (diff / lsJ);
        }
        yNorm += m.sv * Math.exp(-0.5 * sq) * m.alpha[i];
    }
    return yNorm * m.y_std + m.y_mean;
}

function predictMlp(m, x, d) {
    let h = new Float32Array(d);
    for (let i = 0; i < d; i++) h[i] = (x[i] - m.mean[i]) / (m.scale[i] + 1e-8);
    for (const layer of m.layers) {
        const outV = new Float32Array(layer.n_out);
        for (let j = 0; j < layer.n_out; j++) {
            let val = layer.b[j];
            for (let k = 0; k < layer.n_in; k++) val += h[k] * layer.W[k * layer.n_out + j];
            outV[j] = val;
        }
        if (layer.act === 0) {
            for (let j = 0; j < outV.length; j++) outV[j] = Math.max(0.0, outV[j]);
        }
        h = outV;
    }
    return h.length === 0 ? 0.0 : h[0];
}

function predictLinearPoly(m, x, d) {
    const s = new Float32Array(d);
    for (let i = 0; i < d; i++) s[i] = (x[i] - m.center[i]) / (m.scale[i] + 1e-8);
    let sum = m.intercept;
    const n = m.coef.length;
    for (let t = 0; t < n; t++) {
        const a = m.termA[t], b = m.termB[t];
        const val = (b < 0) ? s[a] : s[a] * s[b];
        sum += m.coef[t] * val;
    }
    return sum;
}

function predictBlend(blend, row) {
    // 各メンバーは自身の型・特徴量・y変換・(後処理なしの)個別予測を完結して持つため、
    // predictRow を再帰的に呼んで得た「実スケールの予測」を重み付き和するだけでよい。
    // 最終的な smear/y_clip/round_output は呼び出し元の predictRow(外側モデル)側で
    // 一度だけ適用される。
    let sum = 0.0;
    for (const mem of blend.members) {
        sum += mem.weight * predictRow(mem.model, row);
    }
    return sum;
}

function buildFeatureVector(model, row) {
    const d = model.feat_cols.length;
    const x = new Float32Array(d);
    for (let i = 0; i < d; i++) {
        const name = model.feat_cols[i];
        let val = NaN;
        if (model.derived_idx.has(name)) {
            val = computeDerived(model.derived[model.derived_idx.get(name)], row);
        } else if (row[name] !== undefined && row[name] !== null && !Number.isNaN(row[name])) {
            val = row[name];
        }
        if (Number.isNaN(val)) {
            val = model.medians.has(name) ? model.medians.get(name) : 0.0;
        }
        if (i < model.x_clip_lo.length) {
            val = Math.min(Math.max(val, model.x_clip_lo[i]), model.x_clip_hi[i]);
        }
        x[i] = val;
    }
    return x;
}

function yeoJohnsonInv(y, lam) {
    if (y >= 0) {
        if (Math.abs(lam) < 1e-6) return Math.expm1(y);
        return Math.pow(lam * y + 1.0, 1.0 / lam) - 1.0;
    } else {
        const lam2 = 2.0 - lam;
        if (Math.abs(lam2) < 1e-6) return 1.0 - Math.exp(-y);
        return 1.0 - Math.pow(-lam2 * y + 1.0, 1.0 / lam2);
    }
}

function invYTransform(pred, yt, lam) {
    if (yt === YT_LOG1P) return Math.expm1(pred);
    if (yt === YT_YEO_JOHNSON) return yeoJohnsonInv(pred, lam);
    return pred;
}

function roundHalfAwayFromZero(x) {
    return Math.sign(x) * Math.floor(Math.abs(x) + 0.5);
}

function predictRow(model, row) {
    const x = buildFeatureVector(model, row);
    const d = model.feat_cols.length;
    let pred;
    if (model.type === MT_LGBM) {
        pred = predictLgbm(model.lgbm, x);
    } else if (model.type === MT_LINEAR) {
        pred = predictLinear(model.linear, x, d);
    } else if (model.type === MT_GP) {
        pred = predictGp(model.gp, x);
    } else if (model.type === MT_MLP) {
        pred = predictMlp(model.mlp, x, d);
    } else if (model.type === MT_LINEAR_POLY) {
        pred = predictLinearPoly(model.linearPoly, x, d);
    } else if (model.type === MT_BLEND) {
        pred = predictBlend(model.blend, row);
    } else {
        throw new Error("unknown model type");
    }
    pred = invYTransform(pred, model.y_transform, model.yeo_lambda);
    pred *= model.smear;
    pred = Math.min(Math.max(pred, model.y_clip_lo), model.y_clip_hi);
    if (model.round_output) pred = roundHalfAwayFromZero(pred);
    return pred;
}

module.exports = { loadTreg, predictRow, roundHalfAwayFromZero };
