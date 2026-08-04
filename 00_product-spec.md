# 案件仕様書（SoT）— Kohei Shintani Portfolio (Motion Rebuild)

> 実データのみ。経歴・受賞・論文・特許は `260705_KoheiPortfolio`（公開中サイト、GitHub: KS278810/website）
> および `260706_MyWeb/spec.md` の確定事項をそのまま継承する。**新規の数値・肩書きは作らない。**

作成: 2026-08-01 / 象限: **特殊（生成系）** — `260706_MyWeb/RADICAL-PLAN_260801.md` の Plan A に基づく
先行案件: `260706_MyWeb/`（調査・批評資産として継続参照。ただし direction_A-F の判断は不採用）

---

## 1. 製品基本情報

| 項目 | 内容 |
|---|---|
| 名称 | Kohei Shintani（新谷 光平）個人ポートフォリオサイト |
| 何者か | AI Researcher / Vehicle Engineer。Toyota Manager → 東大客員研究員 → Toyota Research Institute (TRI) Technical Advisor |
| 提供 | 自己紹介・研究実績・実装力の提示（非商用） |

## 2. ターゲット（一人に絞る）

**採用担当・ヘッドハンター**（2026-08-01 本人確認済み）

- 主要デバイスは**デスクトップ**（同上確認）
- 実績リストより「この人が何を作れるか」を短時間で判断したい
- 技術用語は通じる相手（AI/機械学習/自動車工学のバックグラウンドを持つか、それを評価する側）

## 3. Before → After（1文）

**Before**: 経歴・論文・受賞は読めば分かるが、「実装力そのもの」はテキストからは伝わらない。

**After**: サイトを開いた瞬間に実装力が伝わり、経歴を"通過"しながら専門性の広がりを体感できる。

## 4. 想定読者レベル

**C: 技術がわかる相手**。専門用語（SDF, Bayesian Active Learning, GPGPU 等）はそのまま使ってよい。
英日混在は既存サイトに合わせ英語主体＋日本語学会誌名等はそのまま表記。

## 5. コア（3本柱・研究軸を流用）

`260705_KoheiPortfolio/research.html` の3軸をそのまま使う（創作しない）:

1. **Generative AI for 3D Shape Design** — SDFベースの3D生成（VehicleSDF, WheelSDF）
2. **Set-Based Concurrent Engineering** — Bayesian Active Learningによる実現可能領域の探索
3. **Structural Optimization** — トポロジー最適化、ブレーキ鳴き抑制、望遠鏡構造

## 6. 数値・固有名詞の扱い

| 項目 | 方針 |
|---|---|
| 受賞歴10件・論文/特許全リスト・経歴 | `260705_KoheiPortfolio` の記載を一切改変せず流用 |
| 連絡先・第一導線 | `260706_MyWeb/spec.md` §1 の確定事項通り **LinkedIn を第一導線** |
| 実写真 | `KoheiShintani.jpg`（GitHub `KS278810/website/figures/` に実在、SHA `eaa2313d`）。
| | **⚠️ ローカルの `260705_KoheiPortfolio/figures/` は空フォルダで実体無し。本ビルドでは `material/raw/KoheiShintani.jpg` にオーナー本人が配置するまで、About面は画像なしテキストのみで表示するフォールバックにする（ストック画像・生成画像で代替しない）** |
| Hero粒子の粒子数等 | 実装値をそのまま記載（§6b） |

## 6b. metrics（HTML に出す数値の登録簿）

```yaml
metrics:
  - name: 受賞数
    value: "10"
    unit: 件
    source: 260705_KoheiPortfolio/index.html の Awards & Honors リスト実数
    owner_verified: true
  - name: 粒子数_デスクトップ
    value: "65,536"
    unit: 個
    source: 実装値（_shared/components/generative-hero/ の desktop ティア 256×256）
    owner_verified: false
    note: コードから導出した事実。実績数値ではない
  - name: 研究軸の数
    value: "3"
    unit: つ
    source: 260705_KoheiPortfolio/research.html の research-card 実数
    owner_verified: true
```

## 7. トーン

| 項目 | 内容 |
|---|---|
| 哲学 | 「サイト自体が実装力の証明」。情報密度は落とさず、動きの層を重ねる |
| 目指す感情 | 「経歴は分かっていたが、この人は本当に作れる人だ」 |
| 絶対避ける | テンプレ感／Editorial Noir（ダーク×金×製図グリッド）の再来／旧SaaSテンプレ（Inter一本・shadcn角丸）の再来（`260706_MyWeb/spec.md` §2.1 の確定済み回避事項を継承） |
| キャラクター | レベル0（マスコット等なし）。生成ビジュアルと本人写真が主役 |

## 8. 制約・公開条件

- 実機（中位 Android + iOS Safari）での確認が完了するまで公開不可（`decision-rules.md` 重量級技法条件4）
- `min-height:100vh` 不使用。`min(100svh, 900px)` 等でクランプ
- `prefers-reduced-motion` は全アニメーションで「置き換え」実装必須

---

## 関連
- 前提確認の記録: `260706_MyWeb/RADICAL-PLAN_260801.md` §7
- 設計書: `brief/00_design-brief.md`
- **モーション仕様: `brief/03b_motion-spec.md`**
- 判断記録: `brief/decisions.md`
- 実データ一次情報: `260705_KoheiPortfolio/*.html`, `260706_MyWeb/spec.md`
