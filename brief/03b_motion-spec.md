# モーション仕様書 — Kohei Shintani Portfolio (Motion Rebuild)

プロンプト: `_prompts/core/06b_motion-spec.md` / 判定基準: `motion-catalog.md`

> **本案件は「派手」が正解の側**（`260706_MyWeb/RADICAL-PLAN_260801.md` Plan A・本人確認済み）。
> `260706_MyWeb` の direction_A〜F 全6方向がこの工程を一度も通さず density「低」に着地した反省を踏まえ、
> `260726_folio_prisma` / `260726_type_tomoshi` と同じ手順で全項目を明示的に判定する。

---

## 1. モーション方針

- **動きが担う役割**: **サイト自体を実装力の証明にする。** 経歴は「読ませる」が、実装力は「感じさせる」
- **絶対に避ける印象**: **テンプレ感**（同時に「学術サイトなのにやりすぎ」も避ける。§3cで尺を絞る根拠にする）
- **象限**: 特殊（生成系）→ **density「高」**（`decisions.md` D2 で確定）

---

## 2. デバイス・ブラウザ前提

| 項目 | 記入 |
|---|---|
| 主要流入デバイス | **デスクトップ中心**（本人確認・`decisions.md` D2） |
| 対応ブラウザ下限 | PC Chrome/Edge/Safari/Firefoxの現行2版。モバイルは軽量ティアで閲覧可能にする |
| Firefox でスクロール連動が効かないことを許容できるか | **Yes（装飾のみに限る）** |

---

## 3. 技法カタログ 網羅チェック

| Tier | 技法 | ○/× | 理由（× のときは必須） |
|---|---|---|---|
| 0 | hover / focus-visible / active 応答 | **○** | 必須 |
| 0 | IntersectionObserver フェードイン | **○** | 必須 |
| 0 | `text-wrap: balance / pretty` | **○** | 見出し・リードの句中改行防止 |
| 0 | `prefers-reduced-motion` 分岐 | **○** | 必須 |
| 0 | `font-variant-numeric: tabular-nums` | **○** | 受賞数・粒子数のカウントアップに使用 |
| 1 | View Transitions（same-document） | × | 1ページ構成でビュー切替が発生しない（`type_tomoshi`と同判断） |
| 1 | CSS scroll-driven animations | **○（条件付き）** | 背景の章インデックス表示のみ装飾利用。`@supports`で囲み、Firefoxでは表示済み状態を最終形にする |
| 1 | `clip-path` / `mask` リビール | **○** | Research/Hobbiesのカードを斜めにワイプ |
| 1 | CSS scroll-snap | × | 縦読みの流れを断つ。sticky ピン留めで代替 |
| 1 | `@property` グラデーションアニメ | **○** | セクション見出しの発光アクセント。非対応でも静的に成立 |
| 1 | `linear()` バネイージング | × | Safari対応が未確定（motion-catalog §5-1）。cubic-bezierで足りる |
| 1 | `@starting-style` + `allow-discrete` | × | 対応状況未検証。使う場面（dialog/popover）が本サイトにない |
| 2a | チャット / UIデモの自動再生 | × | 会話系プロダクトではない |
| 2a | タイピングインジケータ | × | 同上 |
| 2a | マスコット・アイコンの微小ループ | × | キャラクターレベル0（仕様書§7） |
| 2a | 数値カウントアップ | **○** | 受賞数「10」・粒子数「65,536」。threshold 0.85（B1） |
| 2a | スクロール進行バー | **○** | 全体尺が長くなるため現在地表示が必要 |
| 2a | stagger（順次出現） | **○** | 研究3軸カード・受賞リストの順次出現 |
| 2b | GPGPU パーティクル | **○** | **主役**。4条件チェックは下記 |
| 2b | WebGL ホバー歪み / シェーダ背景 | × | 歪ませる写真がない。粒子Heroで代替 |
| 2b | GSAP + ScrollTrigger | × | 採用するスクロール連動はIntersectionObserver + rAFで足りる。依存を増やさない（`folio_prisma`と同判断） |
| 2b | kinetic typography / テキスト分割 | **○（条件付き）** | 「Toyota→東大→TRI」の転機語のみ帯で流す。**日本語は文節単位**。停止ボタン必須（WCAG 2.2.2） |
| 2b | sticky ピン留めストーリーテリング | **○** | 経歴の転換をscrubで見せる（`type_tomoshi`と同型） |
| 2b | カスタムカーソル / magnetic button | × | 得るものが雰囲気だけ。ネイティブカーソルを消すのはa11y上マイナス |
| 2b | Lottie / Rive | × | ランタイム追加不要。GPU + CSSで足りる |
| 2b | スクロール scrub（進行度直結） | **○** | 経歴転換・研究軸の展開に使用。「派手さ」の主成分（motion-catalog §2c） |
| 2b | 3D transform（perspective / rotateY） | **○（hoverのみ）** | 論文/特許カードのhoverで奥行きを軽く付与。reduced-motionで無効化 |
| 1 | backdrop-filter レイヤリング | **○** | 章インデックス・進行バーを静止パネルとして浮かせる。連続アニメはしない |
| 2a | ループ動画 KV / 背景 | × | 動画素材がない。GPU生成で代替 |
| 1 | 章立てスクロールナラティブ | **○** | Hero / Career / Research / Hobbies の章番号とインデックスを可視化 |
| 2b | 慣性スクロール（Lenis） | × | ネイティブスクロールで十分。依存を増やさない方針（D5, decisions.md）と整合 |

**× の理由の要点**: チャット系・マスコット系はキャラクターレベル0の仕様書方針と衝突。GSAP/Lenis/Lottieは「依存を増やさない」という本サイトの姿勢（研究者としての実装の堅実さ）と衝突するため不採用。

---

## 3b. 採用エフェクト一覧

| # | 技法 | Tier | 適用箇所 | 誰のどの認知を助けるか | 実装手段 | 確度 |
|---|---|---|---|---|---|---|
| 1 | hover / focus / active | 0 | 全インタラクティブ要素 | 操作可能性の判別 | CSS | ★★★ |
| 2 | IntersectionObserver フェード | 0 | 各セクション | 密度の高い画面で1かたまりずつ提示 | JS | ★★★ |
| 3 | 禁則処理 / `text-wrap` | 0 | 見出し・リード | 英日混在の行末孤立を防ぐ | CSS | ★★★ |
| 4 | reduced-motion 分岐 | 0 | 全体 | 動きが苦手な人に静止版を出す | CSS + JS | ★★★ |
| 5 | `tabular-nums` | 0 | 受賞数・粒子数 | カウントアップ中に桁が揺れない | CSS | ★★★ |
| 6 | **GPGPU 粒子 Hero** | 2b | Hero（CH.00） | **3秒で実装力を示す**。"Kohei Shintani"の字形に集合→カーソルで散る→クリックで衝撃波 | `_shared/components/generative-hero/` | ★★★（実装済みコンポーネント） |
| 7 | **scrub① Career タイムライン** | 2b | CH.01 | Toyota→東大→TRIの転機を、スクロール量に直結した進行で体感させる | rAF + `position:sticky` | ★★☆ |
| 8 | **scrub② 研究3軸の展開** | 2b | CH.02 | 3軸（生成AI設計／Set-Based工学／構造最適化）が1画面から3画面へ分岐する過程を見せる | rAF + sticky | ★★☆ |
| 9 | kinetic 転機語の帯 | 2b | CH.01内 | 転機のキーワード（Toyota / UTokyo / TRI）を主役として強調 | CSS transform + 文節分割 | ★★☆ |
| 10 | 数値カウントアップ | 2a | 受賞数・粒子数 | 実績の量を体感させる | rAF、threshold 0.85 | ★★★ |
| 11 | stagger | 2a | 研究カード・受賞/論文リスト | 高密度リストを順に提示し圧迫感を減らす | `transition-delay` | ★★★ |
| 12 | スクロール進行バー | 2a | 全体 | 長尺化した現在地の把握 | rAF スロットル | ★★★ |
| 13 | `clip-path` リビール | 1 | Research/Hobbiesカード | カードが開く動きで情報単位を示す | CSS transition | ★★★ |
| 14 | `@property` 光量グラデ | 1 | セクション見出し | Hero粒子のテーマ（秩序⇄カオス）を静的セクションにも波及させる | `@property` + `@keyframes` | ★★☆ |
| 15 | backdrop-filter パネル | 1 | 章インデックス・進行バー | 本文の上に浮かせて現在地を常時提示（静止パネルのみ） | CSS | ★★★ |
| 16 | 章立てナラティブ＋インデックス | 1 | 全体 | 4章（Hero/Career/Research/Hobbies）の現在地 | HTML + JS | ★★★ |
| 17 | 3D transform（hover） | 2b | 論文/特許カード | hover時のみ軽い奥行き。情報の"モノ感"を強調 | `perspective` + `rotateY`（hoverのみ） | ★★☆ |

### Tier 2b の4条件チェック（`decision-rules.md`）

| # | 条件 | 判定 | 根拠 |
|---|---|---|---|
| 1 | 対象読者が「技術力・没入感の視覚的証明」を評価する相手か | **Yes** | `decisions.md` D2: 採用担当・ヘッドハンター |
| 2 | 主要流入がデスクトップか | **Yes** | 同上 |
| 3 | CV が「感じさせる」側か | **Yes** | 仕様書§7 |
| 4 | 実機検証の時間があるか | **未確保** | `decisions.md` D2。**実機確認完了まで公開不可**（`00_product-spec.md` §8） |

→ **条件1〜3クリア。条件4は着手時点で未確保のため、実装後・公開前に必ず満たす。**

---

## 3c. density 整合チェック

宣言 density = **高**（特殊 生成系）。下限は `motion-catalog.md` §2b:
`@keyframes`≥5 / `animation:`≥3 / IntersectionObserver≥2 / Tier2a採用1件以上 / 尺10×vp以上 / scrub 3箇所以上。

| 項目 | 下限 | 本案件の見込み | 判定 |
|---|---|---|---|
| `@keyframes` 定義数 | ≥5 | 見込み7（Hero粒子の内部/光量グラデ/進行バー/カウントアップ/stagger/kinetic帯/カードリビール） | ○ |
| `animation:` 使用数 | ≥3 | 見込み4 | ○ |
| IntersectionObserver | ≥2 | 各章フェード＋カウントアップのトリガーで2以上 | ○ |
| Tier 2a の採用件数 | 1件以上 | 3件（カウントアップ・進行バー・stagger） | ○ |
| **尺（ページ長 ÷ viewport）** | **10×vp以上** | 4章構成（Hero1 + Career2 + Research3 + Hobbies2 + Footer1 = 約9〜10×vp） | **要注意・実装時に9×vpを割らないよう調整** |
| **scrub 連動の箇所数** | **3箇所以上** | Career転機3箇所 + 研究3軸展開1箇所 = 4箇所 | ○ |

→ 尺のみ下限ぎりぎりのため、実装時にHero1章分の余白・Researchの全論文リスト表示を厚めに取り、9×vp未満に落ちないようにする。

---

## 4. duration / easing

| 用途 | duration | easing | 備考 |
|---|---|---|---|
| hover / focus 応答 | 150–300ms | ease-out | |
| スクロール連動の出現 | 600–1200ms | ease-out | |
| ページ内アンカー遷移 | 300–500ms | ease-in-out | 既存script.jsのスムーススクロールを流用 |
| scrub（Career/Research） | スクロール量に直結（durationなし） | linear（進行度直結） | rAFで毎フレーム再計算 |

---

## 5. 検討して不採用にしたもの

| 技法 | 不採用の理由 |
|---|---|
| GSAP + ScrollTrigger | IntersectionObserver + rAFで要件を満たせる。研究者個人サイトとして依存を増やさない姿勢を保つ（`folio_prisma`と同判断） |
| 慣性スクロール（Lenis） | キーボード/スクロールバー操作の割り込み報告（motion-catalog §2d）があり、個人サイトでのUX劣化リスクがメリットを上回る |
| ループ動画KV | 動画素材が存在しない。GPU生成（粒子Hero）で同じ役割を代替できる |
| マスコット・タイピングインジケータ | キャラクターレベル0の方針（`00_product-spec.md`§7）と衝突。会話系プロダクトでもない |
| カスタムカーソル | 得られる効果が雰囲気のみで、ネイティブカーソル除去はa11y上マイナス |

---

## 6. `prefers-reduced-motion: reduce` 時の置換表

| 通常時 | reduce 時 |
|---|---|
| 粒子がKohei Shintaniの字形に集合→カーソルで散る→クリックで衝撃波 | 文字に集合した静止1フレームを表示（`generative-hero`の既定動作） |
| Career/Research の scrub アニメ | scrub を無効化し、各転機・研究軸を通常の静的縦積みで表示 |
| 数値カウントアップ | 最終値を即時表示 |
| stagger（順次出現） | 同時にフェードインのみ（transformなし） |
| kinetic 転機語の帯 | 静止したテキストラベル表示 |
| `@property` 光量グラデ | 静止したグラデーション |
| 3D transform（カードhover） | `translateY`のみの通常hoverに置換 |

---

## 7. WCAG 2.2.2 チェック（Level A）

| 該当する動き | 5秒超か | 停止手段 |
|---|---|---|
| 粒子Heroのカオス⇄秩序ループ | Yes（背景として継続） | ユーザー操作（カーソル/クリック）で状態が変わるため「自動的に継続する情報提示」には当たらないが、念のためスクロールでカオス側に収束させ静止に近づける設計にする |
| kinetic 転機語の帯 | No（各語の表示は5秒以内、scrubでスクロール停止時は静止） | scrub連動のためスクロールが止まれば動きも止まる。停止ボタン不要（自動継続ではない） |
| スクロール進行バー | No | ユーザースクロールに完全追従。自動更新ではない |

→ 自動的に開始し5秒を超えて継続するのは粒子Hero背景のみだが、装飾であり情報提示ではないため、`prefers-reduced-motion`での静止1フレーム化を必須の緩和手段とする。

---

## 8. 日本語まわりの確認

- [x] 文字単位のアニメは使わない。kinetic帯は「Toyota」「東大」「TRI」等の**語単位**に限定
- [x] Webフォント読込を待つ演出はHero粒子のみ。`document.fonts.ready`を待ち、待機中はCSSグラデーション背景のみ表示（レイアウトシフトなし）
- [x] 禁則処理（`text-wrap: balance` / `word-break: keep-all`）を見出し・リードに適用

---

## 9. 実装への引き渡し

| 項目 | 記入 |
|---|---|
| 使う共通コンポーネント | `_shared/components/generative-hero/`（Hero）／`_shared/js/fade-in.js`（フェードのベース、scrub部分は個別実装） |
| 追加する外部依存 | **なし**（Vanilla JS + WebGL2のみ。仕様書§7の「依存を増やさない」姿勢と整合） |
| 品質ゲートで見るべき点 | モバイルでの粒子ティアダウングレード／reduced-motion時のスクショ／Career scrubがモバイルでも縦積みとして破綻しないか |

---

## 完了条件チェック

- [x] §3 網羅チェック表が全行埋まっている
- [x] Tier 0 が全部○
- [x] 採用エフェクト全件に「誰のどの認知を助けるか」を記載
- [x] Tier 2b の4条件チェックを記載（条件4は「実装後・公開前に満たす」として明記）
- [x] §3c density整合チェック（尺のみ要注意、対応方針を記載）
- [x] reduced-motion置換表が採用エフェクト全件をカバー
- [x] WCAG 2.2.2チェックを記載
- [x] 未検証技法（scroll-driven animationsのFirefox非対応等）に対応方針を明記
