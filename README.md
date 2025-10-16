# FrontLine: 気象前線の自動検出・精緻化・可視化パイプライン（Swin-UNet ベース）

本リポジトリは、気象数値モデル（GSM）から得られる多変量・時系列の格子データを入力として、Swin-UNet による前線推定（セマンティックセグメンテーション）、データ拡張を用いた精緻化、スケルトン化による線形化、評価・可視化・動画化・SVG 出力、さらに SHAP による説明可能性解析までを一貫して行う研究用パイプラインです。

- 入力: GSM 由来の各種気象変数（128×128 格子、t−6h / t0 / t+6h を連結）、および正解前線データ（5 クラスの 1/0 マスク）
- 出力:
  - Stage1: 前線（全 6 クラス: 0=背景, 1=温暖, 2=寒冷, 3=停滞, 4=閉塞, 5=接合）の画素ごとの確率/予測
  - Stage2: 精緻化（Swin もしくは 拡散モデル）。v3 は拡散モデルを「条件付き・ペア学習」で実装（詳細は下記）
  - Stage3: スケルトン化（線形化）したクラスマップ
  - Stage4: 前線ポリラインの SVG 出力
  - 可視化画像・比較動画・評価図表、SHAP 解析のサマリー

研究目的は、前線存在の有無と種別をピクセルレベルで高精度に推定し、その後の解析・可視化・予報支援に活用可能な「線」表現まで接続することです。

---

## コマンド

```bash
notify-run wsl-ubuntu -- nohup python main_v3.py > output_v3.log 2>&1 &

notify-run wsl-ubuntu -- nohup python main_v4.py > output_v4.log 2>&1 &

notify-run gpu02 -- nohup python main_v3.py > output_v3.log 2>&1 &
```

タスクの削除

```bash
pkill -f "main_v3.py"

pkill -f "main_v4.py"
```

権利を takumi ユーザーに指定

```bash
sudo chown -R takumi:takumi /home/takumi/docker_miniconda/src/FrontLine/
```

gpu02 → mac

```bash
rsync -avz --progress gpu02:/home/devel/work_takasuka_git/docker_miniconda/src/FrontLine/v3_result /Users/takumi0616/Develop/docker_miniconda/src/FrontLine/result_gpu02
```

wsl-ubuntu → mac

```bash
rsync -avz --progress wsl-ubuntu:/home/takumi/docker_miniconda/src/FrontLine/v31_result /Users/takumi0616/Develop/docker_miniconda/src/FrontLine/result_wsl-ubuntu
```

---

# バージョン別プログラムの詳細（v1 / v2 / v3）

このリポジトリには、目的は同じ（前線の検出→精緻化→線形化→可視化/評価/説明）でありながら、第二段階以降の戦略やコード構成が異なる3系統が共存します。以下では、各バージョンの特徴・手法・処理フロー・アルゴリズムを整理します。

参考ファイル:
- v1: `src/FrontLine/main_v1.py`, `src/FrontLine/swin_unet.py`
- v2: `src/FrontLine/main_v2.py`, `src/FrontLine/swin_unet.py`, `src/FrontLine/diffusion-model.py`, `src/FrontLine/denoising_diffusion_pytorch/*`
- v3: `src/FrontLine/main_v3.py`, `src/FrontLine/main_v3/*`, `src/FrontLine/diffusion-model.py`, `src/FrontLine/denoising_diffusion_pytorch/*`

---

## v1: Swin-UNet + 伝統的スケルトン化の3段構成

- 概要
  - Stage1（検出）: 93チャネル（31変数×3時刻: t−6h/t0/t+6h）の GSM データを入力し、Swin-UNet で6クラス（0=背景,1..5=前線）のセマンティックセグメンテーションを実施。
  - Stage2（精緻化, Swin）: 正解前線（GT）クラスマップに対して「歪な形に劣化」させた入力（n_augment=10）とGTのペアで復元学習（膨張/欠損/ランダム置換/偽前線追加）。
  - Stage3（線形化）: any-front（二値）→`skimage.morphology.skeletonize` で細線化→近傍多数決でクラス付与。
  - Stage4（SVG）: `skan.Skeleton` でポリライン抽出、地理座標の SVG を保存。
  - 可視化/評価/動画/SHAP: NetCDF（probabilities/class_map）保存、Cartopy 可視化、評価指標、動画生成、SHAP 解析。

---

## v2: 拡散モデル（DiffusionCorrector）による確率場の修正（unpaired）

- Stage2 を拡散モデルベースへ置換（unpaired）。
- 学習: GT 前線（5ch）→ one-hot 6ch 確率を作成し、「確率分布そのもの」を pred_v（v-parameterization）で学習。
- 推論: Stage1 確率を入力し、PSD 比（S1 vs GT）に基づく `t_start` 自動推定 → アンサンブル生成 → PMM（Probability Matched Mean）で秩統計を保った合成出力。
- 出力は v1 と同一（`refined_*.nc`）。v2 の設計は変えず、v3 の変更の影響を受けません（v2は本READMEの記述通り）。

---

## v3: モジュール分割 ＋ Stage3のDL-FRONT系アルゴリズム化 ＋ Stage2を「条件付き・ペア学習の拡散」に刷新

- コード構成
  - `main_v3.py` は薄いオーケストレータ。詳細は `main_v3/` 配下（config/datasets/models/stage1/stage2_diffusion/stage2_swin/stage3/visualize/evaluation/shap）。
  - `CFG` 一元管理（出力は `./v3_result/...`）。デフォルト `epochs` は短め（2）。

- Stage2（拡散, 条件付き・ペア学習）への刷新
  - 学習データ（ペア）:
    - 条件 cond: v1 Stage2 と同じ「歪な形に劣化」マップ（n_augment=10）を one-hot 6ch に変換（背景0, 前線1..5、各画素で合計1）。
    - 正解 x0: GT を one-hot 6ch に変換（背景0=1−any_front、各画素で合計1）。
  - 学習アルゴリズム（paired-conditional diffusion）:
    - x0 に前向きノイズを付加して x_t を作成し、UNet に [x_t, cond]（チャネル結合、計12ch）を入力。pred_v 損失（MSE）で学習。
    - 拡散は x0 側にのみ適用。cond にはノイズを加えず、ステップ毎に固定条件として結合。
  - 推論（条件付き DDIM/アンサンブル）:
    - 条件 cond に「Stage1 の確率（prob_*.nc）」を使用（6ch、各画素で合計1に正規化）。
    - `t_start_frac` により逆拡散の開始時刻を制御。cond を毎ステップ結合しつつ x_t→x_0 を再構成。アンサンブル平均＋（任意で）class_weights と Stage1 との blend（λ）を適用。
  - 実装:
    - `diffusion-model.py`: `ConditionalDiffusionCorrector` を追加（UNetの入力を x_t(6)+cond(6)=12ch としてラップ）。
    - `main_v3/main_v3_stage2_diffusion.py`: 学習データを `FrontalRefinementDataset(train/val)` に切り替え（v1 と同等の劣化生成を再利用）、paired 学習ループを実装。推論は `correct_from_probs_cond(...)` を使用。
    - 既存の unpaired 版API（`DiffusionCorrector` / `correct_from_probs(...)`）も後方互換として保持。
  - 期待効果:
    - 条件 cond が「Stage1 に起こりやすい歪み特性」を模擬しているため、実運用での S1 確率を条件にしたときに、拡散の逆過程が GT へ整合的に近づく方向に誘導されやすい。

- Stage3（DL-FRONT系線形化）
  - any-front（max(prob[...,1:5])）→ラプラシアン（リッジ検出）→二値化→`medial_axis` で骨格抽出→端点同士を最短路（コスト=1−any-front）で連結→パス上は近傍多数決でクラス付与。
  - 出力は v1/v2 と同じ `skeleton_*.nc`。

- 代替 Stage2（Swin）も同梱
  - `main_v3_stage2_swin.py` は v1 と同様の「劣化→復元」のSwin版（デフォルト未使用）。

---

# Diffusion Stage2（条件付き・ペア学習）詳説（v3）

- 何を学習し何を出力するか
  - 学習: 条件 c（劣化）に対する GT 確率 x0 の条件付き分布 p(x0 | c) を、拡散で学習（pred_v/MSE）。
  - 出力: 各ピクセルの6クラス確率（合計1）。NetCDF: `probabilities[lat,lon,class]` として `refined_*.nc` に保存。

- 学習入出力（正確）
  - 条件 cond: FrontalRefinementDataset が生成する劣化クラスマップ（n_augment=10）→ one-hot 6ch → [0,1]、sum=1。
  - 正解 x0: GT クラスマップ → one-hot 6ch → [0,1]、sum=1。
  - 形状: (B, C=6, H=128, W=128)。

- 損失/時系列
  - 時刻 t を一様にサンプルし、x0 にノイズを加え x_t を作成。UNet には [x_t, cond] をチャネル結合して入力。
  - pred_v（v-parameterization）の MSE を最小化。

- 推論入出力（正確）
  - 入力 cond: Stage1 の確率（6ch、sum=1）を使用（f32）。
  - 出力 rec: refined 確率（6ch、sum=1）。アンサンブル平均後、class_weights（背景0抑制）と Stage1 との blend（λ）→ 合計1に再正規化。

- ハイパラ（CFG["STAGE2"]["diffusion"]）
  - `timesteps=1000`, `sampling_timesteps=20`, `objective='pred_v'`, `beta_schedule='sigmoid'`, `base_dim`, `dim_mults` など。
  - `t_start_frac`: 逆拡散の開始時刻（0..1の比）。v2 の PSD 自動推定は v2 のみ（v3は固定制御）。
  - `class_weights`, `blend_lambda`: 背景支配の抑制、S1連続性の温存。

- v2 との違い
  - v2: unpaired 学習＋PSD自動 t_start＋PMM（秩統計保持）。
  - v3: paired-conditional 学習＋固定 t_start_frac＋平均＋（weights+blend）を基本（必要なら PMM 併用も可）。

---

# 処理フローの差分と戦略まとめ

- Stage1（共通）
  - 入力: 93ch（GSM 31変数×3時刻）、画像 128×128。
  - モデル: Swin-UNet。損失=CE+Dice。チェックポイント再開、最良モデル保存、SHAP対応。

- Stage2（精緻化）
  - v1（Swin）: 劣化（n_augment=10）→復元の教師あり。
  - v2（拡散, unpaired）: GT確率の分布を学習し、S1確率を PSD 自動 t_start ＋ アンサンブル＋PMMで修正。
  - v3（拡散, paired-conditional）: 「劣化（条件）＋GT（正解）」のペアで条件付き拡散を学習。推論では S1確率を条件に DDIM 逆拡散で refined を生成。平均＋weights+blend を標準採用。

- Stage3（線形化）
  - v1/v2: skeletonize（二値）一発。
  - v3: リッジ検出→骨格→最短路連結（コスト=1−any-front）で連続性・滑らかさを改善。

---

# 入出力・成果物の命名規則

- 入力（既定: `CFG["PATHS"]`）
  - GSM: `./128_128/nc_gsm9/gsmYYYYMM.nc`
  - 前線GT: `./128_128/nc_0p5_bulge_v2/YYYYMM.nc`
- 出力（バージョン別）
  - v1: `./v1_result/...`
  - v2: `./v2_result/...`
  - v3: `./v3_result/...`
- ファイル命名
  - Stage1 確率: `prob_YYYYMMDDHHMM.nc`（dims: lat, lon, class, time）
  - Stage2 確率: `refined_YYYYMMDDHHMM.nc`
  - Stage3 クラスマップ: `skeleton_YYYYMMDDHHMM.nc`（`class_map`）
  - 可視化: `visualizations/comparison_YYYYMMDDHHMM.png`
  - 動画: `comparison_YYYYMM.mp4`, `comparison_2023_full_year.mp4`（低解像度版も生成）
  - SVG: `stage4_svg/skeleton_YYYYMMDDHHMM.svg`

---

# 実行方法（例）

- v1 全段実行
  ```bash
  cd /home/takumi/docker_miniconda/src/FrontLine
  python main_v1.py
  ```
- v2 全段実行（Stage2=拡散, unpaired）
  ```bash
  cd /home/takumi/docker_miniconda/src/FrontLine
  python main_v2.py
  ```
- v3 全段実行（モジュール化＋DL-FRONT系 Stage3＋Stage2=拡散 paired-conditional）
  ```bash
  cd /home/takumi/docker_miniconda/src/FrontLine
  python main_v3.py
  ```

出力ルート（v1/v2/v3 でそれぞれ `./v*_result/`）に成果物が生成されます。チェックポイントが存在する場合は自動で最新から再開します。

---

# 評価・可視化・SHAP

- 評価: 2023年をデフォルトテスト区間として、全クラス/前線クラスのみの双方で Accuracy/Macro Precision/Recall/F1/κ を算出。混同行列やサマリー画像・ログを保存。
- 可視化: 海面更正気圧偏差（`surface_prmsl`）の等値線/塗り＋Stage1/2/3/GT を重畳。低気圧中心（`surface_low_center==1`）があれば赤×を表示。
- SHAP（Stage1）: 最良モデルに対して GradientExplainer によるクラス別 SHAP を計算し、CSV/PNG（beeswarm/bar/waterfall）を保存（GPUの空きに応じ自動調整・OOM時は再試行）。

---

# 今後の方針・改善案

- v3 Stage2（paired-conditional）の高度化
  - v2 での PSD 自動 t_start と PMM をオプション化して併用（適応制御＋秩統計保持）。
  - 確率校正（温度スケーリング/Dirichlet）で信頼度の整合性を向上。
  - class_weights / blend_lambda の自動最適化（ベイズ最適化等）。

- Stage3 の強化
  - 端点検出・分岐の堅牢化、小領域での補助点自動挿入、風向・温度傾度など物理量のコスト反映。

- 学習・運用
  - マルチタスク（前線＋低気圧中心）、混合精度/Flash-Attn/分散学習、物理整合的データ拡張。
  - 季節/地域別評価、OOD検知・不確かさ推定、Webビューア、CFGプロファイル（fast/accurate）。

---

以上の通り、v3 Stage2 を「歪な形に劣化させた10種類のデータ（n_augment=10）とGTのペア」に基づく条件付き拡散へ刷新しました。これにより Stage1 の歪みを模擬した条件に対して、GT へ近づく確率分布の再構成が可能になり、線形化（Stage3）前の位相整合・連続性・鋭さのバランス改善が期待できます。
