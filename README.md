# FrontLine: 気象前線の自動検出・精緻化・可視化パイプライン（Swin-UNetベース）

本リポジトリは、気象数値モデル（GSM）から得られる多変量・時系列の格子データを入力として、Swin-UNet による前線推定（セマンティックセグメンテーション）、データ拡張を用いた精緻化、スケルトン化による線形化、評価・可視化・動画化・SVG出力、さらに SHAP による説明可能性解析までを一貫して行う研究用パイプラインです。

- 入力: GSM 由来の各種気象変数（128×128 格子、t−6h / t0 / t+6h を連結）、および正解前線データ（5 クラスの 1/0 マスク）
- 出力:
  - Stage1: 前線（全6クラス: 0=背景, 1=温暖, 2=寒冷, 3=停滞, 4=閉塞, 5=接合）の画素ごとの確率/予測
  - Stage2: データ拡張を模した擬似劣化入力からの復元学習による精緻化（確率/予測）
  - Stage3: スケルトン化（線形化）したクラスマップ
  - Stage4: 前線ポリラインの SVG 出力
  - 可視化画像・比較動画・評価図表、SHAP 解析のサマリー

研究目的は、前線存在の有無と種別をピクセルレベルで高精度に推定し、その後の解析・可視化・予報支援に活用可能な「線」表現まで接続することです。

---

## リポジトリ構成

- `main/main.py`（推奨, v31 系）
  - 全ステージ（学習・評価・可視化・動画作成・SVG・SHAP）を一気通貫で実行するメインスクリプト
  - Swin-UNet 実装は `main/swin_unet.py` をインポート
  - 統一設定 `CFG` でパス・ハイパラ・可視化・動画・SHAP を集中管理（出力は `./v31_result/...`）
- `main/swin_unet.py`
  - Swin-UNet 本体の実装（エンコーダ/デコーダ/パッチ操作/注意機構など）
- `main_v31.py`（単体実行版の改良スクリプト）
  - `main/main.py` と同等の流れを 1 ファイルに集約したバリアント
  - ただし出力ディレクトリが `v30_result` のままになっている箇所があるため、利用時は注意（下記参照）
- `main_v30.py`（旧版）
  - v30 系の単体実行スクリプト。ピクセル一致率の時系列解析など追加可視化あり
  - 出力は `./v30_result/...`
- `data.py`
  - `gsm7 → gsm8` 変換。各変数を規則に基づき正規化＋int16 量子化（_FillValue=-32768）、blosc-LZ4 圧縮、チャンク `(1,128,128)`
- `data1.py`
  - `gsm7 → gsm9` 変換。正規化・量子化なし（dtype 踏襲）、blosc-LZ4 圧縮、チャンク `(1,128,128)`
- `data2.py`
  - `gsm7 → gsm10` 変換。規則に基づき正規化のみ（float32 のまま保存）、blosc-LZ4 圧縮、チャンク `(1,128,128)`
- `main/README.md`
  - 旧来の運用コマンド断片（プロセス起動/停止/所有者変更）

---

## パイプライン概要と手法

- 入力特徴（Stage1）
  - GSM の多変量 3 時刻（t−6h, t0, t+6h）をチャネル方向に結合（合計 93ch）
  - 画像サイズは 128×128（`CFG["IMAGE"]`）
- 目標クラス（全 6 クラス）
  - 0=背景、1=温暖、2=寒冷、3=停滞、4=閉塞、5=接合（warm_cold）
- モデル
  - Swin-UNet（Swin Transformer をバックボーンとした U-Net 架構）
  - 損失: CrossEntropyLoss + DiceLoss の複合
- 3 段階の処理
  - Stage1: 画素単位の多クラス分類（確率を NetCDF で保存）
  - Stage2: 正解前線マスクに対し膨張/欠損/擬似前線付与など多数の劣化を加えた入力から復元学習（ノイズに頑健、精緻化）
  - Stage3: 1～5 クラスを対象にスケルトン化（thin、線形の中心線抽出に近い）
  - Stage4: スケルトンを `skan` でポリライン化し、地理座標で SVG 出力
- 可視化・評価
  - GSM の海面更正気圧偏差（`surface_prmsl`）の等値線/塗りつぶしに重畳して、Stage1/2/3/GT を並列表示
  - 混同行列・各種指標（Accuracy, Macro Precision/Recall/F1, Cohen’s κ）
  - 時刻単位の一致率/ピクセル一致の解析（v30）
  - 2023 年をテスト区間（`CFG` で指定）

---

## 入出力データ仕様

- 入力（既定のパスは `CFG["PATHS"]` / v31）
  - GSM データ: `./128_128/nc_gsm9/gsmYYYYMM.nc`
    - 変数例: `surface_prmsl`（可視化用）、`surface_low_center`（低気圧中心、ある場合）
  - 正解前線: `./128_128/nc_0p5_bulge_v2/YYYYMM.nc`
    - 変数: `warm`, `cold`, `stationary`, `occluded`, `warm_cold`（各 1/0 マスク）
- 出力（v31）
  - Stage1 確率: `./v31_result/stage1_nc/prob_YYYYMMDDHHMM.nc`（dims: lat, lon, class=6, time）
  - Stage2 精緻化: `./v31_result/stage2_nc/refined_YYYYMMDDHHMM.nc`
  - Stage3 スケルトン: `./v31_result/stage3_nc/skeleton_YYYYMMDDHHMM.nc`（変数 `class_map`）
  - 画像: `./v31_result/visualizations/comparison_YYYYMMDDHHMM.png`
  - 動画: `./v31_result/comparison_YYYYMM.mp4`, `comparison_2023_full_year.mp4`, 低解像度版
  - SVG: `./v31_result/stage4_svg/skeleton_YYYYMMDDHHMM.svg`
  - SHAP: `./v31_result/shap_stage1/` に CSV/PNG サマリー

※ v30 系は `./v30_result/...` に出力します。

---

## セットアップ

推奨: Linux/Mac + CUDA GPU（任意）

- Python ライブラリ（主要）
  - torch, torchvision
  - timm, einops
  - numpy, pandas
  - xarray, netCDF4
  - scikit-learn, seaborn, matplotlib, japanize-matplotlib
  - scikit-image（skimage）, skan
  - cartopy
  - opencv-python
  - psutil, tqdm
  - shap
- 外部コマンド
  - ffmpeg（動画作成に使用）

例（conda/mamba）:
```bash
mamba install pytorch torchvision -c pytorch
mamba install -c conda-forge xarray netcdf4 cartopy shap skan ffmpeg
pip install timm einops opencv-python seaborn japanize-matplotlib psutil tqdm scikit-image
```

---

## 使い方（推奨フロー: v31 = `main/main.py`）

全ステージを通しで実行:
```bash
python main/main.py
```

- `main/main.py` は以下を順に実行します
  1) Stage1 学習 → 最良モデル保存 → 評価（確率を NetCDF 保存）
  2) Stage1 の最良モデルで SHAP 解析（GPU 空きメモリに応じ自動選択）
  3) Stage2 学習・評価（Stage1 出力から）
  4) Stage3 スケルトン化
  5) 可視化（比較パネル画像大量）
  6) 評価図表の作成
  7) 比較動画の作成（年月/年間）
  8) Stage4 SVG 生成

バックグラウンド起動（例）:
```bash
nohup python main/main.py > out_v31.log 2>&1 &
```

プロセス停止（例）:
```bash
pkill -f "main.py"     # 旧コマンド例（参考）
pkill -f "main/main.py"
```

所有者変更（例）:
```bash
sudo chown -R takumi:takumi /home/takumi/docker_miniconda/src/FrontLine/
```

---

## ステージ別に実行（関数）

`main/main.py` 内の関数を使えば段階実行も可能です（インタラクティブに利用する場合など）:
- `run_stage1()`, `run_stage1_shap_evaluation_cpu(...)`, `run_stage2()`, `run_stage3()`, `run_visualization()`, `run_evaluation()`, `create_comparison_videos(...)`, `run_stage4()`

Python から例:
```python
from main.main import run_stage1, run_stage2, run_stage3, run_visualization, run_evaluation, run_stage4
run_stage1()
run_stage2()
run_stage3()
run_visualization()
run_evaluation()
run_stage4()
```

---

## 旧/代替スクリプト（v30/v31 一体版）

- `python main_v31.py`
  - v31 の改良内容を 1 ファイルにまとめた版
  - ただし、フォルダ名が一部 `v30_result` のままになっているため、結果保存先に注意してください
- `python main_v30.py`
  - 旧版の完全一体実装。ピクセル一致率の時系列解析や図表なども含まれます
  - 出力は `./v30_result/...`

必要に応じて、ソース先頭付近の出力パス/エポック数などを手動で調整してください。

---

## データ変換ユーティリティ

GSM 原データ（`./128_128/nc_gsm7/*.nc`）を異なる形式へ変換します。保存は NetCDF（NETCDF4）+ blosc-LZ4 圧縮、チャンク `(1,128,128)`。

- `data.py`: `gsm7 → gsm8`
  - 変数名に応じて `offset/scale/sqrt` で正規化 → int16 量子化（_FillValue = -32768）
  - 実行:
    ```bash
    python data.py
    ```
  - 出力: `./128_128/nc_gsm8/*.nc`
- `data1.py`: `gsm7 → gsm9`
  - 正規化/量子化なし（元 dtype を踏襲）
  - 実行:
    ```bash
    python data1.py
    ```
  - 出力: `./128_128/nc_gsm9/*.nc`
- `data2.py`: `gsm7 → gsm10`
  - 正規化のみ（float32 のまま保存、量子化なし）
  - 実行:
    ```bash
    python data2.py
    ```
  - 出力: `./128_128/nc_gsm10/*.nc`

blosc-LZ4 フィルタが利用可能か起動時に検査します（利用不可の場合 RuntimeError）。

---

## 可視化・動画・SVG

- 比較パネル画像
  - 背景: 海面更正気圧の偏差（`surface_prmsl` の場から領域平均を引いた値）を等値線/塗りで表現
  - Stage1/2/3/GT クラスマップを半透明で重畳
  - 低気圧中心（`surface_low_center==1`）があれば赤 × を重畳
- 動画
  - `comparison_YYYYMM.mp4` と `comparison_2023_full_year.mp4`（低解像度版も生成）
  - 生成には ffmpeg が必要です
- SVG（Stage4）
  - スケルトン化結果を `skan` で経路抽出し、地理座標のポリラインで出力
  - クラスごとに色分け（温暖=赤, 寒冷=青, 停滞=緑, 閉塞=紫, 接合=橙）

---

## 評価・指標

- 全クラス（0〜5）: 混同行列/Accuracy/Macro Precision/Macro Recall/Macro F1/Cohen’s κ
- 前線のみ（1〜5）: 背景 0 を除外した評価も実行
- v30 には、時刻ごとピクセル一致率の詳細 CSV/プロット出力（`./v30_result/pixel_analysis/`）も含まれます

---

## SHAP による説明可能性解析（Stage1）

- 最良モデル（`model_final.pth`）を読み込み、各クラス（1〜5）に対し GradientExplainer で SHAP を算出
- 特徴（93ch）の平均 |SHAP| 等を CSV/PNG（beeswarm/bar/waterfall）で保存
- GPU の空きメモリ状況に応じて GPU/CPU を自動選択、OOM 時は `nsamples` を段階的に縮小して再試行

---

## 乱数・再現性・チェックポイント

- 乱数シード（`CFG["SEED"]`）を設定
- PyTorch/CuDNN の決定論オプションを設定（学習速度は低下し得る）
- 学習時に `checkpoint_epoch_*.pth` を保存し、次回実行時は最新のチェックポイントから自動再開
- 最良エポックの `model_final.pth` も併せて保存

---

## 並列・メモリ最適化

- Dataset は「オンデマンド読み込み」＋「ファイルハンドル/サンプルキャッシュ」を実装
- xarray で必要時刻のみ読み出し、GC/psutil によるメモリ監視ログ出力あり
- 可視化は初回 1 枚をシリアルでウォームアップ後、Multiprocessing で並列処理

---

## よくあるトラブルとヒント

- ffmpeg が無い / PATH が通っていない → 動画作成が失敗します。`ffmpeg -version` で確認してください
- `surface_prmsl` が GSM に無い → 可視化で気圧場の重畳がスキップされます
- `surface_low_center` が無い → 低気圧中心の × は描画されません
- v31 と v30 の出力先が混在 → 使用スクリプトに合わせて `CFG`（もしくはスクリプト先頭のパス定数）を整合させてください
- GPU メモリ不足（特に SHAP） → 自動でサンプル数を減らし再試行。必要に応じバッチ/画像サイズ/モデル幅を調整

---

## ライセンス/著作

- 研究用途を想定。ライセンス未定義の場合は私的利用・検証目的での使用を想定してください
- 著者: takumi

---

## 参考（旧コマンド抜粋）

`main/README.md` に記載の一部コマンド（環境に応じて読み替えてください）

```bash
# バックグラウンド起動（例）
notify-run gpu02 -- nohup python main.py > output.log 2>&1 &

# タスクの停止（例）
pkill -f "main.py"

# 所有者変更（例）
sudo chown -R takumi:takumi /home/takumi/docker_miniconda/src/FrontLine/
```

---

## 付録: 主要パラメータ（`main/main.py` 内 CFG）

- パス: `./128_128/nc_gsm9`, `./128_128/nc_0p5_bulge_v2`, 出力は `./v31_result/...`
- Stage1
  - `in_chans=93`, `num_classes=6`, `epochs=50`
  - モデル幅: `embed_dim=192`, `window_size=16`, `ape=True` など
  - 損失: CE + Dice
- Stage2
  - 入力 1ch（クラスマップ） → 6 クラス
  - 多様な擬似劣化（膨張、ギャップ、ピクセル入替、偽前線）
  - `embed_dim=96`, `ape=False` など
- 可視化
  - クラス色: 0=白, 1=赤, 2=青, 3=緑, 4=紫, 5=橙
  - 海面更正気圧偏差: [-40, 40] hPa, 21 レベル

必要に応じて `CFG` を編集し、入出力ディレクトリや学習条件・可視化条件を切り替えてください。
