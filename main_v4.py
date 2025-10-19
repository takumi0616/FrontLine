"""
ファイル概要（main_v4.py）:
- 役割:
  v4 パイプラインのオーケストレーター（ステージ 1→1.5→2→2.5→3→3.5→4→4.5 を順に実行）。
  各ステージの学習/推論およびステージ終了ごとの可視化を直列に実行・計測する。
- 入出力:
  入力: 直接的な関数引数は無し（各ステージ内で CFG と NetCDF 入出力を行う）。
  出力: 各ステージの成果物（NetCDF）と、その可視化 PNG、標準出力ログ。
- 全体フロー（要点のみ。詳細は各ステージのモジュールを参照）:
  1) Stage1   : 6クラス (0:none,1:warm,2:cold,3:stationary,4:occluded,5:junction) を Swin-UNet で学習・推論
                出力は probabilities(H,W,C=6) と 0/5 二値 class_map（class_map_0_5）→ prob_*.nc
  2) Stage1.5 : junction の論理整形（小領域除去・過大領域の中心2x2縮退）→ junction_*.nc（junction:0/1）
  3) Stage2   : warm/cold（3クラス: none/warm/cold、学習=GSM+GT junc、推論=GSM+Stage1.5 junc）
                出力は probabilities と「junction=5 固定＋1/2 重畳」の 4値 class_map（class_map_combined: 5/1/2/0）
  4) Stage2.5 : 3段階の厳格化
                (a) front（1/2）を“元の junction”に連結するものだけ一次フィルタ
                (b) junction を「warm/cold 両側接触のみ」に厳格化
                (c) 厳格化済み junction に“連結する” front のみ再フィルタ
                → refined_*.nc（class_map:0/1/2, junction:0/1）
                以降（Stage3/4 系）はこの厳格化 junction を一貫して使用
  5) Stage3   : occluded（二値、学習=GSM+GT junc+GT warm+GT cold、推論=GSM+Stage2.5 junc+warm+cold）
                出力は probabilities と 5値 class_map（class_map_combined: 5/1/2/4/0）
  6) Stage3.5 : occluded は warm/cold/junction のいずれかに付着するもののみ残す → occluded_*.nc（class_map:0/1）
  7) Stage4   : stationary（二値、学習=GSM+GT junc+GT warm+GT cold+GT occ、推論=GSM+Stage2.5 junc+warm+cold + Stage3.5 occ）
                出力は probabilities と 6値 class_map（class_map_combined: 5/1/2/4/3/0）
  8) Stage4.5 : 小停滞除去（面積≤25） + 「寒冷に付着した停滞」を寒冷へ再分類
                最終合成の優先順位は 5 > 4 > 3 > 2 > 1 > 0（上書き禁止の順序合成）
                出力: stationary_*.nc（class_map:0/1）, final_*.nc（class_map:0..5）
                final は final_out_dir と stage4_5_out_dir の双方に保存
- 実装上の注意:
  - 計測用のメモリ・時間ログを出力する。
  - 可視化は各ステージ終了直後に run_visualization_for_stage で実行。
  - 各ステージの入出力仕様・前段成果物の使用先は main_v4_config.CFG と各 stage_* モジュールに準拠。
"""

import time

from main_v4.main_v4_config import print_memory_usage, format_time
from main_v4.main_v4_stage1 import run_stage1
from main_v4.main_v4_stage1_5 import run_stage1_5
from main_v4.main_v4_stage2 import run_stage2
from main_v4.main_v4_stage2_5 import run_stage2_5
from main_v4.main_v4_stage3 import run_stage3
from main_v4.main_v4_stage3_5 import run_stage3_5
from main_v4.main_v4_stage4 import run_stage4
from main_v4.main_v4_stage4_5 import run_stage4_5
from main_v4.main_v4_visualize import run_visualization_for_stage


def main():
    """
    関数概要:
      v4 パイプラインの全ステージ（1→1.5→2→2.5→3→3.5→4→4.5）を順順に実行し、各ステージの処理時間と
      メモリ使用量をログ出力する。各ステージ完了後に可視化も行う。

    入力:
      なし（関数引数は無し）。各ステージは内部で CFG（環境設定）と入出力パスを参照する。

    処理:
      - Stage1: 6クラス（0..5）を学習/推論し、probabilities と class_map_0_5（0/5）を保存
      - Stage1.5: junction の論理整形（小領域除去・過大領域の中心2x2化）
      - Stage2: warm/cold（3クラス）を学習/推論し、probabilities と「junction=5 固定＋1/2」の class_map_combined を保存
      - Stage2.5: front の junction 連結で一次フィルタ → junction を両側接触のみへ厳格化 → 厳格化 junction への再連結 front のみを残す
                  → 以降はこの厳格化 junction を一貫して使用
      - Stage3: occluded（二値）を学習/推論し、probabilities と「5/1/2/4/0」の class_map_combined を保存
      - Stage3.5: occluded の付着制約（warm/cold/junction のいずれかに接している画素のみ残す）
      - Stage4: stationary（二値）を学習/推論し、probabilities と「5/1/2/4/3/0」の class_map_combined を保存
      - Stage4.5: 小停滞除去（面積≤25） + 停滞→寒冷再分類 + 最終合成（優先順位: 5>4>3>2>1>0）
      各ステージ後に run_visualization_for_stage により PNG 可視化を出力。

    出力:
      - 標準出力: ステージごとの経過時間, メモリ使用量, 進捗ログ
      - ファイル: 各ステージの NetCDF 出力（prob_*, junction_*, refined_*, occluded_*, stationary_*, final_*）、
                  各ステージの可視化 PNG（output_viz ディレクトリ）
    """
    total_start = time.time()
    print_memory_usage("Start main_v4")

    # Stage 1
    t = time.time()
    run_stage1()
    print(f"[main_v4] Stage1: {format_time(time.time() - t)}")
    run_visualization_for_stage("stage1")

    # Stage 1.5
    t = time.time()
    run_stage1_5()
    print(f"[main_v4] Stage1.5: {format_time(time.time() - t)}")
    run_visualization_for_stage("stage1_5")

    # Stage 2
    t = time.time()
    run_stage2()
    print(f"[main_v4] Stage2: {format_time(time.time() - t)}")
    run_visualization_for_stage("stage2")

    # Stage 2.5
    t = time.time()
    run_stage2_5()
    print(f"[main_v4] Stage2.5: {format_time(time.time() - t)}")
    run_visualization_for_stage("stage2_5")

    # Stage 3
    t = time.time()
    run_stage3()
    print(f"[main_v4] Stage3: {format_time(time.time() - t)}")
    run_visualization_for_stage("stage3")

    # Stage 3.5
    t = time.time()
    run_stage3_5()
    print(f"[main_v4] Stage3.5: {format_time(time.time() - t)}")
    run_visualization_for_stage("stage3_5")

    # Stage 4
    t = time.time()
    run_stage4()
    print(f"[main_v4] Stage4: {format_time(time.time() - t)}")
    run_visualization_for_stage("stage4")

    # Stage 4.5
    t = time.time()
    run_stage4_5()
    print(f"[main_v4] Stage4.5: {format_time(time.time() - t)}")
    run_visualization_for_stage("stage4_5")

    print_memory_usage("End main_v4")
    print(f"[main_v4] Total: {format_time(time.time() - total_start)}")


if __name__ == "__main__":
    main()
