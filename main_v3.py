"""
概要:
    v3 パイプライン（学習→SHAP→拡散補正→スケルトン化→可視化→評価→動画→SVG）の
    各ステージをモジュール化した実装を順番に呼び出す「オーケストレータ」スクリプト。
    本ファイルは実装の詳細を持たず、各ステージのランナー関数を呼ぶことで全処理を実行する。

構成:
    - main_v3.main_v3_config:
        設定辞書 CFG とユーティリティ（print_memory_usage, format_time）を提供
    - main_v3.main_v3_stage1:
        Stage1（Swin-UNet）学習・評価のランナー run_stage1()
    - main_v3.main_v3_shap:
        Stage1 モデルの SHAP 解析 run_stage1_shap_evaluation_cpu()
    - main_v3.main_v3_stage2_diffusion:
        Stage2（DiffusionCorrector による確率補正）の学習・推論ランナー run_stage2_diffusion()
    - main_v3.main_v3_stage3:
        Stage3（スケルトン化）ランナー run_stage3()
    - main_v3.main_v3_visualize:
        可視化ランナー run_visualization() および比較動画作成 create_comparison_videos()
    - main_v3.main_v3_evaluation:
        総合評価ランナー run_evaluation()
    - main_v3.main_v3_stage4_svg:
        Stage4（SVG 出力）ランナー run_stage4()

使い方:
    python main_v3.py を実行すると、main() が各ステージを順次実行し、
    成果物（モデル、NetCDF、PNG、SVG、ログ、動画）を v3 の出力ディレクトリ配下に保存する。
"""

import time

# Thin orchestrator that wires modularized v3 pipeline

from main_v3.main_v3_config import CFG, print_memory_usage, format_time
from main_v3.main_v3_stage1 import run_stage1
from main_v3.main_v3_shap import run_stage1_shap_evaluation_cpu
from main_v3.main_v3_stage2_diffusion import run_stage2_diffusion
from main_v3.main_v3_stage3 import run_stage3
from main_v3.main_v3_visualize import run_visualization, create_comparison_videos
from main_v3.main_v3_evaluation import run_evaluation
from main_v3.main_v3_stage4_svg import run_stage4


def main():
    """
    概要:
        v3 パイプライン全体（Stage1→SHAP→Stage2(拡散)→Stage3→可視化→評価→動画→Stage4）を順次実行する
        エントリポイント。各ステージの実行時間を計測し、メモリ使用量も適宜出力する。

    入力:
        なし（設定は main_v3_config.CFG を参照）

    処理:
        1) Stage1 学習・評価（Swin-UNet）
        2) Stage1 モデルの SHAP 解析
        3) Stage2 拡散モデル（DiffusionCorrector）による確率補正（学習・推論）
        4) Stage3 スケルトン化
        5) Stage1/2/3 の結果を地図上に可視化（PNG出力）
        6) 2023年共通時刻で総合評価（混同行列、指標の図表・ログ）
        7) 比較画像から動画作成（通常/低解像度）
        8) Stage4 SVG 出力（骨格ポリライン）

    出力:
        なし（副作用として各ステージの成果物を v3 の出力ディレクトリ配下に保存）
    """
    total_start_time = time.time()
    print_memory_usage("Start main")

    # Stage 1
    stage1_start = time.time()
    run_stage1()
    stage1_end = time.time()
    print(f"Stage1 実行時間: {format_time(stage1_end - stage1_start)}")

    # Stage 1 SHAP
    shap_start = time.time()
    run_stage1_shap_evaluation_cpu(
        use_gpu=CFG["SHAP"]["use_gpu"],
        max_samples_per_class=CFG["SHAP"]["max_samples_per_class"],
        out_root=CFG["SHAP"]["out_root"],
    )
    shap_end = time.time()
    print(f"Stage1 SHAP分析 実行時間: {format_time(shap_end - shap_start)}")

    # Stage 2 (DiffusionCorrector)
    stage2_start = time.time()
    run_stage2_diffusion()
    stage2_end = time.time()
    print(f"Stage2 実行時間: {format_time(stage2_end - stage2_start)}")

    # Stage 3
    stage3_start = time.time()
    run_stage3()
    stage3_end = time.time()
    print(f"Stage3 実行時間: {format_time(stage3_end - stage3_start)}")

    # Visualization
    vis_start = time.time()
    run_visualization()
    vis_end = time.time()
    print(f"可視化処理 実行時間: {format_time(vis_end - vis_start)}")

    # Evaluation
    eval_start = time.time()
    run_evaluation()
    eval_end = time.time()
    print(f"評価処理 実行時間: {format_time(eval_end - eval_start)}")

    # Video
    video_start = time.time()
    create_comparison_videos(
        image_folder=CFG["VIDEO"]["image_folder"],
        output_folder=CFG["VIDEO"]["output_folder"],
        frame_rate=CFG["VIDEO"]["frame_rate"],
        low_res_scale=CFG["VIDEO"]["low_res_scale"],
        low_res_frame_rate=CFG["VIDEO"]["low_res_frame_rate"],
    )
    video_end = time.time()
    print(f"動画作成 実行時間: {format_time(video_end - video_start)}")

    # Stage 4 (SVG export)
    stage4_start = time.time()
    run_stage4()
    stage4_end = time.time()
    print(f"Stage4 実行時間: {format_time(stage4_end - stage4_start)}")

    print_memory_usage("End main")
    print("All Stages Done.")
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"プログラム全体の実行時間: {format_time(total_elapsed_time)}")


if __name__ == "__main__":
    main()
