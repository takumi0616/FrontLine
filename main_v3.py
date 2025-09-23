import time

# Thin orchestrator that wires modularized v3 pipeline

from main_v3.main_v3_config import CFG, print_memory_usage, format_time
from main_v3.main_v3_ddp import init_distributed, is_main_process, barrier, cleanup
from main_v3.main_v3_stage1 import run_stage1
from main_v3.main_v3_shap import run_stage1_shap_evaluation_cpu
from main_v3.main_v3_stage2_diffusion import run_stage2_diffusion
from main_v3.main_v3_stage3 import run_stage3
from main_v3.main_v3_visualize import run_visualization, create_comparison_videos
from main_v3.main_v3_evaluation import run_evaluation
from main_v3.main_v3_stage4_svg import run_stage4


def main():
    initialized = init_distributed()
    total_start_time = time.time()
    if is_main_process():
        print_memory_usage("Start main")

    # Stage 1
    stage1_start = time.time()
    run_stage1()
    barrier()
    stage1_end = time.time()
    if is_main_process():
        print(f"Stage1 実行時間: {format_time(stage1_end - stage1_start)}")

    # Stage 1 SHAP
    shap_start = time.time()
    if is_main_process():
        run_stage1_shap_evaluation_cpu(
            use_gpu=CFG["SHAP"]["use_gpu"],
            max_samples_per_class=CFG["SHAP"]["max_samples_per_class"],
            out_root=CFG["SHAP"]["out_root"],
        )
    barrier()
    shap_end = time.time()
    if is_main_process():
        print(f"Stage1 SHAP分析 実行時間: {format_time(shap_end - shap_start)}")

    # Stage 2 (DiffusionCorrector)
    stage2_start = time.time()
    run_stage2_diffusion()
    barrier()
    stage2_end = time.time()
    if is_main_process():
        print(f"Stage2 実行時間: {format_time(stage2_end - stage2_start)}")

    # Stage 3
    stage3_start = time.time()
    if is_main_process():
        run_stage3()
    barrier()
    stage3_end = time.time()
    if is_main_process():
        print(f"Stage3 実行時間: {format_time(stage3_end - stage3_start)}")

    # Visualization
    vis_start = time.time()
    if is_main_process():
        run_visualization()
    barrier()
    vis_end = time.time()
    if is_main_process():
        print(f"可視化処理 実行時間: {format_time(vis_end - vis_start)}")

    # Evaluation
    eval_start = time.time()
    if is_main_process():
        run_evaluation()
    barrier()
    eval_end = time.time()
    if is_main_process():
        print(f"評価処理 実行時間: {format_time(eval_end - eval_start)}")

    # Video
    video_start = time.time()
    if is_main_process():
        create_comparison_videos(
            image_folder=CFG["VIDEO"]["image_folder"],
            output_folder=CFG["VIDEO"]["output_folder"],
            frame_rate=CFG["VIDEO"]["frame_rate"],
            low_res_scale=CFG["VIDEO"]["low_res_scale"],
            low_res_frame_rate=CFG["VIDEO"]["low_res_frame_rate"],
        )
    barrier()
    video_end = time.time()
    if is_main_process():
        print(f"動画作成 実行時間: {format_time(video_end - video_start)}")

    # Stage 4 (SVG export)
    stage4_start = time.time()
    if is_main_process():
        run_stage4()
    barrier()
    stage4_end = time.time()
    if is_main_process():
        print(f"Stage4 実行時間: {format_time(stage4_end - stage4_start)}")

    if is_main_process():
        print_memory_usage("End main")
        print("All Stages Done.")
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    if is_main_process():
        print(f"プログラム全体の実行時間: {format_time(total_elapsed_time)}")
    cleanup()


if __name__ == "__main__":
    main()
