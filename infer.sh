CUDA_VISIBLE_DEVICES=1 python vibe_check_video_qa.py \
    --model-id weights/qwen3_vl_8b_inst \
    --video-path raw_data/0026_09uVWWLKNdc.mp4 \
    --question "In the first experiment, did the yellow ball go to door number 1?"