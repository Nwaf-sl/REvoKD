CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift deploy \
    --model checkpoints \
    --infer_backend vllm \
    --torch_dtype bfloat16 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 40000 \
    --max_new_tokens 4096 \
    --vllm_limit_mm_per_prompt '{"image": 20, "video": 2}' \
    --served_model_name Qwen3-VL-4B-my