#huggingface-cli download --resume-download openai-community/gpt2 --local-dir /data1/xiaojin/physics/gpt2/
export CUDA_LAUNCH_BLOCKING=1 
CUDA_VISIBLE_DEVICES=6 python vfinetune.py --epochs 20 --batch_size 32 --gradient_accumulation 1 --learning_rate 5e-5 --rank 8 --label_num 2 --save_model_name gpt2_medium_nece_small.pth --task nece --model_path /physics/pretrain/models/medium
