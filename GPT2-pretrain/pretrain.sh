NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 torchrun --nproc_per_node=8 --master_port=29511 pretrain_dpp.py \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --dataset_path /cpfs01/user/xiaojin/xiaojin/physics/dataset/1128op21.json \
    --model_save_path /cpfs01/user/xiaojin/xiaojin/physics/pretrain/models/1128_hard_no_rope \
    --accumulation_steps 2
