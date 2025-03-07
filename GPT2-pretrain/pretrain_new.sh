NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 torchrun --nproc_per_node=8 --master_port=29510 pretrain_new.py \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --dataset_path /cpfs01/user/xiaojin/xiaojin/physics/dataset/hard_more_even_1127.json \
    --model_save_path /cpfs01/user/xiaojin/xiaojin/physics/pretrain/models/1127_hard \
    --accumulation_steps 2
