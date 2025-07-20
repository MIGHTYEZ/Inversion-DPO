export VAE="madebyollin/sdxl-vae-fp16-fix"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="yuvalkirstain/pickapic_v2"


accelerate launch --mixed_precision "bf16" train_ddim.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataloader_num_workers=0 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=1000 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/root/Datasets" \
  --cache_dir_model='/root/diffusers' \
  --checkpointing_steps=200 \
  --beta_dpo=2000 \
  --variant="fp16" \
  --resolution=1024 \
  --sdxl \
  --gradient_checkpointing \
  --num_inference_steps=80 \
  --output_dir="tmp-sdxl"
