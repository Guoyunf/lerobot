#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}" lerobot-train \
  --dataset.repo_id=local/rotating_knob_split_rgb_tactile_force_q01q99 \
  --dataset.root=/home/yunfei/project/lerobot_v050/datasets/rotating_knob_split_rgb_tactile_lerobot_v30 \
  --dataset.use_imagenet_stats=false \
  --dataset.video_backend=torchcodec \
  --policy.type=smolvla \
  --policy.pretrained_path=/home/yunfei/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205_migrated \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.push_to_hub=false \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}' \
  --policy.max_state_dim=32 \
  --policy.max_action_dim=32 \
  --policy.resize_imgs_with_padding='[256, 256]' \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=2e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=200000 \
  --policy.scheduler_decay_lr=1e-6 \
  --policy.vlm_model_name=/home/yunfei/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --policy.load_vlm_weights=true \
  --policy.prefix_length=0 \
  --policy.pad_language_to=max_length \
  --policy.num_expert_layers=0 \
  --policy.tactile_feature_keys='["observation.tactile.force"]' \
  --output_dir=outputs/train/20260626_rotating_knob_split_rgb_tactile_force_q01q99_smolvla_1gpu_bs32_pred16_exec8_256 \
  --job_name=20260626_rotating_knob_split_rgb_tactile_force_q01q99_smolvla_1gpu_bs32_pred16_exec8_256 \
  --seed=1000 \
  --num_workers=16 \
  --batch_size=32 \
  --steps=200000 \
  --eval_freq=-1 \
  --log_freq=20 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --use_policy_training_preset=true \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --wandb.run_id=20260626_rotating_knob_split_rgb_tactile_force_q01q99_smolvla_1gpu_bs32_pred16_exec8_256
