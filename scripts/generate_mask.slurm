#!/bin/bash
#SBATCH -p a100_short,radiology
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200GB
#SBATCH --time=1-12:00:00
#SBATCH --job-name=ap-train
#SBATCH --output=/gpfs/data/shenlab/wz1492/HandTrack-SAM2/logs/%J.log
#SBATCH --exclude=a100-4020

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate sam2


python src/main.py \
    --input_video_path "./data/test.mp4" \
    --output_video_path "./output/hands_mask_only.mp4" \
    --sam2_checkpoint "./sam2/checkpoints/sam2.1_hiera_large.pt" \
    --sam2_config "./sam2/configs/sam2.1/sam2.1_hiera_l.yaml" \
    --tmp_dir "./tmp_frames_mask" \
    --mediapipe_model_path "./models/hand_landmarker.task" \
    --prompt_mode "box" \
    --additional_runs \
        '{"output_video_path":"./output/hands_overlay.mp4","overlay_original":true}' \
        '{"output_video_path":"./output/hands_bbox_only.mp4","overlay_mode":"bbox"}'


