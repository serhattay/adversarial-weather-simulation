#!/bin/bash -l

# ==============================================================================
# SGE Job Script — DETR Fine-tuning on KITTI
# Submit with: qsub submit_detr.sh
# Check job status: qstat
# ==============================================================================

#$ -N detr_kitti                          # Job name
#$ -P ec523                               # BU SCC project
#$ -pe omp 4                              # 4 CPU cores (matches --num_workers)
#$ -l gpus=1                              # Request 1 GPU
#$ -l gpu_c=7.0                           # Minimum CUDA compute capability
#$ -l h_rt=12:00:00                       # Max wall time (HH:MM:SS)
#$ -l mem_per_core=8G                     # Memory per core (4 cores x 8G = 32G total)
#$ -o /projectnb/ec523/students/serhat/detr_finetuning/outputs/job_$JOB_ID.log
#$ -e /projectnb/ec523/students/serhat/detr_finetuning/outputs/job_$JOB_ID.err
#$ -j y                                   # Merge stdout and stderr into one log file

# ==============================================================================
# PATHS — edit these if anything moves
# ==============================================================================
IMG_DIR=/projectnb/ec523/projects/proj_adversarial_weather/kitti/training/image_02
LBL_DIR=/projectnb/ec523/projects/proj_adversarial_weather/kitti/training/label_02
SRC_DIR=/projectnb/ec523/students/serhat/detr_finetuning/src
OUT_DIR=/projectnb/ec523/students/serhat/detr_finetuning/outputs

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
mkdir -p $OUT_DIR

module load miniconda
conda activate detr_env

# ==============================================================================
# RUN
# ==============================================================================
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd $SRC_DIR

# Normal run (first time):
python train.py \
    --img_dir     $IMG_DIR \
    --lbl_dir     $LBL_DIR \
    --output_dir  $OUT_DIR \
    --epochs      10 \
    --batch_size  4 \
    --num_workers 4

# To resume a job that was cut short, comment out the block above and
# uncomment the block below:
# python train.py \
#     --img_dir     $IMG_DIR \
#     --lbl_dir     $LBL_DIR \
#     --output_dir  $OUT_DIR \
#     --epochs      10 \
#     --batch_size  4 \
#     --num_workers 4 \
#     --resume

echo "Job finished: $(date)"
