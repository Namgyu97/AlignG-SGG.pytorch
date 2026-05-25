#!/usr/bin/env bash
# AlignG training launcher.
# Override MODE (PredCls | SGCls | SGDet) or MODEL_NAME via env vars if desired.

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_GPU=1

MODE=${MODE:-PredCls}
MODEL_NAME=${MODEL_NAME:-AlignG_${MODE}}

case "$MODE" in
  PredCls) USE_GT_BOX=True;  USE_GT_LBL=True  ;;
  SGCls)   USE_GT_BOX=True;  USE_GT_LBL=False ;;
  SGDet)   USE_GT_BOX=False; USE_GT_LBL=False ;;
  *) echo "MODE must be PredCls | SGCls | SGDet" >&2; exit 1 ;;
esac

OUTPUT_DIR=./checkpoints/${MODEL_NAME}
mkdir -p "${OUTPUT_DIR}"

echo "Training ${MODEL_NAME} (${MODE})"

python tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${USE_GT_BOX} \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${USE_GT_LBL} \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR SGGPredictor \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH ${NUM_GPU} \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 1024 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
  SOLVER.CHECKPOINT_PERIOD 30000 \
  GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ${OUTPUT_DIR} \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0
