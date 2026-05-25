#!/usr/bin/env bash
# AlignG evaluation launcher.
# Set MODEL_NAME to the checkpoint dir under ./checkpoints/ you want to evaluate.

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODE=${MODE:-PredCls}
MODEL_NAME=${MODEL_NAME:-AlignG_${MODE}}

case "$MODE" in
  PredCls) USE_GT_BOX=True;  USE_GT_LBL=True  ;;
  SGCls)   USE_GT_BOX=True;  USE_GT_LBL=False ;;
  SGDet)   USE_GT_BOX=False; USE_GT_LBL=False ;;
  *) echo "MODE must be PredCls | SGCls | SGDet" >&2; exit 1 ;;
esac

echo "Testing ${MODEL_NAME} (${MODE})"

python tools/relation_test_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${USE_GT_BOX} \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${USE_GT_LBL} \
  MODEL.ROI_RELATION_HEAD.PREDICTOR SGGPredictor \
  TEST.IMS_PER_BATCH 1 DTYPE "float32" \
  GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_final.pth \
  OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
  TEST.ALLOW_LOAD_FROM_CACHE False
