## Learning Context-Conditioned Predicate Semantics via Prototype Feedback

### Overview
<img width="1774" height="827" alt="image" src="https://github.com/user-attachments/assets/1f0750b1-9f1c-4e54-8a59-f8d47f639033" />

---

### Abstract
In scene graph generation, a central challenge is modeling polysemous predicates whose meanings shift across contexts. Prior approaches address this issue by decomposing predicates into multiple static prototypes or retrieving semantically similar exemplars. However, these strategies keep predicate representations static and cannot reorganize semantics to reflect image-specific evidence, leading to systematic confusions in ambiguous contexts. We propose **AlignG**, which learns context-conditioned predicate semantics via prototype feedback. AlignG infers image-conditioned predicate semantics from the relations within each image and feeds the adapted semantics back to recalibrate relation representations. The learning objective anchors this adaptation to global semantic centers, preventing semantic drift while still allowing selective reorganization when the scene provides consistent relational cues. Experiments on VG-150 and GQA-200 show consistent improvements over state-of-the-art baselines, with F@100 improvements of +1.4 and +2.7 under SGDet. We further visualize per-image prototype similarity shifts and observe coherent context-dependent reorganization where prototypes selectively merge or separate predicates according to scene evidence.

---

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Train
The predictor for AlignG is registered as `SGGPredictor`. A convenience
launcher is provided in `scripts/train.sh` (set `MODE` to `PredCls`, `SGCls`,
or `SGDet`); the full command it runs is:

```bash
python tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR SGGPredictor \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 1024 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
  SOLVER.CHECKPOINT_PERIOD 30000 \
  GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ./checkpoints/AlignG_PredCls \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0
```

For GQA-200 use `configs/e2e_relation_X_101_32_8_FPN_1x_GQA.yaml` instead.

## Test
Use `scripts/test.sh` (same `MODE` switch) or run
`python tools/relation_test_net.py` with the same flags and
`MODEL.WEIGHT ./checkpoints/AlignG_<MODE>/model_final.pth`.

## Device
All our experiments are conducted on a single NVIDIA GeForce RTX 4090.
For distributed training across multiple GPUs, follow the launch instructions
in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Acknowledgement
The code is built on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and the SGG-benchmark fork released with [PE-NET](https://github.com/VL-Group/PENET).

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{jung2026alignG,
  title     = {Learning Context-Conditioned Predicate Semantics via Prototype Feedback},
  author    = {Jung, NamGyu and Choi, Chang},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
