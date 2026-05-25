## Installation

AlignG was developed and verified on the following environment:

- Ubuntu 22.04 / 24.04
- Python 3.12
- PyTorch 2.5.1 (CUDA 12.1)
- torchvision 0.20.1
- NVIDIA GeForce RTX 4090 (single GPU)

Newer PyTorch / CUDA combinations should work as long as `python setup.py build develop` succeeds. Older PyTorch (<1.9) will likely fail because the C++ extensions in `maskrcnn_benchmark/csrc/` use the modern TORCH_CHECK / dispatch macros.

### 1. Create a fresh conda environment

```bash
conda create -n alignG python=3.12 -y
conda activate alignG
```

### 2. Install PyTorch

Pick the wheel that matches your CUDA driver. The exact combination we used:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs `numpy`, `scipy`, `h5py`, `Pillow`, `opencv-python`, `yacs`, `tqdm`, `matplotlib`, `overrides`, and `pycocotools`.

### 4. Build the C++/CUDA extensions in-place

```bash
python setup.py build develop
```

This compiles `maskrcnn_benchmark._C` (custom NMS, ROIAlign, deform conv) against your local PyTorch. If you change CUDA arch / PyTorch version later, rerun this step.

### 5. (Optional) GloVe embeddings

AlignG reads word vectors from `cfg.GLOVE_DIR` (default `./datasets/vg/`). If you don't already have it, download `glove.6B.300d.txt` from the [Stanford NLP website](https://nlp.stanford.edu/projects/glove/) and place it under that directory. The training pipeline will convert it to a `.pt` cache on first use.

### 6. Datasets

See [DATASET.md](./DATASET.md) for the layout that `paths_catalog.py` expects:

- Visual Genome (VG-150): under `./datasets/vg/`
- GQA-200: under `./datasets/gqa/`

The expected sub-files are the standard SGG-benchmark splits.

### Verifying the install

After `setup.py build develop` succeeds:

```bash
python -c "from maskrcnn_benchmark import _C; print('csrc OK')"
python -c "from maskrcnn_benchmark.modeling import registry; \
           import maskrcnn_benchmark.modeling.roi_heads.relation_head.roi_relation_predictors; \
           print('Predictors:', list(registry.ROI_RELATION_PREDICTOR.keys()))"
```

The second command should print a list containing `SGGPredictor` (the AlignG predictor).
