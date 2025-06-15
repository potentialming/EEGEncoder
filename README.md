<<<<<<< HEAD
# EEG2Gaussian

> **Decoding and Visualizing Visual‑Evoked EEG for VR Scenes Using 3D Gaussian Splatting**

This repository hosts the **official implementation** of the paper **“EEG2Gaussian: Decoding and Visualizing Visual‑Evoked EEG for VR Scenes Using 3D Gaussian Splatting.”** It provides everything you need to reproduce the experiments—from EEG signal decoding and vision–language alignment to final 3D Gaussian‑Splatting reconstruction.

![alt text](images/teaser.png)

---

## 🗂️ Project Layout

```text
EEG2GAUSSIAN/
├── NTF-Encoder/           # Core models & training scripts
│   ├── dataset.py                 # EEG / image loading & preprocessing
│   ├── model_imagenet.py          # EEG encoder for pure classification
│   ├── model_imagenet_clip.py     # EEG encoder with CLIP‑alignment head
│   ├── train_ImageNet.py          # ImageNet‑style classification training
│   └── train_ImageNet_clip.py     # Joint classification + alignment training
│
├── results/               # Auto‑generated weights, logs, plots
└── README.md              # You are here
```

---

## ✨ Highlights

- **EEG ⇄ Image Semantic Alignment** – Uses [OpenAI CLIP](https://github.com/openai/CLIP) embeddings to explicitly pull EEG features into the visual–semantic space.
- **Pluggable NTF‑Encoder** – A lightweight, dual‑domain attention network that works with ImageNet‑EEG, 360‑EEG, and more.
- **One‑Click Training & Evaluation** – `train_ImageNet_clip.py` performs joint classification + alignment training and automatically saves the best checkpoint.
- **3D Gaussian Splatting Output** – Converts Stable‑Diffusion panoramas to Gaussian point clouds for immersive VR visualization (to be open‑sourced soon).

---

## ⚙️ Requirements

| Package                      | Version          | Notes                                                |
| ---------------------------- | ---------------- | ---------------------------------------------------- |
| Python                       | ≥ 3.8            |                                                      |
| PyTorch                      | ≥ 1.13           | CUDA 11 recommended                                  |
| torchvision                  | matching PyTorch |                                                      |
| numpy / scipy / scikit‑learn | latest           | Signal processing & PCA                              |
| pillow                       | ≥ 9.0            | Image I/O                                            |
| ftfy / regex                 | latest           | CLIP dependencies                                    |
| **openai‑clip**              | latest           | `pip install git+https://github.com/openai/CLIP.git` |

### Quick Setup

```bash
conda create -n eeg2gs python=3.10 -y
conda activate eeg2gs
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn pillow ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

---

## 📦 Data Preparation

### 1. ImageNet‑EEG

1. Place `eeg_5_95_std.pth` and `block_splits_by_image_single.pth` under `./dataset/ImageNet/EEG/`.
2. Download the corresponding ImageNet images to `./dataset/ImageNet/imageNet_images/` (or symlink the folder).

> A helper inside `NTF-Encoder/dataset.py` can convert raw `.mat`/`.edf` EEG files to the `.pth` format above if you wish to start from scratch.

### 2. 360‑EEG (optional)

Align 360° panoramas with your EEG recordings and point `EEGDataset_360EEG` to the new paths—no other changes required.

---

## 🚀 Training & Evaluation

### 1️⃣  Joint ImageNet + CLIP Training

```bash
python NTF-Encoder/train_ImageNet_clip.py \
  --data_path   ./dataset/ImageNet/EEG/eeg_5_95_std.pth \
  --images_path ./dataset/ImageNet/imageNet_images \
  --split_file  ./dataset/ImageNet/block_splits_by_image_single.pth \
  --batch_size 32 --epochs 200 --lr 3e-4
```

During training the script creates a time‑stamped folder in `./results/imageNet_clip/` containing:

- `best_eeg_encoder.pth` — checkpoint with the lowest validation loss
- `train_val_loss.png` — loss curves
- `results.txt` — top‑1 accuracy and embedding‑matching accuracy

### 2️⃣  Classification‑Only Baseline

```bash
python NTF-Encoder/train_ImageNet.py --help  # full list of options
```

### 3️⃣  Inference / Feature Extraction

```python
import torch
from model_imagenet_clip import EEGEncoder
model = EEGEncoder().eval()
model.load_state_dict(torch.load('best_eeg_encoder.pth'))
with torch.no_grad():
    logits, eeg_embed, _ = model(eeg_tensor)  # eeg_tensor: (B, C, T)
```

---

## 📈 Reproducing Our Results

On the public ImageNet‑EEG split we obtain (single RTX 3090 24 GB):

| Setup                 | Top‑1 Acc  | CLIP Matching (> 0.5) |
| --------------------- | ---------- | --------------------- |
| **Ours (EEG + CLIP)** | **41.2 %** | **67.8 %**            |
| EEG Baseline          | 34.5 %     | –                     |

Qualitative reconstructions and additional visuals are available in `images/result.png`.

---

## 📜 Citation

```bibtex
@article{zhang2025eeg2gaussian,
  title   = {EEG2Gaussian: Decoding and Visualizing Visual‑Evoked EEG for VR Scenes Using 3D Gaussian Splatting},
  author  = {Zhang, Liming and Li, X.X. and Wang, X.X.},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2025}
}
```

---

## 📝 License

Released under the **MIT License**—see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP) — vision–language alignment
- [3D Gaussian Splatting](https://repo-supplied-link) — fast 3D scene reconstruction
- The ImageNet‑EEG team and the open‑source community for their invaluable resources.

=======
This is the EEG encoder section of the EEG2Gaussian project.
![alt text](images/overview.png)
![alt text](images/pipeline.png)
![alt text](images/result.png)
>>>>>>> 7438cc1171a1d896c677bf999526be40d35f943c
