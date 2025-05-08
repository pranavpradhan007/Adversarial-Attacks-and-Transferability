# Adverserial-Attacks-and-Transferability

Team Name - Maha Vikas Aghadi 

This repository contains a Jupyter notebook (`dl-project-3.ipynb`) and accompanying scripts that implement and evaluate adversarial attacks on ImageNet-like data using a pre-trained ResNet-34 backbone. We explore single-step and iterative methods, patch-based attacks, and the transferability of these attacks to other architectures.

## Structure

```
├── dl-project-3.ipynb          # Main notebook implementing Tasks 1–5
├── README.md                   # This document
├── adversarial_sets.zip        # Zip file containing all the data sets created by tasks
├── TestDataSet.zip             # Zip file containing Test data set used for the project
```

## Tasks Overview

1. **Evaluation (Task 1)**
   - Load pre-trained ResNet-34.
   - Evaluate on the clean TestDataSet using top-1 and top-5.

2. **Pixel-Space FGSM (Task 2)**
   - Denormalize inputs to [0,1], apply ε=0.02 FGSM directly in pixel space.
   - Save **Adversarial_Test_Set_1** and re-evaluate performance.

3. **PGD (Task 3)**
   - Use 10-step PGD (Basic Iterative Method) with α=ε/10, random start.
   - Save **Adversarial_Test_Set_2** and measure drops in accuracy.

4. **Patch-Based Attack (Task 4)**
   - Implement single 32×32 patch PGD with ε=0.5, targeted to each image’s least-likely class.
   - Dynamically select the most sensitive patch per iteration.
   - Save **Adversarial_Test_Set_3** and report results.

5. **Transferability (Task 5)**
   - Evaluate all four datasets (clean + 3 adversarial) on ResNet-34, DenseNet-121, and ViT-B_16.
   - Plot top-1/top-5 accuracies and analyze transfer trends.

## Getting Started

### Requirements

- Python ≥ 3.8  
- PyTorch  
- torchvision  
- matplotlib  
- numpy  

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

### Running the Notebook

1. Place `dl-project-3.ipynb` in your working directory.  
2. Ensure the `datasets/TestDataSet` folder and `labels_list.json` are present.  
3. Execute cells in order to reproduce Tasks 1–5, generate adversarial sets, and view plots.

## Results

- **ResNet-34** drops from 76.0 → 3.0 (FGSM) → 0.0 (PGD) → 6.0 (patch PGD).  
- **DenseNet-121** and **ViT-B_16** trials show varying degrees of transferability.  
- Vision Transformers (ViT) display greater robustness to both full-image and localized perturbations.

## Lessons Learned

- **Simple FGSM transfers strongly among CNNs**, causing large accuracy drops across ResNet and DenseNet.  
- **Pixel-space and patch-based attacks** can be tuned under L∞ bounds to degrade performance further.  
- **Architectural diversity** (e.g., mixing CNNs and Transformers) and **adversarial training** mitigate cross-model transferability.  
- **Pre-/post-processing defenses** (random resize, JPEG compression, bit-depth reduction) can disrupt adversarial noise.
