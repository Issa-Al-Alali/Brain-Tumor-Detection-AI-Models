# **Brain Tumor MRI Classification using Deep Learning**

A comprehensive research project comparing custom Convolutional Neural Networks (CNNs) against state-of-the-art Transfer Learning architectures for the automated classification of Brain Tumors.

## **📌 Project Overview**

The objective of this project is to build a robust deep learning model capable of classifying brain MRI scans into four distinct categories with high clinical accuracy.

**The Four Classes:**

1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor**

The research was conducted in two distinct phases:

- **Phase 1:** Iterative engineering of a custom CNN architecture from scratch.
- **Phase 2:** Advanced Transfer Learning using pre-trained ImageNet models (ResNet, EfficientNet) with a progressive unfreezing strategy.

## **📂 Dataset**

The dataset used for this project is the **Brain Tumor MRI Dataset** available on Kaggle.

- **Link:** [Brain Tumor MRI Dataset (Masoud Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Preprocessing:**
  - **Custom Models:** Resized to 128x128, Normalized \[-1, 1\].
  - **Transfer Learning:** Resized to 224x224, Normalized using ImageNet mean/std.
- **Splitting Protocol:**
  - **Train:** 85% (with Augmentation)
  - **Validation:** 15% (No Augmentation)
  - **Test:** Held-out set evaluated **only once** per model version to ensure zero data leakage.

## **🧪 Phase 1: Custom CNN Engineering**

We iteratively improved a custom architecture through four versions to find the optimal balance between model capacity and regularization.

| Version            | Description                                  | Accuracy   | Status          |
| :----------------- | :------------------------------------------- | :--------- | :-------------- |
| **V1 (Baseline)**  | Shallow 2-layer CNN. No regularization.      | **92.37%** | Overfitting     |
| **V2 (Improved)**  | Deeper 3-layer CNN with Dropout (0.5).       | **94.97%** | Stable          |
| **V3 (Advanced)**  | Heavy Augmentation \+ L2 Decay.              | **82.84%** | Underfitting    |
| **V4 (Optimized)** | **Balanced Regularization \+ LR Scheduler.** | **98.09%** | **Best Custom** |

### **Key Takeaway (Phase 1\)**

Aggressive regularization (V3) destroys performance on medical images. A balanced approach (V4) with mild augmentation and learning rate scheduling yields the best results for custom architectures.

## **🚀 Phase 2: Transfer Learning Optimization**

To surpass the 98.09% benchmark, we utilized models pre-trained on ImageNet. We employed a **Progressive Unfreezing** strategy (4 phases) and **Discriminative Learning Rates** to preserve the pre-trained weights while adapting the classifier head.

| Model       | Architecture    | Test Accuracy | Verdict               |
| :---------- | :-------------- | :------------ | :-------------------- |
| **Model A** | EfficientNet-B0 | 95.42%        | Discarded             |
| **Model B** | ResNet18        | 97.25%        | Good                  |
| **Model C** | **ResNet50**    | **98.47%**    | **SOTA / Production** |

### **Why ResNet50?**

ResNet50 achieved the project's highest accuracy (**98.47%**). Its deep residual architecture successfully modeled complex tumor boundaries, resolving the confusion between _Glioma_ and _Meningioma_ classes that persisted in lighter models.

## **🛠️ Usage**

### **1\. Installation**

Ensure you have Python 3.8+ and PyTorch installed.

pip install torch torchvision matplotlib numpy scikit-learn tqdm seaborn

### **2\. Training a Model**

To train the optimal ResNet50 model:

1. Download the dataset from Kaggle.
2. Update the train_path and test_path variables in models/phase2_transfer_learning/resnet_50_model.py.
3. Run the script:

python models/phase2_transfer_learning/resnet_50_model.py

## **📈 Final Results Summary**

| Metric            | Best Custom (V4) | Best Transfer Learning (ResNet50) |
| :---------------- | :--------------- | :-------------------------------- |
| **Test Accuracy** | 98.09%           | **98.47%**                        |
| **Precision**     | 98.0%            | **98.5%**                         |
| **Pituitary Acc** | 100%             | **100%**                          |
| **No Tumor Acc**  | 98.8%            | **99.1%**                         |

**Conclusion:** While custom lightweight models are highly effective, leveraging deep residual networks via transfer learning provides the superior feature extraction needed for subtle medical distinctions, achieving state-of-the-art performance.
