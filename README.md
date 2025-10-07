# Image Classification on CIFAR-10: Comparing CNNs and Vision Transformers

## Overview
This repository contains the final project for COMS 573, presenting a comparative analysis of three neural network architectures for image classification on the CIFAR-10 dataset. We explore the performance trade-offs between a baseline Convolutional Neural Network (CNN), a DenseNet, and a pre-trained Vision Transformer (ViT).

Our key achievement was fine-tuning the state-of-the-art DINOv2-Small model using a parameter-efficient approach, where we updated only 8.06% of its total parameters to achieve an impressive 95.95% test accuracy.

## Team Members
- Md Muzakker Hossain: DINO-v2 pre-trained ViT + fine-tune with LoRA + Evaluation
- Nhat Le: custom CNN model + Architectures + Evaluation
- Jasper Khor: Data Preprocessing

## Key Findings & Results
Our experiments show a clear progression in performance, highlighting the power of modern architectures and transfer learning.

| Model | Test Accuracy (%) | Test Loss | Total Parameters | Fine-Tuned Params (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Shallow CNN** | [cite_start]83.46% [cite: 9] | [cite_start]0.5299 [cite: 154] | [cite_start]~4.8M [cite: 211] | 100% |
| **DenseNet** | [cite_start]91.22% [cite: 9] | [cite_start]0.3692 [cite: 154] | [cite_start]~7.4M [cite: 213] | 100% |
| **DINOv2-Small (ViT)** | 95.95%** [cite: 9] | 0.1243** [cite: 154] | [cite_start]22.06M [cite: 120] | 8.06%** [cite: 8] |

### Insights:
* Transformer Superiority**: The pre-trained DINOv2 Vision Transformer significantly outperformed both CNN architectures, demonstrating the effectiveness of self-attention mechanisms for capturing global image features[cite: 159, 217, 223].
* Power of Transfer Learning**: DINOv2's strong performance, achieved by fine-tuning just a fraction of its weights, proves that knowledge from large-scale, self-supervised pre-training is highly transferable, even to smaller datasets like CIFAR-10[cite: 10, 161, 224].
* Rapid Convergence**: Thanks to its robust pre-trained features, the DINOv2 model converged extremely quickly, achieving over 95% validation accuracy within the first 10 epochs of training[cite: 195, 196].
* Architectural Value**: The DenseNet model's architecture, with its emphasis on feature reuse, provided a substantial accuracy boost over the baseline CNN, making it a strong intermediate choice[cite: 158, 222].

[cite_start]*The training and validation curves for our fine-tuned DINOv2 model, showing rapid convergence and stable performance[cite: 163].*
---

## Repository Structure
This repository is organized to provide a complete overview of our project methodology and findings.
* `COMS_573_Final_Project.pdf`: The complete project report containing detailed methodology, analysis, and conclusions.
* `notebooks/`: Jupyter/Colab notebooks used for implementing, training, and evaluating the models.
* `images/`: Contains diagrams and plots used in the report and this README.
* `dataset/`: Information regarding the CIFAR-10 dataset used for this project.
---

## Dataset & Preprocessing
[cite_start]We used the **CIFAR-10 dataset**, a standard benchmark for image classification[cite: 24, 67].
* Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) [cite: 23]
* Image Size**: 32x32 RGB color images [cite: 23]
* Split**: The original 50,000 training images were split into 45,000 for training and 5,000 for validation[cite: 26]. [cite_start]The 10,000 test images were used for final evaluation[cite: 27].

### Preprocessing Steps:
For the training set, we applied data augmentation to improve model generalization:
* [cite_start]Random 32x32 cropping with a padding of 4 [cite: 69]
* [cite_start]Random horizontal flipping [cite: 69]
* [cite_start]Random rotation (up to 15 degrees) [cite: 69]
* [cite_start]Normalization (mean: `[0.4914, 0.4822, 0.4465]`, std: `[0.247, 0.243, 0.261]`) [cite: 69]

[cite_start]The validation and test sets were only normalized[cite: 70].
---

## Models & Configurations
### 1. Shallow CNN (Baseline)
[cite_start]A simple, lightweight CNN to establish a performance baseline[cite: 7, 74].
* Architecture**: 3 convolutional blocks (Conv -> Batch Norm -> ReLU -> MaxPool) followed by 2 fully connected layers[cite: 74, 75].
* Filters**: Progressively increased from 32 to 64 to 128[cite: 75].
* Regularization**: Dropout layer to prevent overfitting[cite: 75].
* Optimizer**: AdamW with a learning rate of `0.001`[cite: 94].

### 2. DenseNet
[cite_start]A custom implementation inspired by the original paper, designed to enhance feature reuse[cite: 79].
* Architecture**: 3 dense blocks, each with 16 layers[cite: 81, 89].
* Growth Rate**: 12 [cite: 81]
* Bottleneck Layers**: Used to reduce dimensionality and improve efficiency[cite: 90].
* Regularization**: Dropout (`rate=0.1`) within each dense block[cite: 90].
* Optimizer**: AdamW (learning rate `0.001`, weight decay `0.0001`)[cite: 94].
* Epochs**: 50 [cite: 94]

### 3. DINOv2-Small Vision Transformer (ViT)
[cite_start]A pre-trained transformer model adapted for CIFAR-10 using a parameter-efficient fine-tuning strategy[cite: 37].
* Base Model**: `dinov2_vits14` from Facebook Research[cite: 115].
* Architecture**: 12 transformer blocks, 6 attention heads, and an embedding dimension of 384[cite: 116, 118].
* **Fine-Tuning Strategy (PEFT)**:
    * [cite_start]Froze all layers except the final classification head and the last transformer block[cite: 127].
    * [cite_start]Implemented **Low-Rank Adaptation (LoRA)** with a rank of 16 to efficiently adapt attention layers[cite: 129].
    * [cite_start]This resulted in only **1,779,082 trainable parameters (8.06% of total)**[cite: 128, 203].
* **Training Details**:
    * Input Size**: CIFAR-10 images were resized from 32x32 to 224x224[cite: 139].
    * Optimizer**: AdamW (learning rate `1e-4`, weight decay `0.01`)[cite: 135].
    * Scheduler**: Cosine annealing learning rate scheduler[cite: 136].
    * Batching**: Effective batch size of 32 (8 physical batch size with 4 gradient accumulation steps)[cite: 137].
    * Epochs**: 80 [cite: 138]
    * **Hardware**: Trained on a single NVIDIA Tesla V100 GPU for approx. [cite_start]2 hours and 9 minutes[cite: 141, 206].
---

## Technology Stack
* Framework**: PyTorch [cite: 68]
* **Libraries**: Torchvision, NumPy, Matplotlib
* Fine-Tuning**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA [cite: 125, 129]
* Hardware**: NVIDIA Tesla V100 GPU with CUDA [cite: 141]

## 📜 References
* Oquab, M., et al. (2023). [cite_start]*DINOv2: Learning Robust Visual Features without Supervision*. [cite: 249]
* Huang, G., et al. (2017). [cite_start]*Densely Connected Convolutional Networks*. [cite: 239]
* Dosovitskiy, A., et al. (2020). [cite_start]*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. [cite: 236]
