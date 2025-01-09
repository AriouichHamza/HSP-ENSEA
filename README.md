# **LeNet-5 Implementation on GPU**

This repository contains the implementation of the inference phase of the **LeNet-5** Convolutional Neural Network (CNN) on GPU using **CUDA**. The project is part of a high-performance computing (HPC) practical course, focusing on the optimization and parallelization of deep learning algorithms.

---

## **Project Objectives**
- **CUDA Programming**: Learn to use CUDA for parallel computing, including thread, block, and grid structures.
- **Algorithm Complexity**: Study the computational complexity of CNN operations and compare execution times on CPU and GPU.
- **GPU Limitations**: Analyze the constraints of GPU usage, such as memory management and performance bottlenecks.
- **CNN Implementation**: Develop the inference phase of the LeNet-5 architecture from scratch (training not included).
- **Data Transfer**: Export data from a Python notebook and import it into a CUDA project.
- **Version Control**: Use Git for project tracking and version management.

---

## **LeNet-5 Architecture**
LeNet-5, introduced by Yann LeCun et al. in 1998, is a pioneering CNN designed for handwritten digit recognition. The architecture consists of:
1. **Input Layer**: 32x32 grayscale images.
2. **Convolutional Layers**: Feature extraction using filters.
3. **Pooling Layers**: Dimensionality reduction with subsampling.
4. **Fully Connected Layers**: Classification based on extracted features.

> **Note**: In this implementation, Layer 3 differs from the original paper by considering **all features for each output**.

For more details, refer to the original [LeNet-5 paper](https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture).