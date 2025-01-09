#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Définitions des tailles
#define IMG_SIZE 28
#define C1_KERNEL_SIZE 5
#define C1_OUTPUT_SIZE 28
#define S2_SIZE 14
#define C3_KERNEL_SIZE 5
#define C3_OUTPUT_SIZE 10
#define S4_SIZE 5
#define FC1_SIZE 120
#define FC2_SIZE 84
#define OUTPUT_SIZE 10

// Charger un fichier binaire dans un tableau
void LoadBinaryFile(const char* filename, float* array, int size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier : %s\n", filename);
        exit(1);
    }
    fread(array, sizeof(float), size, file);
    fclose(file);
}

// CUDA Kernel pour la convolution
__global__ void Convolution2D(float* input, float* kernel, float* bias, float* output, int inputSize, int kernelSize, int outputSize, int numKernels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernelIdx = blockIdx.z;

    if (x < outputSize && y < outputSize && kernelIdx < numKernels) {
        float sum = bias[kernelIdx];
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                int inputX = x + j - kernelSize / 2;
                int inputY = y + i - kernelSize / 2;
                if (inputX >= 0 && inputX < inputSize && inputY >= 0 && inputY < inputSize) {
                    sum += input[inputY * inputSize + inputX] * kernel[kernelIdx * kernelSize * kernelSize + i * kernelSize + j];
                }
            }
        }
        output[(kernelIdx * outputSize + y) * outputSize + x] = sum;
    }
}

// CUDA Kernel pour le sous-échantillonnage (Average Pooling)
__global__ void AveragePooling(float* input, float* output, int inputSize, int outputSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputSize && y < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                sum += input[(y * 2 + i) * inputSize + (x * 2 + j)];
            }
        }
        output[y * outputSize + x] = sum / 4.0f;
    }
}

// Fonction d'activation Tanh
__device__ float* activation_tanh(float* M, int M_ligne, int M_colonne, int M_prof) {
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < M_ligne && col < M_colonne) {
        int tot_M = M_ligne * M_colonne;
        for (int n_prof = 0; n_prof < M_prof; n_prof++) {
            M[lig * M_colonne + col + n_prof * tot_M] = tanh(M[lig * M_colonne + col + n_prof * tot_M]);
        }
    }
    return M;
}

__global__ void cudaTanh(float* M, int M_ligne, int M_colonne, int M_prof) {
    activation_tanh(M, M_ligne, M_colonne, M_prof);
}

// CUDA Kernel pour les couches entièrement connectées
__global__ void FullyConnected(float* input, float* weights, float* bias, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < outputSize) {
        float sum = bias[idx];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[idx * inputSize + i];
        }
        output[idx] = sum;
    }
}
// Fonction pour afficher l'image MNIST dans le terminal
void PrintImage(float* image, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (image[i * size + j] > 0.5)
                printf("██");
            else if (image[i * size + j] > 0.2)
                printf("▓▓");
            else
                printf("  ");
        }
        printf("\n");
    }
}

// Fonction argmax pour obtenir la classe prédite
__device__ int ArgMax(float* array, int size) {
    int maxIdx = 0;
    for (int i = 1; i < size; ++i) {
        if (array[i] > array[maxIdx]) {
            maxIdx = i;
        }
    }
    return maxIdx;
}

__global__ void cudaArgMax(float* array, int size, int* maxIdx) {
    *maxIdx = ArgMax(array, size);
}


// Fonction d'activation softmax
__global__ void Softmax(float* input, float* output, int size) {
    float sum = 0.0f;
    float max = 0.0f;
    for (int i = 0; i < size; ++i) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    for (int i = 0; i < size; ++i) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i]  /= sum;
    }
}




int main() {
    // Charger l'image MNIST
    float* image = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/mnist_image3.bin", image, IMG_SIZE * IMG_SIZE);

    // Charger les poids et biais
    float* C1_weights = (float*)malloc(6 * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float));
    float* C1_bias = (float*)malloc(6 * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_0_weights.bin", C1_weights, 6 * C1_KERNEL_SIZE * C1_KERNEL_SIZE);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_0_bias.bin", C1_bias, 6);

    float* C3_weights = (float*)malloc(16 * C3_KERNEL_SIZE * C3_KERNEL_SIZE * sizeof(float));
    float* C3_bias = (float*)malloc(16 * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_2_weights.bin", C3_weights, 16 * C3_KERNEL_SIZE * C3_KERNEL_SIZE);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_2_bias.bin", C3_bias, 16);

    float* FC1_weights = (float*)malloc(FC1_SIZE * S4_SIZE * S4_SIZE * 16 * sizeof(float));
    float* FC1_bias = (float*)malloc(FC1_SIZE * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_5_weights.bin", FC1_weights, FC1_SIZE * S4_SIZE * S4_SIZE * 16);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_5_bias.bin", FC1_bias, FC1_SIZE);

    float* FC2_weights = (float*)malloc(FC2_SIZE * FC1_SIZE * sizeof(float));
    float* FC2_bias = (float*)malloc(FC2_SIZE * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_6_weights.bin", FC2_weights, FC2_SIZE * FC1_SIZE);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_6_bias.bin", FC2_bias, FC2_SIZE);

    float* Output_weights = (float*)malloc(OUTPUT_SIZE * FC2_SIZE * sizeof(float));
    float* Output_bias = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_7_weights.bin", Output_weights, OUTPUT_SIZE * FC2_SIZE);
    LoadBinaryFile("/home/ariouich/Bureau/3A/HP/HSP/Cuda_HSP/python/weights1/layer_7_bias.bin", Output_bias, OUTPUT_SIZE);

    // Afficher l'image MNIST
    printf("Image MNIST :\n");
    PrintImage(image, IMG_SIZE);


    // print the weights of the first layer
    printf("Poids de la première couche :\n");
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < C1_KERNEL_SIZE; j++) {
            for (int k = 0; k < C1_KERNEL_SIZE; k++) {
                printf("%.2f ", C1_weights[i * C1_KERNEL_SIZE * C1_KERNEL_SIZE + j * C1_KERNEL_SIZE + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
    printf("Biais de la première couche :\n");
    for (int i = 0; i < 6; i++) {
        printf("%.2f ", C1_bias[i]);
    }

    //  le traitement de l'image à travers les couches C1, S2, C3, S4, et Fully Connected.
    // Allocation de mémoire sur le GPU
    float *d_image, *d_C1_weights, *d_C1_bias, *d_C1_output;
    float *d_S2_output, *d_C3_weights, *d_C3_bias, *d_C3_output;
    float *d_S4_output, *d_FC1_weights, *d_FC1_bias, *d_FC1_output;
    float *d_FC2_weights, *d_FC2_bias, *d_FC2_output;
    float *d_Output_weights, *d_Output_bias, *d_Output;

    cudaMalloc(&d_image, IMG_SIZE * IMG_SIZE * sizeof(float));
    cudaMalloc(&d_C1_weights, 6 * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_C1_bias, 6 * sizeof(float));
    cudaMalloc(&d_C1_output, 6 * C1_OUTPUT_SIZE * C1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_S2_output, 6 * S2_SIZE * S2_SIZE * sizeof(float));
    cudaMalloc(&d_C3_weights, 16 * C3_KERNEL_SIZE * C3_KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_C3_bias, 16 * sizeof(float));
    cudaMalloc(&d_C3_output, 16 * C3_OUTPUT_SIZE * C3_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_S4_output, 16 * S4_SIZE * S4_SIZE * sizeof(float));
    cudaMalloc(&d_FC1_weights, FC1_SIZE * S4_SIZE * S4_SIZE * 16 * sizeof(float));
    cudaMalloc(&d_FC1_bias, FC1_SIZE * sizeof(float));
    cudaMalloc(&d_FC1_output, FC1_SIZE * sizeof(float));
    cudaMalloc(&d_FC2_weights, FC2_SIZE * FC1_SIZE * sizeof(float));
    cudaMalloc(&d_FC2_bias, FC2_SIZE * sizeof(float));
    cudaMalloc(&d_FC2_output, FC2_SIZE * sizeof(float));
    cudaMalloc(&d_Output_weights, OUTPUT_SIZE * FC2_SIZE * sizeof(float));
    cudaMalloc(&d_Output_bias, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_Output, OUTPUT_SIZE * sizeof(float));

    // Copie des données sur le GPU
    cudaMemcpy(d_image, image, IMG_SIZE * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_weights, C1_weights, 6 * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_bias, C1_bias, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_weights, C3_weights, 16 * C3_KERNEL_SIZE * C3_KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_bias, C3_bias, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FC1_weights, FC1_weights, FC1_SIZE * S4_SIZE * S4_SIZE * 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FC1_bias, FC1_bias, FC1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FC2_weights, FC2_weights, FC2_SIZE * FC1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FC2_bias, FC2_bias, FC2_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Output_weights, Output_weights, OUTPUT_SIZE * FC2_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Output_bias, Output_bias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Définir les dimensions des blocs et des grilles
    dim3 blockDim(16, 16);
    dim3 gridDimC1((C1_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (C1_OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, 6);
    dim3 gridDimS2((S2_SIZE + blockDim.x - 1) / blockDim.x, (S2_SIZE + blockDim.y - 1) / blockDim.y);
    dim3 gridDimC3((C3_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (C3_OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, 16);
    dim3 gridDimS4((S4_SIZE + blockDim.x - 1) / blockDim.x, (S4_SIZE + blockDim.y - 1) / blockDim.y);
    dim3 gridDimFC1((FC1_SIZE + blockDim.x - 1) / blockDim.x);
    dim3 gridDimFC2((FC2_SIZE + blockDim.x - 1) / blockDim.x);
    dim3 gridDimOutput((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x);

    // Pipeline de traitement
    Convolution2D<<<gridDimC1, blockDim>>>(d_image, d_C1_weights, d_C1_bias, d_C1_output, IMG_SIZE, C1_KERNEL_SIZE, C1_OUTPUT_SIZE, 6);
    cudaTanh<<<gridDimC1, blockDim>>>(d_C1_output, C1_OUTPUT_SIZE, C1_OUTPUT_SIZE, 6);
    AveragePooling<<<gridDimS2, blockDim>>>(d_C1_output, d_S2_output, C1_OUTPUT_SIZE, S2_SIZE);
    Convolution2D<<<gridDimC3, blockDim>>>(d_S2_output, d_C3_weights, d_C3_bias, d_C3_output, S2_SIZE, C3_KERNEL_SIZE, C3_OUTPUT_SIZE, 16);
    cudaTanh<<<gridDimC3, blockDim>>>(d_C3_output, C3_OUTPUT_SIZE, C3_OUTPUT_SIZE, 16);
    AveragePooling<<<gridDimS4, blockDim>>>(d_C3_output, d_S4_output, C3_OUTPUT_SIZE, S4_SIZE);
    FullyConnected<<<gridDimFC1, blockDim>>>(d_S4_output, d_FC1_weights, d_FC1_bias, d_FC1_output, S4_SIZE * S4_SIZE * 16, FC1_SIZE);
    cudaTanh<<<gridDimFC1, blockDim>>>(d_FC1_output, FC1_SIZE, 1, 1);
    FullyConnected<<<gridDimFC2, blockDim>>>(d_FC1_output, d_FC2_weights, d_FC2_bias, d_FC2_output, FC1_SIZE, FC2_SIZE);
    cudaTanh<<<gridDimFC2, blockDim>>>(d_FC2_output, FC2_SIZE, 1, 1);
    FullyConnected<<<gridDimOutput, blockDim>>>(d_FC2_output, d_Output_weights, d_Output_bias, d_Output, FC2_SIZE, OUTPUT_SIZE);
    Softmax<<<1, 1>>>(d_Output, d_Output, OUTPUT_SIZE);

    // Copier le résultat sur le CPU
    float prediction[OUTPUT_SIZE];
    cudaMemcpy(prediction, d_Output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Afficher la prédiction
    int* d_maxIdx;
    int maxIdx;
    cudaMalloc(&d_maxIdx, sizeof(int));
    cudaArgMax<<<1, 1>>>(d_Output, OUTPUT_SIZE, d_maxIdx);
    cudaMemcpy(&maxIdx, d_maxIdx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_maxIdx);

    printf("Prédiction finale :\n");
    printf("Classe %d : %.2f\n", maxIdx, prediction[maxIdx]);
    

    // Libération de la mémoire
    cudaFree(d_image);
    cudaFree(d_C1_weights);
    cudaFree(d_C1_bias);
    cudaFree(d_C1_output);
    cudaFree(d_S2_output);
    cudaFree(d_C3_weights);
    cudaFree(d_C3_bias);
    cudaFree(d_C3_output);
    cudaFree(d_S4_output);
    cudaFree(d_FC1_weights);
    cudaFree(d_FC1_bias);
    cudaFree(d_FC1_output);
    cudaFree(d_FC2_weights);
    cudaFree(d_FC2_bias);
    cudaFree(d_FC2_output);
    cudaFree(d_Output_weights);
    cudaFree(d_Output_bias);
    cudaFree(d_Output);
    free(image);
    free(C1_weights);
    free(C1_bias);
    free(C3_weights);
    free(C3_bias);
    free(FC1_weights);
    free(FC1_bias);
    free(FC2_weights);
    free(FC2_bias);
    free(Output_weights);
    free(Output_bias);

    return 0;
}
