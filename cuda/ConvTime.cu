#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define RAW_DATA_SIZE 32
#define C1_SIZE 28
#define S1_SIZE 14
#define KERNEL_SIZE 5
#define NUM_KERNELS 6

__device__ float activation_tanh(float x) {
    return tanhf(x);
}

// Fonction d'activation pour le CPU
float activation_tanh_cpu(float x) {
    return tanhf(x);
}

// Initialisation des matrices avec des valeurs aléatoires entre 0 et 1
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX);
    }
}

// Convolution 2D sur CPU
void convolution2D_CPU(float *input, float *kernel, float *output, int inputSize, int outputSize, int kernelSize) {
    for (int k = 0; k < NUM_KERNELS; k++) { // Pour chaque noyau
        for (int i = 0; i < outputSize; i++) { // Parcourir les lignes de sortie
            for (int j = 0; j < outputSize; j++) { // Parcourir les colonnes de sortie
                float sum = 0.0f;
                for (int ki = 0; ki < kernelSize; ki++) { // Parcourir les lignes du noyau
                    for (int kj = 0; kj < kernelSize; kj++) { // Parcourir les colonnes du noyau
                        int inputRow = i + ki;
                        int inputCol = j + kj;
                        sum += input[inputRow * inputSize + inputCol] * kernel[k * kernelSize * kernelSize + ki * kernelSize + kj];
                    }
                }
                output[k * outputSize * outputSize + i * outputSize + j] = activation_tanh_cpu(sum);
            }
        }
    }
}

// Convolution 2D sur GPU
__global__ void cudaConvolution2D(float *input, float *kernel, float *output, int inputSize, int outputSize, int kernelSize) {
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputRow < outputSize && outputCol < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                int inputRow = outputRow + i;
                int inputCol = outputCol + j;
                sum += input[inputRow * inputSize + inputCol] * kernel[blockIdx.z * kernelSize * kernelSize + i * kernelSize + j];
            }
        }
        output[(blockIdx.z * outputSize * outputSize) + (outputRow * outputSize + outputCol)] = activation_tanh(sum);
    }
}


void MatrixPrint(float *M) {
    int n  = sizeof(M) / sizeof(M[0]);
    int p = sizeof(M[0]) / sizeof(M[0]);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%f ", M[i * p + j]);
        }
        printf("\n");
    }
}

int main() {
    // Allocation des matrices
    float *raw_data = (float *)malloc(RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float));
    float *C1_kernel = (float *)malloc(NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *C1_data_cpu = (float *)malloc(NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float));
    float *C1_data_gpu = (float *)malloc(NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float));

    // Initialisation des données
    initializeMatrix(raw_data, RAW_DATA_SIZE * RAW_DATA_SIZE);
    initializeMatrix(C1_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE);

    // Mesurer le temps de la convolution sur CPU
    clock_t start_cpu = clock();
    convolution2D_CPU(raw_data, C1_kernel, C1_data_cpu, RAW_DATA_SIZE, C1_SIZE, KERNEL_SIZE);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    printf("Convolution CPU Time: %f ms\n", cpu_time);

    // Allocation sur GPU
    float *d_raw_data, *d_C1_kernel, *d_C1_data;
    cudaMalloc((void **)&d_raw_data, RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C1_data, NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float));

    // Copie des données vers le GPU
    cudaMemcpy(d_raw_data, raw_data, RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Mesurer le temps de la convolution sur GPU
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    dim3 blockDim(16, 16);
    dim3 gridDim((C1_SIZE + blockDim.x - 1) / blockDim.x, (C1_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS);

    cudaEventRecord(start_gpu);
    cudaConvolution2D<<<gridDim, blockDim>>>(d_raw_data, d_C1_kernel, d_C1_data, RAW_DATA_SIZE, C1_SIZE, KERNEL_SIZE);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    printf("Convolution GPU Time: %f ms\n", gpu_time);

    // Copie des résultats GPU vers l'hôte
    cudaMemcpy(C1_data_gpu, d_C1_data, NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Libération de la mémoire
    free(raw_data);
    free(C1_kernel);
    free(C1_data_cpu);
    free(C1_data_gpu);
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);

    return 0;
}
