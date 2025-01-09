#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

// Définition des tailles
#define RAW_DATA_SIZE 32
#define C1_SIZE 28
#define S1_SIZE 14
#define KERNEL_SIZE 5
#define NUM_KERNELS 6

__device__ float activation_tanh(float x) {
    return tanhf(x);
}

void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX);
    }
}

__global__ void cudaConvolution2D(float *input, float *kernel, float *output, int inputSize, int outputSize, int kernelSize) {
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputRow < outputSize && outputCol < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                int inputRow = outputRow + i;
                int inputCol = outputCol + j;
                sum += input[inputRow * inputSize + inputCol] * kernel[i * kernelSize + j];
            }
        }
        output[(blockIdx.z * outputSize * outputSize) + (outputRow * outputSize + outputCol)] = activation_tanh(sum);
    }
}

__global__ void cudaSubsampling(float *input, float *output, int inputSize, int outputSize) {
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputRow < outputSize && outputCol < outputSize) {
        int inputRow = outputRow * 2;
        int inputCol = outputCol * 2;

        float sum = 0.0f;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                sum += input[(inputRow + i) * inputSize + (inputCol + j)];
            }
        }
        output[blockIdx.z * outputSize * outputSize + outputRow * outputSize + outputCol] = sum / 4.0f;
    }
}

int main() {
    float *raw_data = (float *)malloc(RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float));
    float *C1_kernel = (float *)malloc(NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *C1_data = (float *)malloc(NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float));
    float *S1_data = (float *)malloc(NUM_KERNELS * S1_SIZE * S1_SIZE * sizeof(float));

    initializeMatrix(raw_data, RAW_DATA_SIZE * RAW_DATA_SIZE);
    initializeMatrix(C1_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE);

    float *d_raw_data, *d_C1_kernel, *d_C1_data, *d_S1_data;
    cudaMalloc((void **)&d_raw_data, RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C1_data, NUM_KERNELS * C1_SIZE * C1_SIZE * sizeof(float));
    cudaMalloc((void **)&d_S1_data, NUM_KERNELS * S1_SIZE * S1_SIZE * sizeof(float));

    cudaMemcpy(d_raw_data, raw_data, RAW_DATA_SIZE * RAW_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Mesurer le temps de la convolution
    cudaEvent_t start_conv, stop_conv;
    cudaEventCreate(&start_conv);
    cudaEventCreate(&stop_conv);

    dim3 blockDim(16, 16);
    dim3 gridDim((C1_SIZE + blockDim.x - 1) / blockDim.x, (C1_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS);

    cudaEventRecord(start_conv);
    cudaConvolution2D<<<gridDim, blockDim>>>(d_raw_data, d_C1_kernel, d_C1_data, RAW_DATA_SIZE, C1_SIZE, KERNEL_SIZE);
    cudaEventRecord(stop_conv);
    cudaEventSynchronize(stop_conv);

    float time_conv = 0.0f;
    cudaEventElapsedTime(&time_conv, start_conv, stop_conv);
    printf("Convolution Time: %f ms\n", time_conv);

    // Mesurer le temps du sous-échantillonnage
    cudaEvent_t start_sub, stop_sub;
    cudaEventCreate(&start_sub);
    cudaEventCreate(&stop_sub);

    dim3 gridDimSub((S1_SIZE + blockDim.x - 1) / blockDim.x, (S1_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS);

    cudaEventRecord(start_sub);
    cudaSubsampling<<<gridDimSub, blockDim>>>(d_C1_data, d_S1_data, C1_SIZE, S1_SIZE);
    cudaEventRecord(stop_sub);
    cudaEventSynchronize(stop_sub);

    float time_sub = 0.0f;
    cudaEventElapsedTime(&time_sub, start_sub, stop_sub);
    printf("Subsampling Time: %f ms\n", time_sub);

    // Libération
    free(raw_data);
    free(C1_kernel);
    free(C1_data);
    free(S1_data);
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);

    return 0;
}
