#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n * p; i++) {
        Mout[i] = M1[i] + M2[i];
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%f ", M[i * p + j]);
        }
        printf("\n");
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += M1[i * n + k] * M2[k * n + j];
            }
            Mout[i * n + j] = sum;
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0;
        for (int k = 0; k < n; k++) {
            value += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = value;
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        int index = row * p + col;
        Mout[index] = M1[index] + M2[index];
    }
}

int main() {
    int n = 100;
    float *M1 = (float*)malloc(n * n * sizeof(float));
    float *M2 = (float*)malloc(n * n * sizeof(float));
    float *Mout = (float*)malloc(n * n * sizeof(float));

    MatrixInit(M1, n, n);
    MatrixInit(M2, n, n);

    // Measure CPU addition time
    clock_t start_cpu_add = clock();
    MatrixAdd(M1, M2, Mout, n, n);
    clock_t end_cpu_add = clock();
    double cpu_add_time = ((double)(end_cpu_add - start_cpu_add)) / CLOCKS_PER_SEC;
    printf("CPU Matrix Addition Time: %f milliseconds\n", cpu_add_time * 1000);

    // Allocate device memory for addition
    float *d_M1_add, *d_M2_add, *d_Mout_add;
    cudaMalloc((void**)&d_M1_add, n * n * sizeof(float));
    cudaMalloc((void**)&d_M2_add, n * n * sizeof(float));
    cudaMalloc((void**)&d_Mout_add, n * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_M1_add, M1, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2_add, M2, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Measure GPU addition time
    cudaEvent_t start_gpu_add, stop_gpu_add;
    cudaEventCreate(&start_gpu_add);
    cudaEventCreate(&stop_gpu_add);
    cudaEventRecord(start_gpu_add);

    dim3 blockDimAdd(16, 16);
    dim3 gridDimAdd((n + blockDimAdd.x - 1) / blockDimAdd.x, (n + blockDimAdd.y - 1) / blockDimAdd.y);
    cudaMatrixAdd<<<gridDimAdd, blockDimAdd>>>(d_M1_add, d_M2_add, d_Mout_add, n, n);

    cudaEventRecord(stop_gpu_add);
    cudaEventSynchronize(stop_gpu_add);
    float gpu_add_time = 0;
    cudaEventElapsedTime(&gpu_add_time, start_gpu_add, stop_gpu_add);
    printf("GPU Matrix Addition Time: %f milliseconds\n", gpu_add_time);

    // Copy result back to host
    cudaMemcpy(Mout, d_Mout_add, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory for addition
    cudaFree(d_M1_add);
    cudaFree(d_M2_add);
    cudaFree(d_Mout_add);

    // Measure CPU multiplication time
    clock_t start_cpu_mult = clock();
    MatrixMult(M1, M2, Mout, n);
    clock_t end_cpu_mult = clock();
    double cpu_mult_time = ((double)(end_cpu_mult - start_cpu_mult)) / CLOCKS_PER_SEC;
    printf("CPU Matrix Multiplication Time: %f milliseconds\n", cpu_mult_time * 1000);

    // Allocate device memory for multiplication
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void**)&d_M1, n * n * sizeof(float));
    cudaMalloc((void**)&d_M2, n * n * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_M1, M1, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Measure GPU multiplication time
    cudaEvent_t start_gpu_mult, stop_gpu_mult;
    cudaEventCreate(&start_gpu_mult);
    cudaEventCreate(&stop_gpu_mult);
    cudaEventRecord(start_gpu_mult);

    dim3 blockDimMult(16, 16);
    dim3 gridDimMult((n + blockDimMult.x - 1) / blockDimMult.x, (n + blockDimMult.y - 1) / blockDimMult.y);
    cudaMatrixMult<<<gridDimMult, blockDimMult>>>(d_M1, d_M2, d_Mout, n);

    cudaEventRecord(stop_gpu_mult);
    cudaEventSynchronize(stop_gpu_mult);
    float gpu_mult_time = 0;
    cudaEventElapsedTime(&gpu_mult_time, start_gpu_mult, stop_gpu_mult);
    printf("GPU Matrix Multiplication Time: %f milliseconds\n", gpu_mult_time);

    // Copy result back to host and free device memory
    cudaMemcpy(Mout, d_Mout, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    free(M1);
    free(M2);
    free(Mout);

    return 0;
}
