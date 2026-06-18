#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SMALL_SIZE 1000
#define LARGE_SIZE 1000
#define SMALL_THREADS 1000
#define LARGE_THREADS 1000
#define SMALL_ITERATIONS 10000
#define LARGE_ITERATIONS 10000

__global__ void kernel_1(float* data1, float* data2, float* data3, float* data4,
                         float* data5, float* data6, float* data7, float* data8) {
    int idx = threadIdx.x;
    for (int i = 0; i < SMALL_ITERATIONS; ++i) {
        data2[idx] = data1[idx];
        data4[idx] = data3[idx];
        data6[idx] = data5[idx];
        data8[idx] = data7[idx];
    }
}

__global__ void kernel_2(float* data1, float* data2, float* data3, float* data4,
                         float* data5, float* data6, float* data7, float* data8,
                         float* data9, float* data10, float* data11, float* data12) {
    int idx = threadIdx.x;
    for (int i = 0; i < LARGE_ITERATIONS; ++i) {
        data2[idx + (i*LARGE_ITERATIONS)] = data1[idx + (i*LARGE_ITERATIONS)];
        data4[idx + (i*LARGE_ITERATIONS)] = data3[idx + (i*LARGE_ITERATIONS)];
        data6[idx + (i*LARGE_ITERATIONS)] = data5[idx + (i*LARGE_ITERATIONS)];
        data8[idx + (i*LARGE_ITERATIONS)] = data7[idx + (i*LARGE_ITERATIONS)];
        data10[idx + (i*LARGE_ITERATIONS)] = data9[idx + (i*LARGE_ITERATIONS)];
        data12[idx + (i*LARGE_ITERATIONS)] = data11[idx + (i*LARGE_ITERATIONS)];
    }
}

int main() {
    // Host memory allocation for small and large kernel inputs (same as original)
    float *h_data1_1 = (float*)malloc(SMALL_SIZE  * sizeof(float));
    float *h_data2_1 = (float*)malloc(SMALL_SIZE  * sizeof(float));
    float *h_data3_1 = (float*)malloc(SMALL_SIZE  * sizeof(float));
    float *h_data4_1 = (float*)malloc(SMALL_SIZE  * sizeof(float));
    float *h_data5_1 = (float*)malloc(SMALL_SIZE  * sizeof(float));
    float *h_data6_1 = (float*)malloc(SMALL_SIZE  * sizeof(float));
    float *h_data7_1 = (float*)malloc(SMALL_SIZE  * sizeof(float));
    float *h_data8_1 = (float*)malloc(SMALL_SIZE  * sizeof(float));

    float *h_data1_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data2_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data3_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data4_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data5_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data6_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data7_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data8_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data9_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data10_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data11_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    float *h_data12_3 = (float*)malloc(LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));

    // Initialize host memory with random values
    for (int i = 0; i < SMALL_SIZE ; ++i) {
        h_data1_1[i] = (rand() % 100) + 1;
        h_data3_1[i] = (rand() % 100) + 1;
        h_data5_1[i] = (rand() % 100) + 1;
        h_data7_1[i] = (rand() % 100) + 1;
    }
    for (int i = 0; i < LARGE_SIZE * SMALL_ITERATIONS ; ++i) {
        h_data1_3[i] = (rand() % 100) + 1;
        h_data3_3[i] = (rand() % 100) + 1;
        h_data5_3[i] = (rand() % 100) + 1;
        h_data7_3[i] = (rand() % 100) + 1;
        h_data9_3[i] = (rand() % 100) + 1;
        h_data11_3[i] = (rand() % 100) + 1;
    }

    // Device memory allocation for kernel inputs
    float *d_data1_1, *d_data2_1, *d_data3_1, *d_data4_1, *d_data5_1, *d_data6_1, *d_data7_1, *d_data8_1;
    cudaMalloc(&d_data1_1, SMALL_SIZE  * sizeof(float));
    cudaMalloc(&d_data2_1, SMALL_SIZE  * sizeof(float));
    cudaMalloc(&d_data3_1, SMALL_SIZE  * sizeof(float));
    cudaMalloc(&d_data4_1, SMALL_SIZE  * sizeof(float));
    cudaMalloc(&d_data5_1, SMALL_SIZE  * sizeof(float));
    cudaMalloc(&d_data6_1, SMALL_SIZE  * sizeof(float));
    cudaMalloc(&d_data7_1, SMALL_SIZE  * sizeof(float));
    cudaMalloc(&d_data8_1, SMALL_SIZE  * sizeof(float));

    float *d_data1_3, *d_data2_3, *d_data3_3, *d_data4_3, *d_data5_3, *d_data6_3, *d_data7_3, *d_data8_3;
    float *d_data9_3, *d_data10_3, *d_data11_3, *d_data12_3;
    cudaMalloc(&d_data1_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data2_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data3_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data4_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data5_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data6_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data7_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data8_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data9_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data10_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data11_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));
    cudaMalloc(&d_data12_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float));

    // Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);

    // Copy data to device asynchronously using streams
    cudaMemcpyAsync(d_data1_1, h_data1_1, SMALL_SIZE  * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_data3_1, h_data3_1, SMALL_SIZE  * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_data5_1, h_data5_1, SMALL_SIZE  * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_data7_1, h_data7_1, SMALL_SIZE  * sizeof(float), cudaMemcpyHostToDevice, stream1);

    cudaMemcpyAsync(d_data1_3, h_data1_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_data3_3, h_data3_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_data5_3, h_data5_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_data7_3, h_data7_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_data9_3, h_data9_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_data11_3, h_data11_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyHostToDevice, stream2);

    // Launch kernels on separate streams
    

    kernel_1<<<1, SMALL_THREADS, 0, stream1>>>(d_data1_1, d_data2_1, d_data3_1, d_data4_1, 
                                               d_data5_1, d_data6_1, d_data7_1, d_data8_1);

    kernel_2<<<1, LARGE_THREADS, 0, stream2>>>(d_data1_3, d_data2_3, d_data3_3, d_data4_3, 
                                               d_data5_3, d_data6_3, d_data7_3, d_data8_3,
                                               d_data9_3, d_data10_3, d_data11_3, d_data12_3);

    kernel_2<<<1, LARGE_THREADS, 0, stream3>>>(d_data5_3, d_data2_3, d_data7_3, d_data4_3, 
                                               d_data1_3, d_data6_3, d_data3_3, d_data8_3,
                                               d_data11_3, d_data10_3, d_data9_3, d_data12_3);

    kernel_2<<<1, LARGE_THREADS, 0, stream4>>>(d_data5_3, d_data4_3, d_data7_3, d_data10_3, 
                                               d_data1_3, d_data12_3, d_data3_3, d_data8_3,
                                               d_data11_3, d_data2_3, d_data9_3, d_data6_3);

    kernel_2<<<1, LARGE_THREADS, 0, stream5>>>(d_data3_3, d_data4_3, d_data1_3, d_data10_3, 
                                               d_data7_3, d_data12_3, d_data5_3, d_data8_3,
                                               d_data9_3, d_data2_3, d_data11_3, d_data6_3);



    // kernel_2<<<1, LARGE_THREADS, 0, stream6>>>(d_data3_3, d_data10_3, d_data1_3, d_data4_3, 
    //                                            d_data7_3, d_data8_3, d_data5_3, d_data12_3,
    //                                            d_data9_3, d_data6_3, d_data11_3, d_data2_3);
    // Synchronize streams to ensure kernel completion before host memory access
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);
    cudaStreamSynchronize(stream5);
    cudaStreamSynchronize(stream6);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaStreamDestroy(stream5);
    cudaStreamDestroy(stream6);


	 // Copy results from device to host
    cudaMemcpy(h_data2_1, d_data2_1, SMALL_SIZE  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data4_1, d_data4_1, SMALL_SIZE  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data6_1, d_data6_1, SMALL_SIZE  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data8_1, d_data8_1, SMALL_SIZE  * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_data2_3, d_data2_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data4_3, d_data4_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data6_3, d_data6_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data8_3, d_data8_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data10_3, d_data10_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data12_3, d_data12_3, LARGE_SIZE * SMALL_ITERATIONS  * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data1_1);
    cudaFree(d_data2_1);
    cudaFree(d_data3_1);
    cudaFree(d_data4_1);
    cudaFree(d_data5_1);
    cudaFree(d_data6_1);
    cudaFree(d_data7_1);
    cudaFree(d_data8_1);
    
    cudaFree(d_data1_3);
    cudaFree(d_data2_3);
    cudaFree(d_data3_3);
    cudaFree(d_data4_3);
    cudaFree(d_data5_3);
    cudaFree(d_data6_3);
    cudaFree(d_data7_3);
    cudaFree(d_data8_3);
    cudaFree(d_data9_3);
    cudaFree(d_data10_3);
    cudaFree(d_data11_3);
    cudaFree(d_data12_3);

    // Free host memory
    free(h_data1_1);
    free(h_data2_1);
    free(h_data3_1);
    free(h_data4_1);
    free(h_data5_1);
    free(h_data6_1);
    free(h_data7_1);
    free(h_data8_1);

    free(h_data1_3);
    free(h_data2_3);
    free(h_data3_3);
    free(h_data4_3);
    free(h_data5_3);
    free(h_data6_3);
    free(h_data7_3);
    free(h_data8_3);
    free(h_data9_3);
    free(h_data10_3);
    free(h_data11_3);
    free(h_data12_3);

    return 0;
}
