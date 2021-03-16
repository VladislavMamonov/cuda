#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

using namespace std;


#define CUDA_CHECK_RETURN(value) {\
    cudaError_t _m_cudaStat = value;\
    if (_m_cudaStat != cudaSuccess) {\
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);\
    }}


__global__ void transpose(float *A, float *B, int n)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = tx + bx * blockDim.x;
    int j = ty + by * blockDim.y;

    B[j * n + i] = A[i * n + j];
}


void InitMatrix(float *A, float *B, int size)
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) {
            int k = size * i + j;
            A[k] = k;
            B[k] = 0;
        }
}


void printMatrix(float *C, int size)
{
    for (int i = 0; i < size * size; i++)
        cout << C[i] << "\t";
    cout << endl;
}


double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);

    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}


int main(int argc, char* argv[])
{
    if (argc != 4) {
	    cout << "launch parametrs: [matrix size] [threads_x] [threads_y]" << endl;
        return 1;
    }

    int size = atoi(argv[1]);
    int threads_per_block_x = atoi(argv[2]);
    int threads_per_block_y = atoi(argv[3]);

    srand(time(NULL));

    float *A = new float[size * size];
    float *B = new float[size * size];

    float *dev_A, *dev_B;

    cudaMalloc((void**)&dev_A, size * size * sizeof(float));
    cudaMalloc((void**)&dev_B, size * size * sizeof(float));

    InitMatrix(A, B, size);

    dim3 threads(threads_per_block_x, threads_per_block_y);
    dim3 blocks(size / threads.x, size / threads.y);

    cudaMemcpy(dev_A, A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size * size * sizeof(float), cudaMemcpyHostToDevice);

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    transpose <<< blocks, threads >>> (dev_A, dev_B, size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(B, dev_B, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrix(B, size);
    cout << "time: " << elapsedTime << " ms" << endl;

    delete [] A; delete [] B;
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dev_A); cudaFree(dev_B);

    return 0;
}
