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


__global__ void InitMatrix(float *A, int threads_per_block_x, int blocks_X, int n)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ia = n * (threads_per_block_x * by + ty);    // Номер строки из A
    int ib = threads_per_block_x * bx + tx;          // Номер столбца из B
    int ic = ia + ib;                       // Номер элемента из C

    int i = (tx * threads_per_block_x + ty) + (bx * (blocks_X) + by) * blockDim.x * blockDim.y;

    A[i] = i;
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

    float *dev_A;
    cudaMalloc((void**)&dev_A, size * size * sizeof(float));

    dim3 threads(threads_per_block_x, threads_per_block_y);
    dim3 blocks(size / threads.x, size / threads.y);
    int blocks_X = size / threads.x;

    cudaMemcpy(dev_A, A, size * size * sizeof(float), cudaMemcpyHostToDevice);

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    InitMatrix <<< blocks, threads >>> (dev_A, threads_per_block_x, blocks_X, size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(A, dev_A, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "time: " << elapsedTime << " ms" << endl;
    printMatrix(A, size);

    delete [] A;
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dev_A);

    return 0;
}
