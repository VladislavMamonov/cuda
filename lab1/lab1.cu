#include <iostream>
#include <sys/time.h>
#include <cuda.h>

using namespace std;

#define SIZE 8000000
#define THREADS_PER_BLOCK 64


__global__ void VecSum(float *A, float *B, float *C)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    C[i] = A[i] + B[i];
}


void printVec(float *C)
{
    for (int i = 0; i < SIZE; i++)
        cout << C[i] << "\t";
    cout << endl;
}


double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);

    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}


int main()
{
    srand(time(NULL));

    float *A = new float[SIZE];
    float *B = new float[SIZE];
    float *C = new float[SIZE];

    float *dev_A, *dev_B, *dev_C;

    cudaMalloc((void**)&dev_A, SIZE * sizeof(float));
    cudaMalloc((void**)&dev_B, SIZE * sizeof(float));
    cudaMalloc((void**)&dev_C, SIZE * sizeof(float));

    for (int i = 0; i < SIZE; i++) {
        A[i] = rand();
        B[i] = rand();
    }

    cudaMemcpy(dev_A, A, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int blockTotal = SIZE / THREADS_PER_BLOCK;

    double time = wtime();
    VecSum <<< blockTotal, THREADS_PER_BLOCK >>> (dev_A, dev_B, dev_C);
    cudaDeviceSynchronize();
    time = wtime() - time;

    cudaMemcpy(C, dev_C, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "time: " << time << endl;
    //printVec(C);

    delete [] A; delete [] B; delete [] C;
    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);

    return 0;
}
