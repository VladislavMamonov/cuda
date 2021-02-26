#include <iostream>
#include <sys/time.h>
#include <cuda.h>

using namespace std;


#define CUDA_CHECK_RETURN(value) {\
    cudaError_t _m_cudaStat = value;\
    if (_m_cudaStat != cudaSuccess) {\
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);\
    }}


__global__ void VecSum(float *A, float *B, float *C, int size)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
        C[i] = A[i] + B[i];
}


void printVec(float *C, int size)
{
    for (int i = 0; i < size; i++)
        cout << C[i] << "\t";
    cout << endl;
}


int main(int argc, char* argv[])
{
    if (argc != 3) {
        cout << "launch parametrs: [vector size] [threads per block]" << endl;
        return 1;
    }

    int size = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);

    srand(time(NULL));

    float *A = new float[size];
    float *B = new float[size];
    float *C = new float[size];

    float *dev_A, *dev_B, *dev_C;

    cudaMalloc((void**)&dev_A, size * sizeof(float));
    cudaMalloc((void**)&dev_B, size * sizeof(float));
    cudaMalloc((void**)&dev_C, size * sizeof(float));

    for (int i = 0; i < size; i++) {
        A[i] = rand();
        B[i] = rand();
    }

    cudaMemcpy(dev_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockTotal = ceilf(float(size) / float(threads_per_block));
    cout << "Block total: " << blockTotal << endl;
    cout << "Threads per block : " << threads_per_block << endl;
    cout << "Threads total: " << blockTotal * threads_per_block << endl;

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    VecSum <<< blockTotal, threads_per_block >>> (dev_A, dev_B, dev_C, size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(C, dev_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "time: " << elapsedTime << " ms" << endl;
    //printVec(C, size);

    delete [] A; delete [] B; delete [] C;
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);

    return 0;
}
