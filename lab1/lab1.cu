#include <iostream>
#include <sys/time.h>
#include <cuda.h>

using namespace std;

int size = 8000000;
int threads_per_block = 64;


__global__ void VecSum(float *A, float *B, float *C)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    C[i] = A[i] + B[i];
}


void printVec(float *C)
{
    for (int i = 0; i < size; i++)
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

    double time = wtime();
    VecSum <<< blockTotal, threads_per_block >>> (dev_A, dev_B, dev_C);
    cudaDeviceSynchronize();
    time = wtime() - time;

    cudaMemcpy(C, dev_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "time: " << time << endl;
    //printVec(C);

    delete [] A; delete [] B; delete [] C;
    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);

    return 0;
}
