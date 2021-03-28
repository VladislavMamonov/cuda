#include <iostream>


#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 10)


using namespace std;



void memcpy()
{
    int *dev;
    int *host = new int[FULL_DATA_SIZE];

    cudaMalloc((void**)&dev, FULL_DATA_SIZE * sizeof(int));

    for (int i = 0; i < FULL_DATA_SIZE; i++)
        host[i] = rand();

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(dev, host, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "host->device elapsed time: " << elapsedTime << " ms" << endl;

    cudaEventRecord(start, 0);
    cudaMemcpy(host, dev, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "device->host elapsed time: " << elapsedTime << " ms" << endl;
}



void memcpy_PageLocked()
{
    int *dev;
    int *host;

    cudaMalloc((void**)&dev, FULL_DATA_SIZE * sizeof(int));
    cudaHostAlloc((void**)&host, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++)
        host[i] = rand();

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(dev, host, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "page-locked host->device elapsed time: " << elapsedTime << " ms" << endl;

    cudaEventRecord(start, 0);
    cudaMemcpy(host, dev, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "page-locked device->host elapsed time: " << elapsedTime << " ms" << endl;
}



int main()
{
    cudaDeviceProp prop;
    int whichDevice;

    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        cout << "Device does not support overlapping" << endl;
        return 1;
    }

    memcpy();
    cout << endl;
    memcpy_PageLocked();

    return 0;
}
