// nvcc -o "Nomal/set_cuda_device" "Nomal/set_cuda_device.cu" -lpng --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// "./Nomal/set_cuda_device"

#include <cuda_runtime.h>
#include <iostream>

void setCudaDevice(int deviceNum) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceNum >= deviceCount || deviceNum < 0) {
        std::cout << "Invalid device number. There are " << deviceCount << " available CUDA devices." << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaSetDevice(deviceNum);
}

int main() {

    setCudaDevice(0);

    setCudaDevice(1);

    return 0;

}
