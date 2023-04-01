// nvcc -o "Nomal/thread_num_test" "Nomal/thread_num_test.cu" -lpng --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// "./Nomal/thread_num_test"

#include <cuda_runtime.h>
#include <iostream>

__global__ void test_kernel() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("ID : %3d  blockIdx.x : %3d   blockDim.x : %3d, threadIdx.x; : %3d --- blockIdx.x * blockDim.x : %3d \n", id, blockIdx.x, blockDim.x, threadIdx.x, blockIdx.x * blockDim.x );
}

int main() {

    std::cout << "test_kernel 4 4" << std::endl;
    test_kernel<<<4, 4>>>();
    cudaDeviceSynchronize();

    return 0;

}
