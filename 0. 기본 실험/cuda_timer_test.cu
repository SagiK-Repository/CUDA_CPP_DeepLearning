// nvcc -o "Nomal/cuda_timer_test" "Nomal/cuda_timer_test.cu" -lpng --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// "./Nomal/cuda_timer_test"

#include <cuda_runtime.h>
#include <iostream>
#include <functional> // std::function

void cuda_timmer(const std::string& msg, std::function<void()> f) {
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    f();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << msg << " Elapsed time: " << elapsedTime << " ms" << std::endl;
}

int main() {

    // Nomal

    cudaEvent_t start0, stop0;
    float elapsedTime;

    cudaEventCreate(&start0);
    cudaEventCreate(&stop0);
    cudaEventRecord(start0, 0);

    // 실행할 코드

    cudaEventRecord(stop0, 0);
    cudaEventSynchronize(stop0);
    cudaEventElapsedTime(&elapsedTime, start0, stop0);

    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;


    // Lamda function
    
    cuda_timmer("Execution time", []() {
        // 실행할 코드
    });

    return 0;

}
