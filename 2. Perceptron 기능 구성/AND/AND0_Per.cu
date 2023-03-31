// AND 학습 - Only Weight
// nvcc -o "DeepLearning/AND/AND0_Per" "DeepLearning/AND/AND0_Per.cu" -lpng --expt-relaxed-constexpr -lcurand -lcuda -lcudart -lcublas
// "./DeepLearning/AND/AND0_Per"

// Using CUDA library : -lcuda -lcudart -lcublas

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

const int INPUT_SIZE = 2;
const int OUTPUT_SIZE = 1;
const int BATCH_SIZE = 4;

__global__ void and_kernel(float* inputs, float* weights, float* output)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int index = threadIdx.x;
    float sum = inputs[id * INPUT_SIZE] * weights[0] + inputs[id * INPUT_SIZE + 1] * weights[1];
    output[index] = sum;
}

int main()
{
    // 입력 값과 가중치 초기화
    float inputs[BATCH_SIZE][INPUT_SIZE] = { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} };
    float weights[2] = { 0.5f, 0.5f };

    // 출력 값을 저장할 배열 초기화
    float output[BATCH_SIZE] = {};

    // CUDA 메모리 할당
    float *d_input, *d_output, *d_weights;
    cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_weights, INPUT_SIZE * sizeof(float));

    // 입력 데이터를 GPU로 복사
    cudaMemcpy(d_input, inputs, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA 커널 함수 호출
    and_kernel<<<1, 4>>>(d_input, d_weights, d_output);

    // 출력 데이터를 호스트로 복사
    cudaMemcpy(output, d_output, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // 결과 출력
    for (int i = 0; i < 4; i++) {
        std::cout << "AND(" << inputs[i][0] << ", " << inputs[i][1] << ") = " << output[i] << std::endl;
    }

    return 0;
}
