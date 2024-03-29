문서정보 : 2023.03.31.~ 작성, 작성자 [@SAgiKPJH](https://github.com/SAgiKPJH)

<br>

# CUDA_CPP_DeepLearning
CUDA C++로 진행하는 DeepLearning의 모든 것!  

- Linux 환경에서 진행했습니다.

### 목표

- [x] 0. 기본 실험
  - [x] [CUDA_TIMMER 활용](https://github.com/SagiK-Repository/CUDA_CPP_DeepLearning/blob/main/0.%20%EA%B8%B0%EB%B3%B8%20%EC%8B%A4%ED%97%98/cuda_timer_test.cu)
  - [x] [CUDA Thread Test](https://github.com/SagiK-Repository/CUDA_CPP_DeepLearning/blob/main/0.%20%EA%B8%B0%EB%B3%B8%20%EC%8B%A4%ED%97%98/thread_num_test.cu)
  - [ ] [Set CUDA Device]
- [x] 1. CUDA C++ 친해지기
  - [x] [Mandelbrot Set 구현](https://github.com/SagiK-Repository/CUDA_CPP_DeepLearning/blob/main/1.%20CUDA%20C%2B%2B%20%EC%B9%9C%ED%95%B4%EC%A7%80%EA%B8%B0/Mandelbrot%20Set/mandel%20brot%20set.cu)
  - [x] [Mandelbrot Set PNG 구현](https://github.com/SagiK-Repository/CUDA_CPP_DeepLearning/blob/main/1.%20CUDA%20C%2B%2B%20%EC%B9%9C%ED%95%B4%EC%A7%80%EA%B8%B0/Mandelbrot%20Set/mandelbrot%20set%20PNG.cu)
  - [x] [Julia Set 구현](https://github.com/SagiK-Repository/CUDA_CPP_DeepLearning/blob/main/1.%20CUDA%20C%2B%2B%20%EC%B9%9C%ED%95%B4%EC%A7%80%EA%B8%B0/Julia%20Set/juliaset.cu)
- [ ] 2. Perceptron 기능 구성
  - [ ] AND
    - [x] : [AND Perceptron Only Weight 구성](https://github.com/SagiK-Repository/CUDA_CPP_DeepLearning/blob/main/2.%20Perceptron%20%EA%B8%B0%EB%8A%A5%20%EA%B5%AC%EC%84%B1/AND/AND0_Per.cu)
    - [x] : [AND Perceptron Only Weight + Bias 구성](https://github.com/SagiK-Repository/CUDA_CPP_DeepLearning/blob/main/2.%20Perceptron%20%EA%B8%B0%EB%8A%A5%20%EA%B5%AC%EC%84%B1/AND/AND1_Per_plus_bias)
    - [x] : [AND Perceptron Only Weight + Bias + HiddenLayer 구성](https://github.com/SagiK-Repository/CUDA_CPP_DeepLearning/blob/main/2.%20Perceptron%20%EA%B8%B0%EB%8A%A5%20%EA%B5%AC%EC%84%B1/AND/AND2_PerPlus_Hidden)
  - [ ] : Perceptron Model 코드 구성
    - [ ] : Custom Forward Nomal Model
    - [ ] : Custom Forward Input Model
    - [ ] : Custom Forward Output Model
    - [ ] : Custom Forward Hiddem Model
    - [ ] : Custom Backward Model


### 제작자
[@SAgiKPJH](https://github.com/SAgiKPJH)

<br>

---

<br>

```mermaid
flowchart TB
  subgraph TOP1["1. CUDA C++ 친해지기"]
      A00["Mandelbrot Set"]--->A01["Mandelbrot Set PNG"]--->A02["Julia Set"]
  end
  subgraph TOP2["2. Perceptron 기능 구성"]
      A10["AND Weight"]--->A11["AND Weight + Bias"]--->A12["AND HiddenLayer"]
  end
TOP1--->TOP2
```

## 내용

