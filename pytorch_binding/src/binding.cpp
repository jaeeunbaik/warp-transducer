#include <iostream>
#include <numeric>

#include <torch/extension.h>
#include "rnnt.h"

#ifdef WARPRNNT_ENABLE_GPU
    // #include "THC.h" // 이 줄은 주석 처리 또는 삭제 유지
    #include <ATen/cuda/CUDAContext.h> // PyTorch의 CUDA 컨텍스트용
    #include <c10/cuda/CUDAStream.h>    // c10::cuda::CUDAStream용

    // THCState* state 변수는 더 이상 필요하지 않습니다. 이 줄은 완전히 제거되어야 합니다.
    // 이전 오류에서 이 줄이 남아있어 'THCState' does not name a type 오류가 발생했습니다.
    // extern THCState* state; // <-- 이 줄은 반드시 삭제되어야 합니다.
#endif

int cpu_rnnt(torch::Tensor acts,
             torch::Tensor labels,
             torch::Tensor input_lengths,
             torch::Tensor label_lengths,
             torch::Tensor costs,
             torch::Tensor grads,
             int blank_label,
             int num_threads) {

    int maxT = acts.size(0);
    int maxU = acts.size(1);
    int minibatch_size = acts.size(2);
    int alphabet_size = acts.size(3);

    // 이 조건문은 항상 true이므로 사실상 불필요하지만, 원본 코드에 있었으므로 유지합니다.
    if (true) {
        minibatch_size = acts.size(0);
        maxT = acts.size(1);
        maxU = acts.size(2);
    }

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.batch_first = true;
    options.loc = RNNT_CPU;
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes = 0;
    switch (acts.type().scalarType()) {
      case torch::ScalarType::Float:
        {
        get_workspace_size(maxT, maxU, minibatch_size,
                             false, &cpu_size_bytes);

        float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];
        compute_rnnt_loss(acts.data<float>(), grads.data<float>(),
                          labels.data<int>(), label_lengths.data<int>(),
                          input_lengths.data<int>(), alphabet_size,
                          minibatch_size, costs.data<float>(),
                          cpu_workspace, options);

        delete[] cpu_workspace; // new unsigned char[] 이므로 delete[] 사용
        return 0;
        }
      case torch::ScalarType::Double:
        {
        get_workspace_size(maxT, maxU, minibatch_size,
                             false, &cpu_size_bytes,
                             sizeof(double));

        double* cpu_workspace = (double*) new unsigned char[cpu_size_bytes];
        compute_rnnt_loss_fp64(acts.data<double>(), grads.data<double>(),
                               labels.data<int>(), label_lengths.data<int>(),
                               input_lengths.data<int>(), alphabet_size,
                               minibatch_size, costs.data<double>(),
                               cpu_workspace, options);

        delete[] cpu_workspace; // new unsigned char[] 이므로 delete[] 사용
        return 0;
        }
      default:
        std::cerr << __FILE__ << ':' << __LINE__ << ": " << "unsupported data type" << std::endl;
    }
    return -1;
}

#ifdef WARPRNNT_ENABLE_GPU
int gpu_rnnt(torch::Tensor acts,
             torch::Tensor labels,
             torch::Tensor input_lengths,
             torch::Tensor label_lengths,
             torch::Tensor costs,
             torch::Tensor grads,
             int blank_label,
             int num_threads) {

    int minibatch_size = acts.size(0);
    int maxT = acts.size(1);
    int maxU = acts.size(2);
    int alphabet_size = acts.size(3);

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.loc = RNNT_GPU;
    options.stream = at::cuda::getCurrentCUDAStream(); // PyTorch의 현재 CUDA 스트림 사용
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    switch (acts.type().scalarType()) {
      case torch::ScalarType::Float:
        {
        size_t gpu_size_bytes;
        get_workspace_size(maxT, maxU, minibatch_size,
                             true, &gpu_size_bytes);

        cudaSetDevice(acts.get_device());

        void* gpu_workspace;
        // cudaMalloc 함수는 void** 타입의 포인터 주소를 첫 인자로 받습니다.
        // 이전 오류에서 'statx'를 사용한 잘못된 호출이 있었습니다.
        cudaError_t err = cudaMalloc(&gpu_workspace, gpu_size_bytes);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        compute_rnnt_loss(acts.data<float>(), grads.data<float>(),
                          labels.data<int>(), label_lengths.data<int>(),
                          input_lengths.data<int>(), alphabet_size,
                          minibatch_size, costs.data<float>(),
                          gpu_workspace, options);

        // cudaFree 함수는 하나의 void* 인자만 받습니다.
        // 이전 오류에서 'statx'를 사용한 잘못된 호출이 있었습니다.
        err = cudaFree(gpu_workspace);
        if (err != cudaSuccess) {
            std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        return 0;
        }
      case torch::ScalarType::Double:
        {
        size_t gpu_size_bytes;
        get_workspace_size(maxT, maxU, minibatch_size,
                             true, &gpu_size_bytes);

        cudaSetDevice(acts.get_device());

        void* gpu_workspace;
        // cudaMalloc 함수 호출 방식 수정
        cudaError_t err = cudaMalloc(&gpu_workspace, gpu_size_bytes);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        compute_rnnt_loss_fp64(acts.data<double>(), grads.data<double>(),
                               labels.data<int>(), label_lengths.data<int>(),
                               input_lengths.data<int>(), alphabet_size,
                               minibatch_size, costs.data<double>(),
                               gpu_workspace, options);

        // cudaFree 함수 호출 방식 수정
        err = cudaFree(gpu_workspace);
        if (err != cudaSuccess) {
            std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        return 0;
        }
      default:
        std::cerr << __FILE__ << ':' << __LINE__ << ": " << "unsupported data type" << std::endl;
    }
    return -1;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_rnnt", &cpu_rnnt, "RNNT CPU version");
#ifdef WARPRNNT_ENABLE_GPU
    m.def("gpu_rnnt", &gpu_rnnt, "RNNT GPU version");
#endif
}
