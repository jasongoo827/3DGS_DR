#include <torch/extension.h>
#include <cubemapencoder.h>

// TORCH_EXTENSION_NAME: setup.py의 name 인자 또는
// Extension 객체 생성 시 지정한 name을 그대로 가져옴.
// PyTorch의 cpp_extension.py를 통해 가져옴.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Python Function name, CUDA Function name, Function Description
    m.def("cubemap_encode_forward", &cubemap_encode_forward, "cubemap encode forward (CUDA)");
    m.def("cubemap_encode_backward", &cubemap_encode_backward, "cubemap encode backward (CUDA)");
}