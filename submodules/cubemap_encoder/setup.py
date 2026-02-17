from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension # for cuda, cpp build


## '-03': 컴파일러 최적화
## -DCUDA_HAS_FP16=1: 소스코드 내에서 FP16 기능 활성화
## -D__CUDA_NO_HALF_OPERATORS__: CUDA가 기본으로 제공하는 half 자료형의 연산자와 변환 규칙 사용 X
## PyTorch 자체적으로 half 연산자 정의하고 있기 때문에, CUDA 표준 연산자와 충돌하는 경우 있음. 따라서 이를 막기 위해 CUDA 쪽 기능 끄는 것.
## -use_fast_math: 하드웨어 기반 수학 연산 수행. 엄격한 IEEE 754 부동소수점 표준 X.

nvcc_flags = [
    '-O3', '-std=c++17',
    #'-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
	'-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__',
    '-use_fast_math'
]

c_flags = ['-O3', '-std=c++17']

setup(
    ## package name
    name="cubemapencoder",
    ## 포함할 package 목록
    packages=["cubemapencoder"],
    ## extension module 정의 - compiile 대상 지정
    ext_modules=[
        CUDAExtension(
            ## Python에서 import할 이름
            name="_cubemapencoder",
            ## compile할 source 파일
            sources=[
            "src/cubemapencoder.cu",
            "src/ext.cpp"
            ],
            ## compile option
            extra_compile_args={"nvcc": nvcc_flags, 
                                "cxx": c_flags}
        )
    ],
    ## custom bulid dictionary
    cmdclass={
        'build_ext': BuildExtension
    }
)