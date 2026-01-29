"""
SEDAC V9.0 - Build Script for Optimized CUDA Kernels
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# CUDA compute capability for RTX 4060 (Ada Lovelace)
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')

setup(
    name='sedac_cuda_v2',
    version='2.0.0',
    description='SEDAC V9.0 High-Performance CUDA Kernels',
    ext_modules=[
        CUDAExtension(
            name='sedac_cuda_v2',
            sources=[
                'sedac_ops_v2.cpp',
                'sedac_kernels_v2.cu',
            ],
            extra_compile_args={
                'cxx': ['/O2', '/std:c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xcompiler', '/O2',
                    '--expt-relaxed-constexpr',
                    '-allow-unsupported-compiler',
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4060
                    '-lineinfo',  # For profiling
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
