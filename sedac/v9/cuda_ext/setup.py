# setup.py
# SEDAC V9.0 - CUDA Extension 编译脚本 (Windows 兼容)
"""
编译命令:
    cd sedac/v9/cuda_ext
    pip install .
    
或者:
    python setup.py install
    
依赖:
    - CUDA Toolkit (与PyTorch版本匹配)
    - Visual Studio 2019/2022 (带C++开发工具)
    - PyTorch with CUDA support
"""

import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Windows 特定配置
if sys.platform == 'win32':
    # MSVC 编译选项
    cxx_args = ['/O2', '/std:c++17']
    nvcc_args = [
        '-O3',
        '--use_fast_math',
        '-Xcompiler', '/O2',
        '--expt-relaxed-constexpr',
        '-allow-unsupported-compiler',  # 允许新版VS
    ]
else:
    # Linux/Mac 编译选项
    cxx_args = ['-O3', '-std=c++17']
    nvcc_args = [
        '-O3',
        '--use_fast_math',
        '-std=c++17',
    ]

setup(
    name='sedac_cuda',
    version='9.0.0',
    author='SEDAC Team',
    description='SEDAC V9.0 High Performance CUDA Kernels',
    long_description='''
    SEDAC V9.0 CUDA Extension
    
    提供以下高性能算子:
    1. fused_entropy_decision: 融合熵决策 (48ms → 0.2ms)
    2. token_router_split: Token路由分割 (零拷贝)
    3. token_router_merge: Token路由合并
    4. kv_cache_update: KV缓存更新 (Paged Attention风格)
    ''',
    ext_modules=[
        CUDAExtension(
            name='sedac_cuda',
            sources=[
                'sedac_ops.cpp',
                'sedac_kernels.cu',
            ],
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args,
            },
            # 如果需要额外的include路径
            # include_dirs=[...],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
    ],
)
