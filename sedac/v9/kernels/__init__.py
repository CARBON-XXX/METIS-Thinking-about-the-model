"""
SEDAC V9.0 - 高性能CUDA/Rust内核

包含:
- fused_entropy_kernel: 融合熵决策内核
- kv_cache_kernel: KV缓存更新内核  
- token_router_kernel: Token路由内核
"""

from sedac.v9.kernels.cuda_ops import (
    fused_entropy_decision,
    kv_cache_update,
    token_router_split,
    token_router_merge,
    CUDA_AVAILABLE,
)

__all__ = [
    "fused_entropy_decision",
    "kv_cache_update", 
    "token_router_split",
    "token_router_merge",
    "CUDA_AVAILABLE",
]
