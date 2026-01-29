// sedac_ops.cpp
// SEDAC V9.0 - PyTorch C++ 绑定接口

#include <torch/extension.h>
#include <vector>

// Forward declarations from sedac_kernels.cu
std::vector<torch::Tensor> fused_entropy_decision_cuda(
    torch::Tensor logits,
    torch::Tensor hidden,
    torch::Tensor prev_hidden,
    float mean_entropy,
    float std_entropy,
    float layer_progress,
    float threshold);

std::vector<torch::Tensor> token_router_split_cuda(
    torch::Tensor hidden,
    torch::Tensor decision_mask);

torch::Tensor token_router_merge_cuda(
    torch::Tensor active_hidden,
    torch::Tensor active_indices,
    torch::Tensor exit_hidden,
    torch::Tensor exit_indices,
    int total_size);

void kv_cache_update_cuda(
    torch::Tensor hidden,
    torch::Tensor wk,
    torch::Tensor wv,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor positions);

// C++ 检查宏
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ==========================================
// 接口函数
// ==========================================

std::vector<torch::Tensor> fused_entropy_decision(
    torch::Tensor logits,
    torch::Tensor hidden,
    torch::Tensor prev_hidden,
    float mean_entropy,
    float std_entropy,
    float layer_progress,
    float threshold) {
    
    CHECK_INPUT(logits);
    CHECK_INPUT(hidden);
    CHECK_INPUT(prev_hidden);
    
    return fused_entropy_decision_cuda(
        logits, hidden, prev_hidden, 
        mean_entropy, std_entropy, 
        layer_progress, threshold
    );
}

std::vector<torch::Tensor> token_router_split(
    torch::Tensor hidden,
    torch::Tensor decision_mask) {
    
    CHECK_INPUT(hidden);
    CHECK_INPUT(decision_mask);
    
    return token_router_split_cuda(hidden, decision_mask);
}

torch::Tensor token_router_merge(
    torch::Tensor active_hidden,
    torch::Tensor active_indices,
    torch::Tensor exit_hidden,
    torch::Tensor exit_indices,
    int total_size) {
    
    CHECK_INPUT(active_hidden);
    CHECK_INPUT(active_indices);
    CHECK_INPUT(exit_hidden);
    CHECK_INPUT(exit_indices);
    
    return token_router_merge_cuda(
        active_hidden, active_indices,
        exit_hidden, exit_indices,
        total_size
    );
}

void kv_cache_update(
    torch::Tensor hidden,
    torch::Tensor wk,
    torch::Tensor wv,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor positions) {
    
    CHECK_INPUT(hidden);
    CHECK_INPUT(wk);
    CHECK_INPUT(wv);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(positions);
    
    kv_cache_update_cuda(hidden, wk, wv, k_cache, v_cache, positions);
}

// ==========================================
// Python 绑定
// ==========================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SEDAC V9.0 CUDA Kernels - High Performance Ops";
    
    m.def("fused_entropy_decision", &fused_entropy_decision, 
          "Fused Entropy Decision Kernel (CUDA)\n"
          "Returns: [entropy, confidence, decision, cognitive_load]",
          py::arg("logits"),
          py::arg("hidden"),
          py::arg("prev_hidden"),
          py::arg("mean_entropy"),
          py::arg("std_entropy"),
          py::arg("layer_progress"),
          py::arg("threshold"));
    
    m.def("token_router_split", &token_router_split,
          "Token Router Split Kernel (CUDA)\n"
          "Returns: [active_hidden, active_indices, exit_hidden, exit_indices]",
          py::arg("hidden"),
          py::arg("decision_mask"));
    
    m.def("token_router_merge", &token_router_merge,
          "Token Router Merge Kernel (CUDA)\n"
          "Returns: merged hidden states",
          py::arg("active_hidden"),
          py::arg("active_indices"),
          py::arg("exit_hidden"),
          py::arg("exit_indices"),
          py::arg("total_size"));
    
    m.def("kv_cache_update", &kv_cache_update,
          "KV Cache Update Kernel (CUDA)\n"
          "Updates KV cache in-place with projected values",
          py::arg("hidden"),
          py::arg("wk"),
          py::arg("wv"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("positions"));
}
