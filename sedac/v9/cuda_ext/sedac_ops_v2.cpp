// sedac_ops_v2.cpp
// SEDAC V9.0 - PyTorch C++ Bindings for Optimized Kernels

#include <torch/extension.h>
#include <vector>

// Forward declarations
std::vector<torch::Tensor> fused_entropy_decision_v2_cuda(
    torch::Tensor logits,
    torch::Tensor hidden,
    torch::Tensor prev_hidden,
    float mean_entropy,
    float std_entropy,
    float layer_progress,
    float threshold);

std::vector<torch::Tensor> token_router_split_v2_cuda(
    torch::Tensor hidden,
    torch::Tensor decision_mask);

torch::Tensor token_router_merge_v2_cuda(
    torch::Tensor active_hidden,
    torch::Tensor active_indices,
    torch::Tensor exit_hidden,
    torch::Tensor exit_indices,
    int total_size);

std::vector<torch::Tensor> batched_entropy_decision_cuda(
    torch::Tensor logits,
    torch::Tensor hidden,
    torch::Tensor prev_hidden,
    float mean_entropy,
    float std_entropy,
    float layer_progress,
    float threshold);

// Input validation macros
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Wrapper with validation
std::vector<torch::Tensor> fused_entropy_decision_v2(
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
    
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [N, vocab]");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [N, hidden]");
    TORCH_CHECK(prev_hidden.dim() == 2, "prev_hidden must be 2D [N, hidden]");
    TORCH_CHECK(logits.size(0) == hidden.size(0), "batch size mismatch");
    TORCH_CHECK(hidden.size(0) == prev_hidden.size(0), "batch size mismatch");
    TORCH_CHECK(hidden.size(1) == prev_hidden.size(1), "hidden size mismatch");
    
    return fused_entropy_decision_v2_cuda(
        logits, hidden, prev_hidden,
        mean_entropy, std_entropy, layer_progress, threshold);
}

std::vector<torch::Tensor> token_router_split_v2(
    torch::Tensor hidden,
    torch::Tensor decision_mask) {
    
    CHECK_INPUT(hidden);
    CHECK_CUDA(decision_mask);
    
    TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [N, hidden]");
    TORCH_CHECK(decision_mask.dim() == 1, "decision_mask must be 1D [N]");
    TORCH_CHECK(hidden.size(0) == decision_mask.size(0), "size mismatch");
    
    return token_router_split_v2_cuda(hidden, decision_mask);
}

torch::Tensor token_router_merge_v2(
    torch::Tensor active_hidden,
    torch::Tensor active_indices,
    torch::Tensor exit_hidden,
    torch::Tensor exit_indices,
    int total_size) {
    
    CHECK_INPUT(active_hidden);
    CHECK_INPUT(exit_hidden);
    CHECK_CUDA(active_indices);
    CHECK_CUDA(exit_indices);
    
    return token_router_merge_v2_cuda(
        active_hidden, active_indices,
        exit_hidden, exit_indices, total_size);
}

std::vector<torch::Tensor> batched_entropy_decision(
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
    
    TORCH_CHECK(logits.dim() == 3, "logits must be 3D [batch, seq, vocab]");
    TORCH_CHECK(hidden.dim() == 3, "hidden must be 3D [batch, seq, hidden]");
    
    return batched_entropy_decision_cuda(
        logits, hidden, prev_hidden,
        mean_entropy, std_entropy, layer_progress, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SEDAC V9.0 High-Performance CUDA Kernels";
    
    m.def("fused_entropy_decision_v2", &fused_entropy_decision_v2,
          "Fused Entropy Decision V2 (Warp Shuffle + Vectorized)",
          py::arg("logits"),
          py::arg("hidden"),
          py::arg("prev_hidden"),
          py::arg("mean_entropy"),
          py::arg("std_entropy"),
          py::arg("layer_progress"),
          py::arg("threshold"));
    
    m.def("token_router_split_v2", &token_router_split_v2,
          "Token Router Split V2 (Prefix Sum)",
          py::arg("hidden"),
          py::arg("decision_mask"));
    
    m.def("token_router_merge_v2", &token_router_merge_v2,
          "Token Router Merge V2 (Vectorized)",
          py::arg("active_hidden"),
          py::arg("active_indices"),
          py::arg("exit_hidden"),
          py::arg("exit_indices"),
          py::arg("total_size"));
    
    m.def("batched_entropy_decision", &batched_entropy_decision,
          "Batched Entropy Decision for [batch, seq, ...] inputs",
          py::arg("logits"),
          py::arg("hidden"),
          py::arg("prev_hidden"),
          py::arg("mean_entropy"),
          py::arg("std_entropy"),
          py::arg("layer_progress"),
          py::arg("threshold"));
}
