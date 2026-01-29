// sedac_kernels_v2.cu
// SEDAC V9.0 - High Performance CUDA Kernels
// Optimizations: Warp Shuffle, Fused Passes, Vectorized Load, Dynamic Block Size
// Target: <1ms for 4K tokens

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Warp-level primitives
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Block sizes for different workloads
#define BLOCK_SMALL 128   // For seq_len < 256
#define BLOCK_MEDIUM 256  // For seq_len < 1024
#define BLOCK_LARGE 512   // For seq_len < 4096
#define BLOCK_XLARGE 1024 // For seq_len >= 4096

// ============================================================================
// Warp-level Reduction Primitives (No Shared Memory)
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Block-level reduction using warp shuffle (minimal shared memory)
__device__ __forceinline__ float block_reduce_max(float val, float* shared, int tid, int block_size) {
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (tid < num_warps) ? shared[tid] : -1e20f;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared, int tid, int block_size) {
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (tid < num_warps) ? shared[tid] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// ============================================================================
// Kernel 1: Ultra-Fast Fused Entropy Decision (Single-Pass)
// ============================================================================
// Optimization: 2-pass instead of 3-pass, vectorized loads, warp shuffle

template <int BLOCK_SIZE>
__global__ void fused_entropy_decision_kernel_v2(
    const float* __restrict__ logits,
    const float* __restrict__ hidden,
    const float* __restrict__ prev_hidden,
    float* __restrict__ entropy_out,
    float* __restrict__ confidence_out,
    bool* __restrict__ decision_out,
    float* __restrict__ load_out,
    const int vocab_size,
    const int hidden_size,
    const float mean_entropy,
    const float std_entropy,
    const float layer_progress,
    const float threshold_base) {

    const int idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // Minimal shared memory: only for cross-warp communication
    __shared__ float warp_data[32];  // Max 32 warps
    
    const float* logits_row = logits + idx * vocab_size;
    const float* hidden_row = hidden + idx * hidden_size;
    const float* prev_row = prev_hidden + idx * hidden_size;
    
    // ========== Pass 1: Find Max (Vectorized) ==========
    float local_max = -1e20f;
    
    // Process 4 elements per iteration when possible
    int vec_end = (vocab_size / 4) * 4;
    for (int i = tid * 4; i < vec_end; i += BLOCK_SIZE * 4) {
        float4 vals = *reinterpret_cast<const float4*>(logits_row + i);
        local_max = fmaxf(local_max, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
    }
    // Handle remainder
    for (int i = vec_end + tid; i < vocab_size; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, logits_row[i]);
    }
    
    float global_max = block_reduce_max(local_max, warp_data, tid, BLOCK_SIZE);
    __syncthreads();
    global_max = __shfl_sync(FULL_MASK, global_max, 0);
    if (tid >= WARP_SIZE) global_max = warp_data[0];
    __syncthreads();
    if (tid == 0) warp_data[0] = global_max;
    __syncthreads();
    global_max = warp_data[0];
    
    // ========== Pass 2: Fused Sum_Exp + Entropy (Single Pass) ==========
    float local_sum_exp = 0.0f;
    float local_sum_p_logp = 0.0f;
    
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = logits_row[i];
        float shifted = val - global_max;
        float exp_val = expf(shifted);
        local_sum_exp += exp_val;
        local_sum_p_logp += exp_val * shifted;  // p * (logit - max) = p * log(p/Z)
    }
    
    float sum_exp = block_reduce_sum(local_sum_exp, warp_data, tid, BLOCK_SIZE);
    __syncthreads();
    if (tid == 0) warp_data[0] = sum_exp;
    __syncthreads();
    sum_exp = warp_data[0];
    
    float sum_p_logp = block_reduce_sum(local_sum_p_logp, warp_data, tid, BLOCK_SIZE);
    __syncthreads();
    if (tid == 0) warp_data[0] = sum_p_logp;
    __syncthreads();
    sum_p_logp = warp_data[0];
    
    // Entropy = -sum(p * log(p)) = log(Z) - sum(p * shifted) / Z
    float log_z = logf(sum_exp);
    float entropy = (log_z - sum_p_logp / sum_exp) / 0.69314718f;  // Convert to bits
    
    // ========== Stability Computation (Vectorized) ==========
    float local_diff_sq = 0.0f;
    float local_norm_sq = 0.0f;
    
    vec_end = (hidden_size / 4) * 4;
    for (int i = tid * 4; i < vec_end; i += BLOCK_SIZE * 4) {
        float4 h = *reinterpret_cast<const float4*>(hidden_row + i);
        float4 ph = *reinterpret_cast<const float4*>(prev_row + i);
        
        float dx = h.x - ph.x, dy = h.y - ph.y, dz = h.z - ph.z, dw = h.w - ph.w;
        local_diff_sq += dx*dx + dy*dy + dz*dz + dw*dw;
        local_norm_sq += h.x*h.x + h.y*h.y + h.z*h.z + h.w*h.w;
    }
    for (int i = vec_end + tid; i < hidden_size; i += BLOCK_SIZE) {
        float h = hidden_row[i];
        float ph = prev_row[i];
        float d = h - ph;
        local_diff_sq += d * d;
        local_norm_sq += h * h;
    }
    
    float diff_sq = block_reduce_sum(local_diff_sq, warp_data, tid, BLOCK_SIZE);
    __syncthreads();
    float norm_sq = block_reduce_sum(local_norm_sq, warp_data, tid, BLOCK_SIZE);
    
    // ========== Final Decision (Single Thread) ==========
    if (tid == 0) {
        float stability = 1.0f / (1.0f + sqrtf(diff_sq) / (sqrtf(norm_sq) + 1e-6f));
        float z_score = (mean_entropy - entropy) / (std_entropy + 1e-6f);
        float confidence = 1.0f / (1.0f + expf(-z_score * 2.0f));
        float cognitive_load = (1.0f - confidence) * 0.5f + (1.0f - stability) * 0.3f + (1.0f - layer_progress) * 0.2f;
        float current_thresh = threshold_base - layer_progress * 0.2f;
        bool should_exit = (confidence * stability * layer_progress) > current_thresh;

        entropy_out[idx] = entropy;
        confidence_out[idx] = confidence;
        load_out[idx] = cognitive_load;
        decision_out[idx] = should_exit;
    }
}

// Half precision version for even faster computation
template <int BLOCK_SIZE>
__global__ void fused_entropy_decision_kernel_fp16(
    const __half* __restrict__ logits,
    const __half* __restrict__ hidden,
    const __half* __restrict__ prev_hidden,
    float* __restrict__ entropy_out,
    float* __restrict__ confidence_out,
    bool* __restrict__ decision_out,
    float* __restrict__ load_out,
    const int vocab_size,
    const int hidden_size,
    const float mean_entropy,
    const float std_entropy,
    const float layer_progress,
    const float threshold_base) {

    const int idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    __shared__ float warp_data[32];
    
    const __half* logits_row = logits + idx * vocab_size;
    const __half* hidden_row = hidden + idx * hidden_size;
    const __half* prev_row = prev_hidden + idx * hidden_size;
    
    // Pass 1: Find Max
    float local_max = -1e20f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, __half2float(logits_row[i]));
    }
    
    float global_max = block_reduce_max(local_max, warp_data, tid, BLOCK_SIZE);
    __syncthreads();
    if (tid == 0) warp_data[0] = global_max;
    __syncthreads();
    global_max = warp_data[0];
    
    // Pass 2: Fused Sum_Exp + Entropy
    float local_sum_exp = 0.0f;
    float local_sum_p_logp = 0.0f;
    
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = __half2float(logits_row[i]);
        float shifted = val - global_max;
        float exp_val = expf(shifted);
        local_sum_exp += exp_val;
        local_sum_p_logp += exp_val * shifted;
    }
    
    float sum_exp = block_reduce_sum(local_sum_exp, warp_data, tid, BLOCK_SIZE);
    __syncthreads();
    if (tid == 0) warp_data[0] = sum_exp;
    __syncthreads();
    sum_exp = warp_data[0];
    
    float sum_p_logp = block_reduce_sum(local_sum_p_logp, warp_data, tid, BLOCK_SIZE);
    __syncthreads();
    if (tid == 0) warp_data[0] = sum_p_logp;
    __syncthreads();
    sum_p_logp = warp_data[0];
    
    float log_z = logf(sum_exp);
    float entropy = (log_z - sum_p_logp / sum_exp) / 0.69314718f;
    
    // Stability
    float local_diff_sq = 0.0f;
    float local_norm_sq = 0.0f;
    
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float h = __half2float(hidden_row[i]);
        float ph = __half2float(prev_row[i]);
        float d = h - ph;
        local_diff_sq += d * d;
        local_norm_sq += h * h;
    }
    
    float diff_sq = block_reduce_sum(local_diff_sq, warp_data, tid, BLOCK_SIZE);
    __syncthreads();
    float norm_sq = block_reduce_sum(local_norm_sq, warp_data, tid, BLOCK_SIZE);
    
    if (tid == 0) {
        float stability = 1.0f / (1.0f + sqrtf(diff_sq) / (sqrtf(norm_sq) + 1e-6f));
        float z_score = (mean_entropy - entropy) / (std_entropy + 1e-6f);
        float confidence = 1.0f / (1.0f + expf(-z_score * 2.0f));
        float cognitive_load = (1.0f - confidence) * 0.5f + (1.0f - stability) * 0.3f + (1.0f - layer_progress) * 0.2f;
        float current_thresh = threshold_base - layer_progress * 0.2f;
        bool should_exit = (confidence * stability * layer_progress) > current_thresh;

        entropy_out[idx] = entropy;
        confidence_out[idx] = confidence;
        load_out[idx] = cognitive_load;
        decision_out[idx] = should_exit;
    }
}

// ============================================================================
// Kernel 2: Optimized Token Router with Prefix Sum
// ============================================================================

__global__ void compute_prefix_sum_kernel(
    const bool* __restrict__ decision_mask,
    int* __restrict__ exit_prefix,
    int* __restrict__ active_prefix,
    int N) {
    
    extern __shared__ int shared_data[];
    int* s_exit = shared_data;
    int* s_active = shared_data + blockDim.x;
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Load data
    int is_exit = (gid < N) ? (decision_mask[gid] ? 1 : 0) : 0;
    int is_active = (gid < N) ? (decision_mask[gid] ? 0 : 1) : 0;
    
    s_exit[tid] = is_exit;
    s_active[tid] = is_active;
    __syncthreads();
    
    // Inclusive scan (Blelloch)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            s_exit[idx] += s_exit[idx - stride];
            s_active[idx] += s_active[idx - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep
    if (tid == 0) {
        s_exit[blockDim.x - 1] = 0;
        s_active[blockDim.x - 1] = 0;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x) {
            int temp_e = s_exit[idx - stride];
            int temp_a = s_active[idx - stride];
            s_exit[idx - stride] = s_exit[idx];
            s_active[idx - stride] = s_active[idx];
            s_exit[idx] += temp_e;
            s_active[idx] += temp_a;
        }
        __syncthreads();
    }
    
    if (gid < N) {
        exit_prefix[gid] = s_exit[tid];
        active_prefix[gid] = s_active[tid];
    }
}

template <typename scalar_t>
__global__ void scatter_tokens_kernel(
    const scalar_t* __restrict__ hidden,
    const bool* __restrict__ decision_mask,
    const int* __restrict__ exit_prefix,
    const int* __restrict__ active_prefix,
    scalar_t* __restrict__ active_out,
    scalar_t* __restrict__ exit_out,
    int* __restrict__ active_indices,
    int* __restrict__ exit_indices,
    int N,
    int H) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = tid / H;
    int h_idx = tid % H;
    
    if (token_idx < N) {
        bool should_exit = decision_mask[token_idx];
        scalar_t val = hidden[token_idx * H + h_idx];
        
        if (should_exit) {
            int out_idx = exit_prefix[token_idx];
            exit_out[out_idx * H + h_idx] = val;
            if (h_idx == 0) exit_indices[out_idx] = token_idx;
        } else {
            int out_idx = active_prefix[token_idx];
            active_out[out_idx * H + h_idx] = val;
            if (h_idx == 0) active_indices[out_idx] = token_idx;
        }
    }
}

// ============================================================================
// Kernel 3: Vectorized Token Router Merge
// ============================================================================

template <typename scalar_t>
__global__ void token_router_merge_kernel_v2(
    const scalar_t* __restrict__ active_hidden,
    const int* __restrict__ active_indices,
    const scalar_t* __restrict__ exit_hidden,
    const int* __restrict__ exit_indices,
    scalar_t* __restrict__ output,
    int n_active,
    int n_exit,
    int H) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_active + n_exit;
    int token_idx = tid / H;
    int h_idx = tid % H;
    
    if (token_idx < n_active) {
        int out_idx = active_indices[token_idx];
        output[out_idx * H + h_idx] = active_hidden[token_idx * H + h_idx];
    } else if (token_idx < total) {
        int exit_token = token_idx - n_active;
        int out_idx = exit_indices[exit_token];
        output[out_idx * H + h_idx] = exit_hidden[exit_token * H + h_idx];
    }
}

// ============================================================================
// Host Functions with Dynamic Block Size Selection
// ============================================================================

std::vector<torch::Tensor> fused_entropy_decision_v2_cuda(
    torch::Tensor logits,
    torch::Tensor hidden,
    torch::Tensor prev_hidden,
    float mean_entropy,
    float std_entropy,
    float layer_progress,
    float threshold) {
    
    const int N = logits.size(0);
    const int vocab_size = logits.size(1);
    const int hidden_size = hidden.size(1);

    auto entropy = torch::empty({N}, logits.options().dtype(torch::kFloat32));
    auto confidence = torch::empty({N}, logits.options().dtype(torch::kFloat32));
    auto decision = torch::empty({N}, logits.options().dtype(torch::kBool));
    auto load = torch::empty({N}, logits.options().dtype(torch::kFloat32));

    // Dynamic block size selection based on vocab_size
    int block_size;
    if (vocab_size < 10000) {
        block_size = BLOCK_SMALL;
    } else if (vocab_size < 50000) {
        block_size = BLOCK_MEDIUM;
    } else if (vocab_size < 100000) {
        block_size = BLOCK_LARGE;
    } else {
        block_size = BLOCK_XLARGE;
    }

    // Check dtype and dispatch
    if (logits.scalar_type() == torch::kFloat16) {
        // FP16 path
        if (block_size == BLOCK_SMALL) {
            fused_entropy_decision_kernel_fp16<BLOCK_SMALL><<<N, BLOCK_SMALL>>>(
                reinterpret_cast<const __half*>(logits.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(prev_hidden.data_ptr<at::Half>()),
                entropy.data_ptr<float>(),
                confidence.data_ptr<float>(),
                decision.data_ptr<bool>(),
                load.data_ptr<float>(),
                vocab_size, hidden_size,
                mean_entropy, std_entropy, layer_progress, threshold);
        } else if (block_size == BLOCK_MEDIUM) {
            fused_entropy_decision_kernel_fp16<BLOCK_MEDIUM><<<N, BLOCK_MEDIUM>>>(
                reinterpret_cast<const __half*>(logits.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(prev_hidden.data_ptr<at::Half>()),
                entropy.data_ptr<float>(),
                confidence.data_ptr<float>(),
                decision.data_ptr<bool>(),
                load.data_ptr<float>(),
                vocab_size, hidden_size,
                mean_entropy, std_entropy, layer_progress, threshold);
        } else if (block_size == BLOCK_LARGE) {
            fused_entropy_decision_kernel_fp16<BLOCK_LARGE><<<N, BLOCK_LARGE>>>(
                reinterpret_cast<const __half*>(logits.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(prev_hidden.data_ptr<at::Half>()),
                entropy.data_ptr<float>(),
                confidence.data_ptr<float>(),
                decision.data_ptr<bool>(),
                load.data_ptr<float>(),
                vocab_size, hidden_size,
                mean_entropy, std_entropy, layer_progress, threshold);
        } else {
            fused_entropy_decision_kernel_fp16<BLOCK_XLARGE><<<N, BLOCK_XLARGE>>>(
                reinterpret_cast<const __half*>(logits.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(prev_hidden.data_ptr<at::Half>()),
                entropy.data_ptr<float>(),
                confidence.data_ptr<float>(),
                decision.data_ptr<bool>(),
                load.data_ptr<float>(),
                vocab_size, hidden_size,
                mean_entropy, std_entropy, layer_progress, threshold);
        }
    } else {
        // FP32 path
        if (block_size == BLOCK_SMALL) {
            fused_entropy_decision_kernel_v2<BLOCK_SMALL><<<N, BLOCK_SMALL>>>(
                logits.data_ptr<float>(),
                hidden.data_ptr<float>(),
                prev_hidden.data_ptr<float>(),
                entropy.data_ptr<float>(),
                confidence.data_ptr<float>(),
                decision.data_ptr<bool>(),
                load.data_ptr<float>(),
                vocab_size, hidden_size,
                mean_entropy, std_entropy, layer_progress, threshold);
        } else if (block_size == BLOCK_MEDIUM) {
            fused_entropy_decision_kernel_v2<BLOCK_MEDIUM><<<N, BLOCK_MEDIUM>>>(
                logits.data_ptr<float>(),
                hidden.data_ptr<float>(),
                prev_hidden.data_ptr<float>(),
                entropy.data_ptr<float>(),
                confidence.data_ptr<float>(),
                decision.data_ptr<bool>(),
                load.data_ptr<float>(),
                vocab_size, hidden_size,
                mean_entropy, std_entropy, layer_progress, threshold);
        } else if (block_size == BLOCK_LARGE) {
            fused_entropy_decision_kernel_v2<BLOCK_LARGE><<<N, BLOCK_LARGE>>>(
                logits.data_ptr<float>(),
                hidden.data_ptr<float>(),
                prev_hidden.data_ptr<float>(),
                entropy.data_ptr<float>(),
                confidence.data_ptr<float>(),
                decision.data_ptr<bool>(),
                load.data_ptr<float>(),
                vocab_size, hidden_size,
                mean_entropy, std_entropy, layer_progress, threshold);
        } else {
            fused_entropy_decision_kernel_v2<BLOCK_XLARGE><<<N, BLOCK_XLARGE>>>(
                logits.data_ptr<float>(),
                hidden.data_ptr<float>(),
                prev_hidden.data_ptr<float>(),
                entropy.data_ptr<float>(),
                confidence.data_ptr<float>(),
                decision.data_ptr<bool>(),
                load.data_ptr<float>(),
                vocab_size, hidden_size,
                mean_entropy, std_entropy, layer_progress, threshold);
        }
    }

    return {entropy, confidence, decision, load};
}

std::vector<torch::Tensor> token_router_split_v2_cuda(
    torch::Tensor hidden,
    torch::Tensor decision_mask) {
    
    const int N = hidden.size(0);
    const int H = hidden.size(1);
    
    auto active_out = torch::empty({N, H}, hidden.options());
    auto exit_out = torch::empty({N, H}, hidden.options());
    auto active_indices = torch::empty({N}, hidden.options().dtype(torch::kInt32));
    auto exit_indices = torch::empty({N}, hidden.options().dtype(torch::kInt32));
    
    // Use PyTorch's cumsum for prefix sum (simpler and fast enough)
    auto exit_mask_int = decision_mask.to(torch::kInt32);
    auto active_mask_int = (~decision_mask).to(torch::kInt32);
    
    auto exit_prefix = (exit_mask_int.cumsum(0) - exit_mask_int).to(torch::kInt32);
    auto active_prefix = (active_mask_int.cumsum(0) - active_mask_int).to(torch::kInt32);
    
    int n_exit = exit_mask_int.sum().item<int>();
    int n_active = N - n_exit;
    
    const int threads = 256;
    const int blocks = (N * H + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hidden.scalar_type(), "scatter_tokens_kernel", ([&] {
        scatter_tokens_kernel<scalar_t><<<blocks, threads>>>(
            hidden.data_ptr<scalar_t>(),
            decision_mask.data_ptr<bool>(),
            exit_prefix.data_ptr<int>(),
            active_prefix.data_ptr<int>(),
            active_out.data_ptr<scalar_t>(),
            exit_out.data_ptr<scalar_t>(),
            active_indices.data_ptr<int>(),
            exit_indices.data_ptr<int>(),
            N, H
        );
    }));
    
    return {
        active_out.slice(0, 0, n_active),
        active_indices.slice(0, 0, n_active),
        exit_out.slice(0, 0, n_exit),
        exit_indices.slice(0, 0, n_exit)
    };
}

torch::Tensor token_router_merge_v2_cuda(
    torch::Tensor active_hidden,
    torch::Tensor active_indices,
    torch::Tensor exit_hidden,
    torch::Tensor exit_indices,
    int total_size) {
    
    const int n_active = active_hidden.size(0);
    const int n_exit = exit_hidden.size(0);
    const int H = active_hidden.size(1);
    
    auto output = torch::empty({total_size, H}, active_hidden.options());
    
    const int threads = 256;
    const int blocks = ((n_active + n_exit) * H + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(active_hidden.scalar_type(), "token_router_merge_v2", ([&] {
        token_router_merge_kernel_v2<scalar_t><<<blocks, threads>>>(
            active_hidden.data_ptr<scalar_t>(),
            active_indices.data_ptr<int>(),
            exit_hidden.data_ptr<scalar_t>(),
            exit_indices.data_ptr<int>(),
            output.data_ptr<scalar_t>(),
            n_active, n_exit, H
        );
    }));
    
    return output;
}

// ============================================================================
// Batched Version for Multi-Head Processing
// ============================================================================

// Batched entropy decision - processes multiple sequences in parallel
std::vector<torch::Tensor> batched_entropy_decision_cuda(
    torch::Tensor logits,      // [batch, seq, vocab]
    torch::Tensor hidden,      // [batch, seq, hidden]
    torch::Tensor prev_hidden, // [batch, seq, hidden]
    float mean_entropy,
    float std_entropy,
    float layer_progress,
    float threshold) {
    
    const int batch = logits.size(0);
    const int seq = logits.size(1);
    const int N = batch * seq;
    
    // Reshape to [N, ...]
    auto logits_flat = logits.view({N, -1});
    auto hidden_flat = hidden.view({N, -1});
    auto prev_flat = prev_hidden.view({N, -1});
    
    auto results = fused_entropy_decision_v2_cuda(
        logits_flat, hidden_flat, prev_flat,
        mean_entropy, std_entropy, layer_progress, threshold
    );
    
    // Reshape outputs
    return {
        results[0].view({batch, seq}),  // entropy
        results[1].view({batch, seq}),  // confidence
        results[2].view({batch, seq}),  // decision
        results[3].view({batch, seq})   // load
    };
}
