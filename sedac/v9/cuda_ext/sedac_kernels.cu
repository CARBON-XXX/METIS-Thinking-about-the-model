// sedac_kernels.cu
// SEDAC V9.0 - Core CUDA Kernels
// Target: 48ms -> 0.2ms

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

// Kernel 1: Fused Entropy Decision
template <typename scalar_t>
__global__ void fused_entropy_decision_kernel(
    const scalar_t* __restrict__ logits,
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ prev_hidden,
    float* __restrict__ entropy_out,
    float* __restrict__ confidence_out,
    bool* __restrict__ decision_out,
    float* __restrict__ load_out,
    int vocab_size,
    int hidden_size,
    float mean_entropy,
    float std_entropy,
    float layer_progress,
    float threshold_base) {

    int idx = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared_max[BLOCK_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_diff_sq[BLOCK_SIZE];
    __shared__ float shared_norm_sq[BLOCK_SIZE];

    // Step A: Find Max
    float max_val = -1e20f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        max_val = fmaxf(max_val, static_cast<float>(logits[idx * vocab_size + i]));
    }
    shared_max[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    __syncthreads();

    // Step B: Compute Sum Exp
    float sum_exp = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        sum_exp += expf(static_cast<float>(logits[idx * vocab_size + i]) - max_val);
    }
    shared_sum[tid] = sum_exp;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    sum_exp = shared_sum[0];
    float log_sum_exp = logf(sum_exp) + max_val;
    __syncthreads();

    // Step C: Compute Entropy
    float sum_p_logits = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = static_cast<float>(logits[idx * vocab_size + i]);
        float p = expf(val - max_val) / sum_exp;
        sum_p_logits += p * val;
    }
    shared_sum[tid] = sum_p_logits;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    sum_p_logits = shared_sum[0];
    float entropy = (log_sum_exp - sum_p_logits) / 0.69314718f;

    // Step D: Stability
    float diff_sq = 0.0f;
    float norm_sq = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float h = static_cast<float>(hidden[idx * hidden_size + i]);
        float ph = static_cast<float>(prev_hidden[idx * hidden_size + i]);
        float diff = h - ph;
        diff_sq += diff * diff;
        norm_sq += h * h;
    }
    shared_diff_sq[tid] = diff_sq;
    shared_norm_sq[tid] = norm_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_diff_sq[tid] += shared_diff_sq[tid + s];
            shared_norm_sq[tid] += shared_norm_sq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        diff_sq = shared_diff_sq[0];
        norm_sq = shared_norm_sq[0];
        
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

std::vector<torch::Tensor> fused_entropy_decision_cuda(
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

    const int threads = BLOCK_SIZE;
    const int blocks = N;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "fused_entropy_decision_kernel", ([&] {
        fused_entropy_decision_kernel<scalar_t><<<blocks, threads>>>(
            logits.data_ptr<scalar_t>(),
            hidden.data_ptr<scalar_t>(),
            prev_hidden.data_ptr<scalar_t>(),
            entropy.data_ptr<float>(),
            confidence.data_ptr<float>(),
            decision.data_ptr<bool>(),
            load.data_ptr<float>(),
            vocab_size,
            hidden_size,
            mean_entropy,
            std_entropy,
            layer_progress,
            threshold
        );
    }));

    return {entropy, confidence, decision, load};
}


// Kernel 2: Token Router Split
template <typename scalar_t>
__global__ void token_router_split_kernel(
    const scalar_t* __restrict__ hidden,
    const bool* __restrict__ decision_mask,
    scalar_t* __restrict__ active_out,
    scalar_t* __restrict__ exit_out,
    int* __restrict__ active_indices,
    int* __restrict__ exit_indices,
    int* __restrict__ active_counter,
    int* __restrict__ exit_counter,
    int N,
    int H) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        bool should_exit = decision_mask[tid];
        
        if (should_exit) {
            int out_idx = atomicAdd(exit_counter, 1);
            exit_indices[out_idx] = tid;
            for (int i = 0; i < H; i++) {
                exit_out[out_idx * H + i] = hidden[tid * H + i];
            }
        } else {
            int out_idx = atomicAdd(active_counter, 1);
            active_indices[out_idx] = tid;
            for (int i = 0; i < H; i++) {
                active_out[out_idx * H + i] = hidden[tid * H + i];
            }
        }
    }
}

std::vector<torch::Tensor> token_router_split_cuda(
    torch::Tensor hidden,
    torch::Tensor decision_mask) {
    
    const int N = hidden.size(0);
    const int H = hidden.size(1);
    
    auto active_out = torch::empty({N, H}, hidden.options());
    auto exit_out = torch::empty({N, H}, hidden.options());
    auto active_indices = torch::empty({N}, hidden.options().dtype(torch::kInt32));
    auto exit_indices = torch::empty({N}, hidden.options().dtype(torch::kInt32));
    auto active_counter = torch::zeros({1}, hidden.options().dtype(torch::kInt32));
    auto exit_counter = torch::zeros({1}, hidden.options().dtype(torch::kInt32));
    
    const int threads = BLOCK_SIZE;
    const int blocks = (N + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hidden.scalar_type(), "token_router_split_kernel", ([&] {
        token_router_split_kernel<scalar_t><<<blocks, threads>>>(
            hidden.data_ptr<scalar_t>(),
            decision_mask.data_ptr<bool>(),
            active_out.data_ptr<scalar_t>(),
            exit_out.data_ptr<scalar_t>(),
            active_indices.data_ptr<int>(),
            exit_indices.data_ptr<int>(),
            active_counter.data_ptr<int>(),
            exit_counter.data_ptr<int>(),
            N, H
        );
    }));
    
    int n_active = active_counter.item<int>();
    int n_exit = exit_counter.item<int>();
    
    return {
        active_out.slice(0, 0, n_active),
        active_indices.slice(0, 0, n_active),
        exit_out.slice(0, 0, n_exit),
        exit_indices.slice(0, 0, n_exit)
    };
}


// Kernel 3: Token Router Merge
template <typename scalar_t>
__global__ void token_router_merge_kernel(
    const scalar_t* __restrict__ active_hidden,
    const int* __restrict__ active_indices,
    const scalar_t* __restrict__ exit_hidden,
    const int* __restrict__ exit_indices,
    scalar_t* __restrict__ output,
    int n_active,
    int n_exit,
    int H) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n_active) {
        int out_idx = active_indices[tid];
        for (int i = 0; i < H; i++) {
            output[out_idx * H + i] = active_hidden[tid * H + i];
        }
    }
    
    int exit_tid = tid - n_active;
    if (exit_tid >= 0 && exit_tid < n_exit) {
        int out_idx = exit_indices[exit_tid];
        for (int i = 0; i < H; i++) {
            output[out_idx * H + i] = exit_hidden[exit_tid * H + i];
        }
    }
}

torch::Tensor token_router_merge_cuda(
    torch::Tensor active_hidden,
    torch::Tensor active_indices,
    torch::Tensor exit_hidden,
    torch::Tensor exit_indices,
    int total_size) {
    
    const int n_active = active_hidden.size(0);
    const int n_exit = exit_hidden.size(0);
    const int H = active_hidden.size(1);
    
    auto output = torch::empty({total_size, H}, active_hidden.options());
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n_active + n_exit + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(active_hidden.scalar_type(), "token_router_merge_kernel", ([&] {
        token_router_merge_kernel<scalar_t><<<blocks, threads>>>(
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


// Kernel 4: KV Cache Update
template <typename scalar_t>
__global__ void kv_cache_update_kernel(
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ wk,
    const scalar_t* __restrict__ wv,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int* __restrict__ positions,
    int N,
    int hidden_size,
    int kv_dim) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = tid / kv_dim;
    int kv_idx = tid % kv_dim;
    
    if (token_idx < N) {
        int pos = positions[token_idx];
        
        float k_val = 0.0f;
        float v_val = 0.0f;
        
        for (int h = 0; h < hidden_size; h++) {
            float h_val = static_cast<float>(hidden[token_idx * hidden_size + h]);
            k_val += h_val * static_cast<float>(wk[h * kv_dim + kv_idx]);
            v_val += h_val * static_cast<float>(wv[h * kv_dim + kv_idx]);
        }
        
        k_cache[pos * kv_dim + kv_idx] = static_cast<scalar_t>(k_val);
        v_cache[pos * kv_dim + kv_idx] = static_cast<scalar_t>(v_val);
    }
}

void kv_cache_update_cuda(
    torch::Tensor hidden,
    torch::Tensor wk,
    torch::Tensor wv,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor positions) {
    
    const int N = hidden.size(0);
    const int hidden_size = hidden.size(1);
    const int kv_dim = wk.size(1);
    
    const int threads = BLOCK_SIZE;
    const int blocks = (N * kv_dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hidden.scalar_type(), "kv_cache_update_kernel", ([&] {
        kv_cache_update_kernel<scalar_t><<<blocks, threads>>>(
            hidden.data_ptr<scalar_t>(),
            wk.data_ptr<scalar_t>(),
            wv.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            positions.data_ptr<int>(),
            N, hidden_size, kv_dim
        );
    }));
}
