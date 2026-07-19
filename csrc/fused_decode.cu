#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <tuple>

template <typename scalar_t>
__global__ void fused_decode_kernel(
    const scalar_t* __restrict__ query,
    const uint8_t* __restrict__ packed_k,
    const uint8_t* __restrict__ packed_v,
    const at::Half* __restrict__ scales_k,
    const at::Half* __restrict__ scales_v,
    const int32_t* __restrict__ valid_tokens,
    float* __restrict__ output,
    float* __restrict__ lse,
    int blocks,
    int tokens_per_block,
    int query_heads,
    int kv_heads,
    int head_dim,
    int packed_dim,
    int groups,
    int bits,
    int group_size,
    float sm_scale) {
  const int qh = blockIdx.x;
  const int tid = threadIdx.x;
  const int kvh = qh / (query_heads / kv_heads);
  const int per_byte = 8 / bits;
  const int code_mask = (1 << bits) - 1;
  const int midpoint = 1 << (bits - 1);
  extern __shared__ float shared[];
  float* reduce = shared;
  float* state = shared + blockDim.x;

  float out_acc = 0.0f;
  float running_m = -INFINITY;
  float running_l = 0.0f;

  for (int b = 0; b < blocks; ++b) {
    for (int t = 0; t < valid_tokens[b]; ++t) {
      float partial = 0.0f;
      for (int d = tid; d < head_dim; d += blockDim.x) {
        const int byte_index = (((b * tokens_per_block + t) * kv_heads + kvh) * packed_dim)
                               + d / per_byte;
        const int shift = (d % per_byte) * bits;
        const int code = (packed_k[byte_index] >> shift) & code_mask;
        const int scale_index = (((b * tokens_per_block + t) * kv_heads + kvh) * groups)
                                + d / group_size;
        const float kval = (code - midpoint) * static_cast<float>(scales_k[scale_index]);
        partial += static_cast<float>(query[qh * head_dim + d]) * kval;
      }
      reduce[tid] = partial;
      __syncthreads();
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
      }
      if (tid == 0) {
        const float score = reduce[0] * sm_scale;
        const float next_m = fmaxf(running_m, score);
        state[0] = expf(running_m - next_m);
        state[1] = expf(score - next_m);
        running_l = state[0] * running_l + state[1];
        running_m = next_m;
        state[2] = running_l;
      }
      __syncthreads();
      if (tid < head_dim) {
        const int byte_index = (((b * tokens_per_block + t) * kv_heads + kvh) * packed_dim)
                               + tid / per_byte;
        const int shift = (tid % per_byte) * bits;
        const int code = (packed_v[byte_index] >> shift) & code_mask;
        const int scale_index = (((b * tokens_per_block + t) * kv_heads + kvh) * groups)
                                + tid / group_size;
        const float v = (code - midpoint) * static_cast<float>(scales_v[scale_index]);
        out_acc = out_acc * state[0] + v * state[1];
      }
      __syncthreads();
    }
  }
  if (tid < head_dim) output[qh * head_dim + tid] = out_acc / state[2];
  if (tid == 0) lse[qh] = running_m + logf(running_l);
}

std::tuple<torch::Tensor, torch::Tensor> fused_decode_cuda(
    torch::Tensor query,
    torch::Tensor packed_k,
    torch::Tensor packed_v,
    torch::Tensor scales_k,
    torch::Tensor scales_v,
    torch::Tensor valid_tokens,
    int64_t bits,
    int64_t group_size,
    double sm_scale) {
  c10::cuda::CUDAGuard guard(query.device());
  query = query.contiguous();
  packed_k = packed_k.contiguous();
  packed_v = packed_v.contiguous();
  scales_k = scales_k.contiguous();
  scales_v = scales_v.contiguous();
  valid_tokens = valid_tokens.contiguous();

  const int blocks = packed_k.size(0);
  const int tokens = packed_k.size(1);
  const int kv_heads = packed_k.size(2);
  const int packed_dim = packed_k.size(3);
  const int query_heads = query.size(0);
  const int head_dim = query.size(1);
  const int groups = scales_k.size(3);
  TORCH_CHECK(head_dim <= 256, "reference CUDA kernel supports head_dim <= 256");
  TORCH_CHECK(query_heads % kv_heads == 0, "query heads must be divisible by KV heads");
  TORCH_CHECK(head_dim % group_size == 0, "group_size must divide head_dim");
  TORCH_CHECK(packed_dim == head_dim * bits / 8, "packed dimension mismatch");
  TORCH_CHECK(groups == head_dim / group_size, "scale dimension mismatch");

  int threads = 32;
  while (threads < head_dim) threads <<= 1;
  auto options = query.options().dtype(torch::kFloat32);
  auto output = torch::empty({query_heads, head_dim}, options);
  auto lse = torch::empty({query_heads}, options);
  const size_t shared_bytes = (threads + 3) * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      "kvssd_fused_decode", [&] {
        fused_decode_kernel<scalar_t><<<query_heads, threads, shared_bytes, stream>>>(
            query.data_ptr<scalar_t>(), packed_k.data_ptr<uint8_t>(), packed_v.data_ptr<uint8_t>(),
            scales_k.data_ptr<at::Half>(), scales_v.data_ptr<at::Half>(),
            valid_tokens.data_ptr<int32_t>(), output.data_ptr<float>(), lse.data_ptr<float>(),
            blocks, tokens, query_heads, kv_heads, head_dim, packed_dim, groups,
            static_cast<int>(bits), static_cast<int>(group_size), static_cast<float>(sm_scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}
