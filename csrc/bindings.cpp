#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> fused_decode_cuda(
    torch::Tensor query,
    torch::Tensor packed_k,
    torch::Tensor packed_v,
    torch::Tensor scales_k,
    torch::Tensor scales_v,
    torch::Tensor valid_tokens,
    int64_t bits,
    int64_t group_size,
    double sm_scale);

std::tuple<torch::Tensor, torch::Tensor> fused_decode(
    torch::Tensor query,
    torch::Tensor packed_k,
    torch::Tensor packed_v,
    torch::Tensor scales_k,
    torch::Tensor scales_v,
    torch::Tensor valid_tokens,
    int64_t bits,
    int64_t group_size,
    double sm_scale) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(packed_k.is_cuda() && packed_v.is_cuda(), "packed KV must be CUDA");
  TORCH_CHECK(scales_k.is_cuda() && scales_v.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(valid_tokens.is_cuda(), "valid_tokens must be CUDA");
  TORCH_CHECK(query.dim() == 2, "query must be [query_heads, head_dim]");
  TORCH_CHECK(packed_k.dim() == 4 && packed_k.sizes() == packed_v.sizes(),
              "packed KV must be matching [blocks,tokens,kv_heads,packed_dim]");
  TORCH_CHECK(scales_k.dim() == 4 && scales_k.sizes() == scales_v.sizes(),
              "scales must be matching [blocks,tokens,kv_heads,groups]");
  TORCH_CHECK(packed_k.scalar_type() == torch::kUInt8, "packed KV must be uint8");
  TORCH_CHECK(scales_k.scalar_type() == torch::kFloat16, "scales must be float16");
  TORCH_CHECK(valid_tokens.scalar_type() == torch::kInt32, "valid_tokens must be int32");
  TORCH_CHECK(bits == 2 || bits == 4, "bits must be 2 or 4");
  return fused_decode_cuda(query, packed_k, packed_v, scales_k, scales_v,
                           valid_tokens, bits, group_size, sm_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_decode", &fused_decode, "Fused 2/4-bit decode attention (CUDA)");
}

