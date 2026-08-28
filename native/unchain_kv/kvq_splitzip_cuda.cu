#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

namespace {

thread_local std::string g_last_error;

int Fail(const char* what, cudaError_t status) {
  g_last_error = std::string(what) + ": " + cudaGetErrorString(status);
  return -1;
}

constexpr int kMapEmpty = -1;
constexpr int kMapClaimed = -2;
constexpr int kFixed6Palette = 64;
constexpr int kFixed5Palette = 32;

__global__ void InitPalette(unsigned char* dst, int mode, int* map, int* count,
                            int* overflow) {
  dst[0] = static_cast<unsigned char>(mode);
  *count = 0;
  *overflow = 0;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) map[i] = kMapEmpty;
}

__global__ void BuildPalette(const uint16_t* src,
                             unsigned long long count_words,
                             unsigned char* palette, int* map, int* count,
                             int* overflow, int palette_limit) {
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count_words; index += blockDim.x * gridDim.x) {
    int high = (src[index] >> 8) & 0xFF;
    int code = map[high];
    if (code >= 0) continue;
    if (atomicCAS(&map[high], kMapEmpty, kMapClaimed) == kMapEmpty) {
      int slot = atomicAdd(count, 1);
      if (slot < palette_limit) {
        palette[slot] = static_cast<unsigned char>(high);
        __threadfence();
        map[high] = slot;
      } else {
        *overflow = 1;
        map[high] = 0;
      }
    }
  }
}

__device__ int WaitCode(const int* map, int high) {
  int code = map[high];
  while (code == kMapClaimed) code = map[high];
  return code;
}

__global__ void EncodeFixed6(const uint16_t* src, unsigned long long count_words,
                             unsigned char* dst, int* map, int* overflow) {
  if (*overflow) {
    if (blockIdx.x == 0 && threadIdx.x == 0) dst[0] = 255;
    return;
  }
  unsigned long long group = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long base = group * 4;
  if (base >= count_words) return;
  unsigned char* low_out = dst + 65;
  unsigned char* code_out = low_out + count_words;
  unsigned int bits = 0;
  for (int item = 0; item < 4; ++item) {
    unsigned long long index = base + item;
    if (index >= count_words) break;
    uint16_t word = src[index];
    low_out[index] = static_cast<unsigned char>(word & 0xFF);
    int high = (word >> 8) & 0xFF;
    int code = WaitCode(map, high);
    bits |= static_cast<unsigned int>(code & 0x3F) << (item * 6);
  }
  unsigned long long out = group * 3;
  code_out[out] = static_cast<unsigned char>(bits & 0xFF);
  code_out[out + 1] = static_cast<unsigned char>((bits >> 8) & 0xFF);
  code_out[out + 2] = static_cast<unsigned char>((bits >> 16) & 0xFF);
}

__global__ void EncodeFixed5(const uint16_t* src, unsigned long long count_words,
                             unsigned char* dst, int* map, int* overflow) {
  if (*overflow) {
    if (blockIdx.x == 0 && threadIdx.x == 0) dst[0] = 255;
    return;
  }
  unsigned long long group = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long base = group * 8;
  if (base >= count_words) return;
  unsigned char* low_out = dst + 33;
  unsigned char* code_out = low_out + count_words;
  unsigned long long bits = 0;
  for (int item = 0; item < 8; ++item) {
    unsigned long long index = base + item;
    if (index >= count_words) break;
    uint16_t word = src[index];
    low_out[index] = static_cast<unsigned char>(word & 0xFF);
    int high = (word >> 8) & 0xFF;
    int code = WaitCode(map, high);
    bits |= static_cast<unsigned long long>(code & 0x1F) << (item * 5);
  }
  unsigned long long out = group * 5;
  code_out[out] = static_cast<unsigned char>(bits & 0xFF);
  code_out[out + 1] = static_cast<unsigned char>((bits >> 8) & 0xFF);
  code_out[out + 2] = static_cast<unsigned char>((bits >> 16) & 0xFF);
  code_out[out + 3] = static_cast<unsigned char>((bits >> 24) & 0xFF);
  code_out[out + 4] = static_cast<unsigned char>((bits >> 32) & 0xFF);
}

__global__ void DecodeFixed6(const unsigned char* src, uint16_t* dst,
                             unsigned long long count_words,
                             unsigned long long block_count,
                             unsigned long long block_words,
                             unsigned long long plane_stride,
                             unsigned long long block_stride,
                             unsigned long long block_start,
                             const int64_t* block_ids) {
  if (src[0] != 3) return;
  const unsigned char* palette = src + 1;
  const unsigned char* low = src + 65;
  const unsigned char* codes = low + count_words;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count_words; index += blockDim.x * gridDim.x) {
    unsigned long long group = index / 4;
    unsigned int packed = static_cast<unsigned int>(codes[group * 3]) |
                          (static_cast<unsigned int>(codes[group * 3 + 1]) << 8) |
                          (static_cast<unsigned int>(codes[group * 3 + 2]) << 16);
    unsigned int code = (packed >> ((index % 4) * 6)) & 0x3F;
    unsigned long long plane_words = block_count * block_words;
    unsigned long long plane = index / plane_words;
    unsigned long long in_plane = index % plane_words;
    unsigned long long block = in_plane / block_words;
    unsigned long long within = in_plane % block_words;
    unsigned long long dst_block = block_ids
                                       ? static_cast<unsigned long long>(block_ids[block])
                                       : block_start + block;
    dst[plane * plane_stride + dst_block * block_stride + within] =
        static_cast<uint16_t>((static_cast<uint16_t>(palette[code]) << 8) |
                              low[index]);
  }
}

__global__ void InitTop16(unsigned char* dst, unsigned int* escape_count,
                          unsigned int* overflow) {
  dst[0] = 5;
  for (int i = 1; i < 5; ++i) dst[i] = 0;
  *escape_count = 0;
  *overflow = 0;
}

__global__ void EncodeTop16(const uint16_t* src,
                            unsigned long long count_words,
                            unsigned char* dst, const unsigned char* tables,
                            unsigned int* escape_count,
                            unsigned int* overflow,
                            unsigned int escape_capacity) {
  unsigned long long groups = (count_words + 1) / 2;
  unsigned char* body = dst + 5;
  unsigned char* codes = body + count_words;
  unsigned char* entries = codes + groups;
  for (unsigned long long group = blockIdx.x * blockDim.x + threadIdx.x;
       group < groups; group += blockDim.x * gridDim.x) {
    unsigned char packed = 0;
    for (int item = 0; item < 2; ++item) {
      unsigned long long index = group * 2 + item;
      if (index >= count_words) break;
      uint16_t word = src[index];
      unsigned int exponent = (word >> 7) & 0xFF;
      unsigned long long plane = index >= count_words / 2;
      unsigned int code = tables[plane * 256 + exponent];
      body[index] = static_cast<unsigned char>((word & 0x7F) |
                                               ((word >> 8) & 0x80));
      if (code == 255) {
        unsigned int slot = atomicAdd(escape_count, 1U);
        if (slot < escape_capacity) {
          unsigned char* entry = entries + slot * 5;
          unsigned int position = static_cast<unsigned int>(index);
          entry[0] = static_cast<unsigned char>(position);
          entry[1] = static_cast<unsigned char>(position >> 8);
          entry[2] = static_cast<unsigned char>(position >> 16);
          entry[3] = static_cast<unsigned char>(position >> 24);
          entry[4] = static_cast<unsigned char>(exponent);
        } else {
          *overflow = 1;
        }
        code = 0;
      }
      packed |= static_cast<unsigned char>((code & 0x0F) << (item * 4));
    }
    codes[group] = packed;
  }
}

__global__ void FinalizeTop16(unsigned char* dst,
                              const unsigned int* escape_count,
                              const unsigned int* overflow,
                              unsigned int escape_capacity) {
  unsigned int count = *escape_count;
  dst[0] = *overflow || count > escape_capacity ? 255 : 5;
  if (count > escape_capacity) count = escape_capacity;
  dst[1] = static_cast<unsigned char>(count);
  dst[2] = static_cast<unsigned char>(count >> 8);
  dst[3] = static_cast<unsigned char>(count >> 16);
  dst[4] = static_cast<unsigned char>(count >> 24);
}

__device__ unsigned long long Top16DstIndex(
    unsigned long long index, unsigned long long block_count,
    unsigned long long block_words, unsigned long long plane_stride,
    unsigned long long block_stride, unsigned long long block_start,
    const int64_t* block_ids) {
  unsigned long long plane_words = block_count * block_words;
  unsigned long long plane = index / plane_words;
  unsigned long long in_plane = index % plane_words;
  unsigned long long block = in_plane / block_words;
  unsigned long long within = in_plane % block_words;
  unsigned long long dst_block = block_ids
                                     ? static_cast<unsigned long long>(block_ids[block])
                                     : block_start + block;
  return plane * plane_stride + dst_block * block_stride + within;
}

__global__ void DecodeTop16(const unsigned char* src, uint16_t* dst,
                            unsigned long long count_words,
                            unsigned long long block_count,
                            unsigned long long block_words,
                            unsigned long long plane_stride,
                            unsigned long long block_stride,
                            unsigned long long block_start,
                            const int64_t* block_ids,
                            const unsigned char* tables) {
  if (src[0] != 5) return;
  const unsigned char* body = src + 5;
  const unsigned char* codes = body + count_words;
  const unsigned char* codebooks = tables + 512;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count_words; index += blockDim.x * gridDim.x) {
    unsigned long long plane = index >= count_words / 2;
    unsigned int code = (codes[index / 2] >> ((index % 2) * 4)) & 0x0F;
    unsigned int exponent = codebooks[plane * 16 + code];
    unsigned char value = body[index];
    uint16_t word = static_cast<uint16_t>(value & 0x7F) |
                    (static_cast<uint16_t>(value & 0x80) << 8) |
                    static_cast<uint16_t>(exponent << 7);
    dst[Top16DstIndex(index, block_count, block_words, plane_stride,
                      block_stride, block_start, block_ids)] = word;
  }
}

__global__ void ApplyTop16Escapes(const unsigned char* src, uint16_t* dst,
                                  unsigned long long count_words,
                                  unsigned long long block_count,
                                  unsigned long long block_words,
                                  unsigned long long plane_stride,
                                  unsigned long long block_stride,
                                  unsigned long long block_start,
                                  const int64_t* block_ids,
                                  unsigned int escape_capacity) {
  if (src[0] != 5) return;
  unsigned int count = static_cast<unsigned int>(src[1]) |
                       (static_cast<unsigned int>(src[2]) << 8) |
                       (static_cast<unsigned int>(src[3]) << 16) |
                       (static_cast<unsigned int>(src[4]) << 24);
  const unsigned char* entries = src + 5 + count_words + (count_words + 1) / 2;
  for (unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
       slot < count && slot < escape_capacity;
       slot += blockDim.x * gridDim.x) {
    const unsigned char* entry = entries + slot * 5;
    unsigned int position = static_cast<unsigned int>(entry[0]) |
                            (static_cast<unsigned int>(entry[1]) << 8) |
                            (static_cast<unsigned int>(entry[2]) << 16) |
                            (static_cast<unsigned int>(entry[3]) << 24);
    if (position >= count_words) continue;
    unsigned long long dst_index =
        Top16DstIndex(position, block_count, block_words, plane_stride,
                      block_stride, block_start, block_ids);
    uint16_t word = dst[dst_index];
    dst[dst_index] = static_cast<uint16_t>((word & 0x807F) |
                                           (entry[4] << 7));
  }
}

}  // namespace

extern "C" const char* kvq_splitzip_last_error() {
  return g_last_error.c_str();
}

extern "C" long long kvq_splitzip_bf16_encode_mode(unsigned long long src_ptr,
                                                   unsigned long long dst_ptr,
                                                   unsigned long long count,
                                                   unsigned long long capacity,
                                                   unsigned long long stream_ptr,
                                                   int bits) {
  if (src_ptr == 0 || dst_ptr == 0 || count == 0) return 0;
  if (bits != 5 && bits != 6) return 0;
  unsigned long long code_bytes =
      bits == 5 ? ((count + 7) / 8) * 5 : ((count + 3) / 4) * 3;
  unsigned long long header_bytes = bits == 5 ? 33 : 65;
  unsigned long long encoded = header_bytes + count + code_bytes;
  unsigned long long scratch_offset = (encoded + 3) & ~3ULL;
  unsigned long long scratch = 256 * sizeof(int) + 2 * sizeof(int);
  if (encoded >= count * 2 || scratch_offset + scratch > capacity) return 0;
  auto* src = reinterpret_cast<const uint16_t*>(src_ptr);
  auto* dst = reinterpret_cast<unsigned char*>(dst_ptr);
  auto* map = reinterpret_cast<int*>(dst + scratch_offset);
  auto* seen = reinterpret_cast<int*>(dst + scratch_offset + 256 * sizeof(int));
  auto* overflow =
      reinterpret_cast<int*>(dst + scratch_offset + 256 * sizeof(int) + sizeof(int));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  InitPalette<<<1, 256, 0, stream>>>(dst, bits == 5 ? 4 : 3, map, seen, overflow);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return Fail("InitPalette", status);
  BuildPalette<<<1024, 256, 0, stream>>>(src, count, dst + 1, map, seen, overflow,
                                         bits == 5 ? kFixed5Palette
                                                   : kFixed6Palette);
  status = cudaGetLastError();
  if (status != cudaSuccess) return Fail("BuildPalette", status);
  if (bits == 5) {
    EncodeFixed5<<<static_cast<unsigned int>((((count + 7) / 8) + 255) / 256),
                   256, 0, stream>>>(src, count, dst, map, overflow);
  } else {
    EncodeFixed6<<<static_cast<unsigned int>((((count + 3) / 4) + 255) / 256),
                   256, 0, stream>>>(src, count, dst, map, overflow);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess) return Fail(bits == 5 ? "EncodeFixed5" : "EncodeFixed6",
                                         status);
  return static_cast<long long>(encoded);
}

extern "C" long long kvq_splitzip_bf16_encode(unsigned long long src_ptr,
                                              unsigned long long dst_ptr,
                                              unsigned long long count,
                                              unsigned long long capacity,
                                              unsigned long long stream_ptr) {
  return kvq_splitzip_bf16_encode_mode(src_ptr, dst_ptr, count, capacity,
                                       stream_ptr, 6);
}

extern "C" long long kvq_splitzip_bf16_decode_fixed6_blocks(
    unsigned long long src_ptr, unsigned long long encoded_bytes,
    unsigned long long dst_ptr, unsigned long long count_words,
    unsigned long long block_count, unsigned long long block_words,
    unsigned long long plane_stride, unsigned long long block_stride,
    unsigned long long block_start, unsigned long long block_ids_ptr,
    unsigned long long stream_ptr) {
  if (src_ptr == 0 || dst_ptr == 0 || count_words == 0 || block_count == 0 ||
      block_words == 0 || count_words != 2 * block_count * block_words) {
    return 0;
  }
  unsigned long long expected =
      65 + count_words + ((count_words + 3) / 4) * 3;
  if (encoded_bytes != expected) return 0;
  auto* src = reinterpret_cast<const unsigned char*>(src_ptr);
  auto* dst = reinterpret_cast<uint16_t*>(dst_ptr);
  auto* block_ids = reinterpret_cast<const int64_t*>(block_ids_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  DecodeFixed6<<<static_cast<unsigned int>((count_words + 255) / 256), 256, 0,
                 stream>>>(src, dst, count_words, block_count, block_words,
                           plane_stride, block_stride, block_start, block_ids);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return Fail("DecodeFixed6", status);
  return static_cast<long long>(count_words * sizeof(uint16_t));
}

extern "C" long long kvq_splitzip_bf16_decode_fixed6(
    unsigned long long src_ptr, unsigned long long encoded_bytes,
    unsigned long long dst_ptr, unsigned long long count_words,
    unsigned long long block_count, unsigned long long block_words,
    unsigned long long plane_stride, unsigned long long block_stride,
    unsigned long long block_start, unsigned long long stream_ptr) {
  return kvq_splitzip_bf16_decode_fixed6_blocks(
      src_ptr, encoded_bytes, dst_ptr, count_words, block_count, block_words,
      plane_stride, block_stride, block_start, 0, stream_ptr);
}

extern "C" long long kvq_splitzip_bf16_encode_top16(
    unsigned long long src_ptr, unsigned long long dst_ptr,
    unsigned long long count, unsigned long long capacity,
    unsigned long long tables_ptr, unsigned long long stream_ptr) {
  if (src_ptr == 0 || dst_ptr == 0 || tables_ptr == 0 || count == 0 ||
      count > 0xFFFFFFFFULL) {
    return 0;
  }
  unsigned long long code_bytes = (count + 1) / 2;
  unsigned int escape_capacity = static_cast<unsigned int>((count + 199) / 200);
  unsigned long long encoded = 5 + count + code_bytes + escape_capacity * 5ULL;
  unsigned long long scratch_offset = (encoded + 3) & ~3ULL;
  if (encoded >= count * 2 || scratch_offset + 2 * sizeof(unsigned int) > capacity) {
    return 0;
  }
  auto* src = reinterpret_cast<const uint16_t*>(src_ptr);
  auto* dst = reinterpret_cast<unsigned char*>(dst_ptr);
  auto* tables = reinterpret_cast<const unsigned char*>(tables_ptr);
  auto* escape_count = reinterpret_cast<unsigned int*>(dst + scratch_offset);
  auto* overflow = escape_count + 1;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  InitTop16<<<1, 1, 0, stream>>>(dst, escape_count, overflow);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return Fail("InitTop16", status);
  EncodeTop16<<<1024, 256, 0, stream>>>(src, count, dst, tables, escape_count,
                                        overflow, escape_capacity);
  status = cudaGetLastError();
  if (status != cudaSuccess) return Fail("EncodeTop16", status);
  FinalizeTop16<<<1, 1, 0, stream>>>(dst, escape_count, overflow,
                                     escape_capacity);
  status = cudaGetLastError();
  if (status != cudaSuccess) return Fail("FinalizeTop16", status);
  return static_cast<long long>(encoded);
}

extern "C" long long kvq_splitzip_bf16_decode_top16_blocks(
    unsigned long long src_ptr, unsigned long long encoded_bytes,
    unsigned long long dst_ptr, unsigned long long count_words,
    unsigned long long block_count, unsigned long long block_words,
    unsigned long long plane_stride, unsigned long long block_stride,
    unsigned long long block_start, unsigned long long block_ids_ptr,
    unsigned long long tables_ptr, unsigned long long stream_ptr) {
  if (src_ptr == 0 || dst_ptr == 0 || tables_ptr == 0 || count_words == 0 ||
      block_count == 0 || block_words == 0 ||
      count_words != 2 * block_count * block_words) {
    return 0;
  }
  unsigned int escape_capacity =
      static_cast<unsigned int>((count_words + 199) / 200);
  unsigned long long expected =
      5 + count_words + (count_words + 1) / 2 + escape_capacity * 5ULL;
  if (encoded_bytes != expected) return 0;
  auto* src = reinterpret_cast<const unsigned char*>(src_ptr);
  auto* dst = reinterpret_cast<uint16_t*>(dst_ptr);
  auto* block_ids = reinterpret_cast<const int64_t*>(block_ids_ptr);
  auto* tables = reinterpret_cast<const unsigned char*>(tables_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  DecodeTop16<<<1024, 256, 0, stream>>>(
      src, dst, count_words, block_count, block_words, plane_stride,
      block_stride, block_start, block_ids, tables);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return Fail("DecodeTop16", status);
  unsigned int blocks = (escape_capacity + 255) / 256;
  if (blocks > 1024) blocks = 1024;
  ApplyTop16Escapes<<<blocks, 256, 0, stream>>>(
      src, dst, count_words, block_count, block_words, plane_stride,
      block_stride, block_start, block_ids, escape_capacity);
  status = cudaGetLastError();
  if (status != cudaSuccess) return Fail("ApplyTop16Escapes", status);
  return static_cast<long long>(count_words * sizeof(uint16_t));
}

extern "C" long long kvq_splitzip_bf16_decode_top16(
    unsigned long long src_ptr, unsigned long long encoded_bytes,
    unsigned long long dst_ptr, unsigned long long count_words,
    unsigned long long block_count, unsigned long long block_words,
    unsigned long long plane_stride, unsigned long long block_stride,
    unsigned long long block_start, unsigned long long tables_ptr,
    unsigned long long stream_ptr) {
  return kvq_splitzip_bf16_decode_top16_blocks(
      src_ptr, encoded_bytes, dst_ptr, count_words, block_count, block_words,
      plane_stride, block_stride, block_start, 0, tables_ptr, stream_ptr);
}
