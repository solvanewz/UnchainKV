cmake_minimum_required(VERSION 3.16)
project(unchain_kv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
find_package(Threads REQUIRED)
find_library(UNCHAIN_KV_ZLIB NAMES z libz.so.1 z.so.1 REQUIRED)

add_library(unchain_kv_tcp SHARED kvq_tcp.cc)
target_link_libraries(unchain_kv_tcp PRIVATE Threads::Threads ${UNCHAIN_KV_ZLIB})

find_path(CUDART_INCLUDE_DIR cuda_runtime_api.h
  PATHS /usr/local/cuda/include /usr/local/cuda/targets/x86_64-linux/include)
find_library(CUDART_LIBRARY cudart
  PATHS /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND CUDART_LIBRARY)
  enable_language(CUDA)
  add_library(unchain_kv_splitzip_cuda SHARED kvq_splitzip_cuda.cu)
  target_link_libraries(unchain_kv_splitzip_cuda PRIVATE ${CUDART_LIBRARY})
endif()
