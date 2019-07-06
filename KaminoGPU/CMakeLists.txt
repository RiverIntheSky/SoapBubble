# Required version of CMake
cmake_minimum_required(VERSION 3.0)

# Project title
project(Kamino)

include_directories(./ ./include/ ${PROJECT_SOURCE_DIR}/)
include_directories(cuda_util_headers)
include_directories(opencv_headers)
include_directories(partio_headers)
link_directories(./lib)
# for libz
link_directories(/usr/lib/x86_64-linux-gnu/)
link_directories(/lib/x86_64-linux-gnu/)
include_directories(${ZLIB_INCLUDE_DIRS})
link_directories(${ZLIB_LIBRARIES})

set(CMAKE_C_COMPILER /usr/bin/gcc-6)
set(CMAKE_CXX_COMPILER  /usr/bin/g++-6)
# set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++0x")
set(CUDA_HOST_COMPILER  /usr/bin/g++-6)
find_package(CUDA REQUIRED)
include_directories(/opt/cuda/include)
include_directories(/opt/cuda/samples/common/inc)
link_directories(/opt/cuda/lib64)
set(CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math -gencode arch=compute_61,code=sm_61;)
message("arch=61")

option(BUILD_TESTS "Build the tests" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

cuda_add_executable(kamino
  ./kernel/KaminoCore.cu
  ./kernel/KaminoInitializer.cu
  ./kernel/KaminoParticles.cu
  ./kernel/KaminoQuantity.cu
  ./kernel/KaminoSolver.cu
  ./kernel/KaminoTimer.cu
  ./kernel/main.cu
  ./kernel/tdm.cu
  ./kernel/vectorUtil.cu
  )
target_link_libraries(kamino
  opencv_core
  opencv_highgui
  opencv_imgproc
  partio
  z
  cudart
  cudadevrt
  cufft
  )
