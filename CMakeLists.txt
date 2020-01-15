# Required version of CMake
cmake_minimum_required(VERSION 3.0)

# Project title
project(soapBubble)

set(CMAKE_BUILD_TYPE Release)
include_directories(./ ./include/ ${PROJECT_SOURCE_DIR}/)
include_directories(cuda_util_headers)
include_directories(opencv_headers)
link_directories(./lib)
# for libz
link_directories(/usr/lib/x86_64-linux-gnu/)
link_directories(/lib/x86_64-linux-gnu/)
include_directories(${ZLIB_INCLUDE_DIRS})
link_directories(${ZLIB_LIBRARIES})

set(CMAKE_C_COMPILER /usr/bin/cc)
set(CMAKE_CXX_COMPILER  /usr/bin/c++)
# set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++0x")
set(CMAKE_CXX_FLAGS "-std=c++0x")

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})
link_directories(/opt/cuda/lib64)
set(CUDA_HOST_COMPILER  /usr/bin/cc)
set(CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math -gencode arch=compute_61,code=sm_61;)
message("arch=61")

find_package(Boost COMPONENTS filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

cuda_add_executable(soapBubble
  ./kernel/bubble.cu
  ./kernel/initializer.cu
  ./kernel/quantity.cu
  ./kernel/solver.cu
  ./kernel/timer.cu
  ./kernel/main.cu
  )

target_link_libraries(soapBubble
  opencv_core
  opencv_highgui
  opencv_imgproc
  z
  cudart
  cusparse
  cublas
  cudadevrt
  cufft
  amgxsh
  ${Boost_LIBRARIES}
  )
