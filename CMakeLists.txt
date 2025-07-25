IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(rnnt_release)

IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
ENDIF()

IF (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    add_definitions(-DAPPLE)
ENDIF()

include_directories(include)

FIND_PACKAGE(CUDA)
MESSAGE(STATUS "cuda found ${CUDA_FOUND}")

option(USE_NAIVE_KERNEL "use naive alpha-beta kernel" OFF)
option(DEBUG_TIME "output kernel time" OFF)
option(DEBUG_KERNEL "output alpha beta" OFF)
if (USE_NAIVE_KERNEL)
    add_definitions(-DUSE_NAIVE_KERNEL)
endif()
if (DEBUG_TIME)
    add_definitions(-DDEBUG_TIME)
endif()
if (DEBUG_KERNEL)
    add_definitions(-DDEBUG_KERNEL)
endif()

option(WITH_GPU "compile warp-rnnt with cuda." ${CUDA_FOUND})
option(WITH_OMP "compile warp-rnnt with openmp." ON)

if(NOT WITH_OMP)
    add_definitions(-DRNNT_DISABLE_OMP)
endif()
if (WITH_OMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
endif()

# CUDA architecture flags for broad compatibility (Maxwell, Pascal, Volta, Turing, Ampere, Ada)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    # List all supported architectures explicitly
    set(CMAKE_CUDA_ARCHITECTURES 50;52;60;61;62;70;75;80;86;89)
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_89,code=sm_89")
endif()

if (NOT APPLE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
ENDIF()


IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


IF (WITH_GPU)

    MESSAGE(STATUS "Building shared library with GPU support")

    CUDA_ADD_LIBRARY(warprnnt SHARED src/rnnt_entrypoint.cu)
    IF (!Torch_FOUND)
        TARGET_LINK_LIBRARIES(warprnnt ${CUDA_curand_LIBRARY})
    ENDIF()

    cuda_add_executable(test_time_gpu tests/test_time.cu tests/random.cpp )
    TARGET_LINK_LIBRARIES(test_time_gpu warprnnt ${CUDA_curand_LIBRARY})
    SET_TARGET_PROPERTIES(test_time_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    cuda_add_executable(test_gpu tests/test_gpu.cu tests/random.cpp )
    TARGET_LINK_LIBRARIES(test_gpu warprnnt ${CUDA_curand_LIBRARY})
    SET_TARGET_PROPERTIES(test_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

ELSE()
    MESSAGE(STATUS "Building shared library with no GPU support")

    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(warprnnt SHARED src/rnnt_entrypoint.cpp)

ENDIF()


add_executable(test_cpu tests/test_cpu.cpp tests/random.cpp )
TARGET_LINK_LIBRARIES(test_cpu warprnnt)
SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

add_executable(test_time tests/test_time.cpp tests/random.cpp )
TARGET_LINK_LIBRARIES(test_time warprnnt)
SET_TARGET_PROPERTIES(test_time PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

INSTALL(TARGETS warprnnt
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")

INSTALL(FILES include/rnnt.h DESTINATION "include")
