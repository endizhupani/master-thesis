cmake_minimum_required(VERSION 3.15)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(project_name mpi_cuda_solver)
project(${project_name})
enable_language(CUDA CXX)

set(executable ${PROJECT_SOURCE_DIR}/app/main.cu)
set(executable_name ${project_name}.exe)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(${executable_name} ${executable} ${SOURCES})
target_include_directories(${executable_name} PUBLIC "${PROJECT_SOURCE_DIR}/include")

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER_ID)
message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER_ID}")
    enable_language(CUDA CXX)
    set( CMAKE_CUDA_FLAGS_GENCODE_FLAGS --gpu-architecture=compute_75 --gpu-code=sm_75 -use_fast_math )
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endif()

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(${executable_name} PUBLIC MPI::MPI_CXX)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${executable_name} PUBLIC OpenMP::OpenMP_CXX)
endif()

