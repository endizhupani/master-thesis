# cmake_minimum_required(VERSION 3.18)
# set(CMAKE_VERBOSE_MAKEFILE ON)
# set(project_name mpi_cuda_solver)
# project(${project_name})
# enable_language(CUDA CXX)
# set(CMAKE_CUDA_HOST_FLAGS " -Xcompiler ")
# set(CMAKE_CUDA_HOST_LINKER_FLAGS " -Xlinker ")
# set(executable ${PROJECT_SOURCE_DIR}/app/main.cu)
# set(executable_name ${project_name}.exe)

# file(GLOB_RECURSE SOURCES "src/*.cpp")
# add_executable(${executable_name} ${executable} ${SOURCES})
# target_include_directories(${executable_name} PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/lib/hemi" "${PROJECT_SOURCE_DIR}/lib/hemi/range")

# include(CheckLanguage)
# check_language(CUDA)
# if (CMAKE_CUDA_COMPILER_ID)
# message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER_ID}")
#     enable_language(CUDA CXX)
#     set( CMAKE_CUDA_FLAGS_INCLUDE --pre-include=hemi.h -use_fast_math)
#     if(NOT DEFINED CMAKE_CUDA_STANDARD)
#         set(CMAKE_CUDA_STANDARD 11)
#         set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#     endif()
#     set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
#     set_property(TARGET ${executable_name} PROPERTY CUDA_ARCHITECTURES 61 75 80)
#     target_include_directories(${executable_name} PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
#     target_compile_options(${executable_name} PRIVATE ${CMAKE_CUDA_FLAGS_INCLUDE} )
# endif()

# find_package(MPI REQUIRED)
# message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
# target_link_libraries(${executable_name} PUBLIC MPI::MPI_CXX)

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(${executable_name} PUBLIC OpenMP::OpenMP_CXX)
# endif()

# ================================================ Version that works with Palma =======================================================

cmake_minimum_required (VERSION 3.10.2)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(project_name mpi_cuda_solver)
project(${project_name})
enable_language(CUDA CXX)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_HOST_FLAGS " -Xcompiler ")
set(CMAKE_CUDA_HOST_LINKER_FLAGS " -Xlinker ")

set(CMAKE_CXX_FLAGS_BENCHMARKPALMA "-O3 -DNDEBUG -march=broadwell -DMPICH_IGNORE_CXX_SEEK -std=c++14")

string(REPLACE " " "," MPI_CXX_LINK_FLAG ${MPI_CXX_LINK_FLAGS})
    
foreach (flag ${MPI_CXX_COMPILE_OPTIONS})
  string(APPEND CMAKE_CUDA_HOST_FLAGS ",${flag}")
endforeach (flag ${MPI_CXX_COMPILE_OPTIONS})
foreach (flag ${MPI_CXX_LINK_FLAG})
  string(APPEND CMAKE_CUDA_HOST_LINKER_FLAGS ",${flag}")
endforeach (flag ${MPI_CXX_LINK_FLAG})

foreach (flag ${OpenMP_CXX_FLAGS})
  string(APPEND CMAKE_CUDA_HOST_FLAGS ",${flag}")
  string(APPEND CMAKE_CUDA_HOST_LINKER_FLAGS ",${flag}")
endforeach (flag ${OpenMP_CXX_FLAGS})

string(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_HOST_FLAGS})
string(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_HOST_LINKER_FLAGS})

set( CMAKE_CUDA_FLAGS_BENCHMARKPALMA "-O3 -arch=compute_35 --expt-extended-lambda -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3 -Xcompiler -O3,-DNDEBUG,-march=broadwell,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")

# add muesli libs

# set(MUESLI_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/muesli-${PROJECT_VERSION})
# set(MUESLI_INCLUDE_INSTALL_DIR ${MUESLI_INSTALL_DIR}/include)
# set(MUESLI_INCLUDE_DETAIL_INSTALL_DIR ${MUESLI_INCLUDE_INSTALL_DIR}/detail)

## add_library(muesli STATIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/detail ${PROJECT_SOURCE_DIR}/src)

##  $<BUILD_INTERFACE:${MPI_CXX_INCLUDE_PATH}>
##  $<INSTALL_INTERFACE:${MUESLI_INCLUDE_INSTALL_DIR}>
##  $<INSTALL_INTERFACE:${MUESLI_INCLUDE_INSTALL_DIR}/detail>

## install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src DESTINATION ${MUESLI_INSTALL_DIR})

## set(Example examples/da_test.cu)
## set(Example examples/mandelbrotDA.cu)

### get name of current example
set(executable ${PROJECT_SOURCE_DIR}/app/main.cu)
set(executable_name ${project_name}.exe)
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(${executable_name} ${executable} ${SOURCES})

target_compile_features(${executable_name} PUBLIC cxx_std_14)
target_include_directories(${executable_name} 
  PUBLIC ${MPI_CXX_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/cache
  PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(${executable_name} PUBLIC ${MPI_CXX_COMPILE_DEFINITIONS})
target_compile_options(${executable_name} PUBLIC )
target_link_libraries(${executable_name} PUBLIC ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
