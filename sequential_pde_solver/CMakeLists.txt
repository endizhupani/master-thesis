cmake_minimum_required(VERSION 3.15)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(project_name sequential_pde_solver)
project(${project_name})

set(CMAKE_CXX_FLAGS_BENCHMARKPALMA "-O3 -DNDEBUG -march=broadwell -DMPICH_IGNORE_CXX_SEEK -std=c++14")

##set(CMAKE_CXX_STANDARD 14)
set(executable ${PROJECT_SOURCE_DIR}/main.cpp)
##SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
##SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
##SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
set(executable_name ${project_name}.exe)

### get name of current example
add_executable(${executable_name} ${executable})
target_compile_features(${executable_name} PUBLIC cxx_std_14)  

target_compile_options(${executable_name} PUBLIC )