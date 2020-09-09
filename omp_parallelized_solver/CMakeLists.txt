cmake_minimum_required(VERSION 3.16)
set(project_name omp_parallelized_solver)
project(${project_name})

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)

set(executable main.cpp)

set(executable_name ${project_name})
#
#set(OPENMP_FLAGS "")
#
#foreach (flag ${OpenMP_CXX_FLAGS})
#    string(APPEND OPENMP_FLAGS " ${flag} ")
#endforeach (flag ${OpenMP_CXX_FLAGS})

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

### get name of current example
add_executable(${executable_name} ${executable})
#target_compile_options(${executable_name} PUBLIC ${OPENMP_FLAGS})
target_link_libraries(${executable_name} PUBLIC ${OpenMP_CXX_LIBRARIES})