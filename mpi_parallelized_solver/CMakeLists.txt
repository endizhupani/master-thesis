cmake_minimum_required(VERSION 3.16)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(project_name mpi_parallelized_solver)
project(${project_name})

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)


set(executable ${PROJECT_SOURCE_DIR}/app/main.cpp)

set(executable_name ${project_name}.exe)
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

#add_library(Helpers mysqrt.cxx)
file(GLOB_RECURSE SOURCES "src/*.cpp")
### get name of current example
add_executable(${executable_name} ${executable} ${SOURCES})
target_include_directories(${executable_name} PUBLIC 
    "${MPI_CXX_INCLUDE_DIRS}"
    "${PROJECT_SOURCE_DIR}/include")

target_compile_definitions(${executable_name} PUBLIC 
    "${MPI_CXX_COMPILE_DEFINITIONS}")
target_compile_options(${executable_name} PUBLIC 
    "${MPI_CXX_COMPILE_FLAGS}")
target_link_libraries(${executable_name} PUBLIC 
    "${MPI_CXX_LINK_FLAGS}"
    "${MPI_CXX_LIBRARIES}")
target_link_libraries(${executable_name} PUBLIC 
    "${OpenMP_CXX_LIBRARIES}")
