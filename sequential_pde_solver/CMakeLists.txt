cmake_minimum_required(VERSION 3.15)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(project_name sequential_pde_solver)
project(${project_name})

set(CMAKE_CXX_STANDARD 14)
set(executable main.cpp)

set(executable_name ${project_name}.exe)

### get name of current example
add_executable(${executable_name} ${executable})