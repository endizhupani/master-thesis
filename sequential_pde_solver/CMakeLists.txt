cmake_minimum_required(VERSION 3.16)
set(project_name sequential_pde_solver)
project(${project_name})

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)

set(executable main.cpp)

set(executable_name ${project_name}.exe)

### get name of current example
add_executable(${executable_name} ${executable})
target_link_libraries(${executable_name} PUBLIC ${OpenMP_CXX_LIBRARIES})