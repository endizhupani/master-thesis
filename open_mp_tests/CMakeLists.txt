cmake_minimum_required(VERSION 3.16)
project(open_mp_tests)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)

#include_directories(/usr/local/cuda-11.0)
#include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
#include_directories(/usr/include)
#include_directories(/usr/local/include)

add_executable(open_mp_tests ./integrate_example/integrate_example.cpp)
target_link_libraries(open_mp_tests PUBLIC ${OpenMP_CXX_LIBRARIES})
