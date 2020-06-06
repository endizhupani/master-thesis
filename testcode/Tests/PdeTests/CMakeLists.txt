# CMakeList.txt : CMake project for PdeTests, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
add_executable (PdeTests "PdeTests.cpp" "PdeTests.h" "complex.h" "complex.cpp" "point.cpp" "vector.cpp")

# TODO: Add tests and install targets if needed.
