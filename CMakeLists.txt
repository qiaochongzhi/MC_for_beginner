cmake_minimum_required(VERSION 3.10)
project(MonteCarlo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(PYTHON_EXECUTABLE "/Users/cz/anaconda3/bin/python3")
set(PYTHON_INCLUDE_DIRECTORY "/Users/cz/anaconda3/include/python3.11")

add_subdirectory(extern/pybind11)

# add_executable(TestMonteCarlo main.cpp MonteCarlo.cpp linkList.cpp)
pybind11_add_module(MonteCarlo bind.cpp MonteCarlo.cpp linkList.cpp)
