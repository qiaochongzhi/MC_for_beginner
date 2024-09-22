cmake_minimum_required(VERSION 3.4)
project(MonteCarlo)

set(CMAKE_CXX_STANDARD_VERSION 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE True)
set(CMAKE_CXX_FLAGS_RELEASE True)

set(PYTHON_EXECUTABLE "/Users/cz/anaconda3/bin/python3")
set(PYTHON_INCLUDE_DIRECTORY "/Users/cz/anaconda3/include/python3.11")

add_subdirectory(extern/pybind11)

pybind11_add_module(MonteCarlo bind.cpp MonteCarlo.cpp)
