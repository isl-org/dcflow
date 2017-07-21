cmake_minimum_required(VERSION 3.2)
project(dcflow)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_definitions(/DMATLAB_MEX_FILE)
find_package(Matlab REQUIRED)

# Build DC Flow
add_subdirectory(src_cl)
