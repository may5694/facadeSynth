cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_PREFIX_PATH /usr/local/libtorch)
set(CMAKE_C_COMPILER /usr/bin/gcc-5)
set(CMAKE_CXX_COMPILER /usr/bin/g++-5)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_HOST_COMPILER /usr/bin/gcc-5)

project(facadeSynth)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GDAL REQUIRED)

set(SOURCES
	src/main.cpp
	src/satellite.cpp
	src/building.cpp
	src/dn_predict.cpp)

add_executable(facadeSynth ${SOURCES})
target_link_libraries(facadeSynth ${TORCH_LIBRARIES})
target_link_libraries(facadeSynth ${OpenCV_LIBS})
target_link_libraries(facadeSynth ${GDAL_LIBRARY})
target_link_libraries(facadeSynth stdc++fs)

install(TARGETS facadeSynth DESTINATION .)
