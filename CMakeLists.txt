cmake_minimum_required(VERSION 3.15)
project(CMProject VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# fix
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# find package needed by the project
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GUROBI)

# if not specified force Relase build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -O0 -fstack-check -fstack-protector-strong -fno-omit-frame-pointer -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3") # To view vectorization -ftree-vectorize -fopt-info-vec

# configure conan and add libraries
if (EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else ()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif ()

# create Neuradillo library target
add_library(Neuradillo
        STATIC
        src/layer/layer.cpp
        src/layer/layer.h
        src/network/network.cpp
        src/network/network.h
        src/preprocessing/preprocessing.cpp
        src/preprocessing/preprocessing.h
        src/preprocessing/preprocessing.cpp
        src/preprocessing/preprocessing.h
        src/activationFunction/activationFunction.h
        src/activationFunction/tanhFunction.cpp
        src/activationFunction/tanhFunction.h
        src/activationFunction/logisticFunction.cpp
        src/activationFunction/logisticFunction.h
        src/load/loadDataset.cpp
        src/load/loadDataset.h
        src/lossFunction/lossFunction.h
        src/lossFunction/meanSquaredError.cpp
        src/lossFunction/meanSquaredError.h
        src/activationFunction/reluFunction.cpp
        src/activationFunction/reluFunction.h
        src/lossFunction/binaryCrossentropy.cpp
        src/lossFunction/binaryCrossentropy.h
        src/gridSearch/gridSearch.cpp
        src/gridSearch/gridSearch.h
        src/activationFunction/linearFunction.cpp
        src/activationFunction/linearFunction.h
        src/crossValidation/crossValidation.cpp
        src/crossValidation/crossValidation.h
        src/lossFunction/meanEuclideanError.cpp
        src/lossFunction/meanEuclideanError.h
        src/gridSearch/parallelGridSearch.cpp
        src/gridSearch/parallelGridSearch.h
        src/optimizer/optimizer.h
        src/optimizer/gradientDescent.cpp
        src/optimizer/gradientDescent.h
        src/optimizer/LBFGS.cpp
        src/optimizer/LBFGS.h
        src/exception/exception.cpp
        src/exception/exception.h
        src/optimizer/ProximalBundleMethod.cpp
        src/optimizer/ProximalBundleMethod.h)

target_include_directories(Neuradillo
        PUBLIC
        ${GUROBI_INCLUDE_DIRS}
        )

target_link_libraries(Neuradillo
        ${CONAN_LIBS}
        OpenMP::OpenMP_CXX
        Threads::Threads
        optimized ${GUROBI_CXX_LIBRARY}
        debug ${GUROBI_CXX_DEBUG_LIBRARY}
        ${GUROBI_LIBRARY}
        )
set_target_properties(Neuradillo PROPERTIES
        CXX_STANDARD 17
        LINKER_LANGUAGE CXX)

# create main target
add_executable(main main.cpp)
set_target_properties(main PROPERTIES
        CXX_STANDARD 17
        LINKER_LANGUAGE CXX)
target_link_libraries(main Neuradillo)