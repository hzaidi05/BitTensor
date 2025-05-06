cmake_minimum_required(VERSION 3.18)
project(BitTensor VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set path to libtorch
set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")  # Update if necessary
find_package(Torch REQUIRED)

# Find Python + ABI compatibility
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE ABI_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${ABI_VERSION})
set(Torch_USE_CXX11_ABI ${ABI_VERSION})

# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Collect pybind11 include flags from Python (optional, if using system pybind11)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDE_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(PYBIND11_INCLUDE_FLAGS)

# Interface library for BitTensor
add_library(bittensor INTERFACE)
target_include_directories(bittensor INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(bittensor INTERFACE
    ${TORCH_LIBRARIES}
)

# Create Python extension module
pybind11_add_module(bittensor_py pybind/bindings.cpp)
target_link_libraries(bittensor_py PRIVATE
    bittensor
    ${TORCH_LIBRARIES}
    Python3::Python
)
target_include_directories(bittensor_py PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/../torch/csrc/api/include
    ${Python3_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_FLAGS}
)

# Add tests
enable_testing()
add_subdirectory(tests/cpp)

# Install targets
install(TARGETS bittensor bittensor_py
    EXPORT bittensor-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)
