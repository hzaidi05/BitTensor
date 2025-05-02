cmake_minimum_required(VERSION 3.18)
project(BitTensor VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set libtorch path
set(CMAKE_PREFIX_PATH /usr/local/libtorch)

# Find required packages
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# Create interface library for header-only library
add_library(bittensor INTERFACE)
target_include_directories(bittensor INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(bittensor INTERFACE
    ${TORCH_LIBRARIES}
)

# Add tests
enable_testing()
add_subdirectory(tests/cpp)

# Install targets
install(TARGETS bittensor
    EXPORT bittensor-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)
