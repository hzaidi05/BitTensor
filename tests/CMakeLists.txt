# Add test executable
add_executable(packed_tensor_test
    packed_tensor_test.cpp
)

# Link test executable with required libraries
target_link_libraries(packed_tensor_test
    bittensor
    ${TORCH_LIBRARIES}
    gtest
    gtest_main
)

# Add test
add_test(NAME packed_tensor_test COMMAND packed_tensor_test) 