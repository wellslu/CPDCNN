#include <iostream>
#include <gtest/gtest.h>
#include "util.h"

TEST(tensor_transformation, reshape_input_tensor) {
    // Set filter size
    int filter_h = 2;
    int filter_w = 2;

    // Create an input tensor [C=1, H=3, W=3] with values from 1 to 9
    torch::Tensor input = torch::arange(1, 1 * 3 * 3 + 1).view({1, 3, 3}).to(torch::kFloat);

    // Extract input dimensions
    int C = input.size(0);  // Number of channels
    int H = input.size(1);  // Height
    int W = input.size(2);  // Width

    // Call the tensor_transformation function
    torch::Tensor output = tensor_transformation(input, filter_h, filter_w);

    // Expected output dimensions
    int H_new = H - filter_h + 1;  
    int W_new = W - filter_w + 1;
    std::vector<int64_t> expected_shape = {H_new, W_new, filter_h, filter_w, C};

    // Test the shape of the output tensor
    EXPECT_EQ(output.sizes().vec(), expected_shape) << "Output shape mismatch.";

    // Expected output tensor values (manually computed patches)
    torch::Tensor expected_output = torch::tensor(
        {
            {{{{1, 2}, {4, 5}}, {{2, 3}, {5, 6}}},  // Patches in (hi, wi)
             {{{4, 5}, {7, 8}}, {{5, 6}, {8, 9}}}},
        },
        torch::kFloat
    );

    // Flatten both tensors for comparison
    torch::Tensor flattened_output = output.flatten();
    torch::Tensor flattened_expected = expected_output.flatten();

    EXPECT_EQ(flattened_output.sizes(), flattened_expected.sizes())
        << "Flattened shape mismatch.";

    // Assert values
    EXPECT_TRUE(torch::allclose(flattened_output, flattened_expected))
        << "Flattened tensor values mismatch.\nExpected:\n"
        << flattened_expected << "\nActual:\n"
        << flattened_output;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}