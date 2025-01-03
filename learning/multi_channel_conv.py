import numpy as np
def multi_conv1(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert image.shape == (3, 28, 28)
    assert kernel.shape == (3, 3, 3)
    output = np.zeros((1, 26, 26))
    for c in range(3):
        for i in range(26):
            for j in range(26):
                output[0,i,j] += np.sum(image[c, i:i+3, j:j+3] * kernel[c])
    assert output.shape == (1, 26, 26)
    return output

def multi_conv2(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert image.shape == (3, 28, 28)
    assert kernel.shape == (3, 3, 3)
    output = np.zeros((1, 26, 26))
    for i in range(26):
        for j in range(26):
            for c in range(3):
                output[0,i,j] += np.sum(image[c, i:i+3, j:j+3] * kernel[c])
    assert output.shape == (1, 26, 26)
    return output


# Create a simple test image with different values in each channel
test_image = np.zeros((3, 28, 28))
test_image[0] = 1  # red channel all 1s
test_image[1] = 2  # green channel all 2s
test_image[2] = 3  # blue channel all 3s

# Create a simple kernel with different values
test_kernel = np.zeros((3, 3, 3))
test_kernel[0] = np.ones((3, 3))  # red kernel all 1s
test_kernel[1] = np.ones((3, 3)) * 2  # green kernel all 2s
test_kernel[2] = np.ones((3, 3)) * 3  # blue kernel all 3s

# Test both functions
result1 = multi_conv1(test_image, test_kernel)
result2 = multi_conv2(test_image, test_kernel)

# Compare results
print("Are results equal?", np.allclose(result1, result2))
print("Sample output value at (0,0):")
print("Function 1:", result1[0,0,0])
print("Function 2:", result2[0,0,0])
