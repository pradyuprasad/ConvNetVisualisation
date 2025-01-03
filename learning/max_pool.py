import numpy as np

array = np.arange(16).reshape((4, 4))

def max_pool(image: np.ndarray) -> np.ndarray:
    '''
    assume image is 4x4 and we are doing a 2x2 thing
    '''
    output = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            output[i, j] = image[i*2:i*2+2, j*2:j*2+2].max()


    assert output.shape == (2, 2)
    return output



pooled = max_pool(array)
print(pooled)
