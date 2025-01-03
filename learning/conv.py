import numpy as np

def conv(image, kernel):
    '''
    image is a 5x5 numpy array
    kernel is a 3x3 numpy array
    '''

    output = np.zeros((3, 3))

    end = 2
    for i in range(end+1):
        for j in range(end+1):
            temp = image[i:i+3, j:j+3] * kernel
            output[i, j] = temp.sum()

    return output


test_image = np.array([[1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1]])

test_kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]])

result = conv(test_image, test_kernel)
print(result)

            
