def ys_resize(image):
    width,height = 672,672
    import numpy as np
    resized_image = np.zeros(shape=[height,width])
    for i in range(height):
        if(i%8==0):
            for j in range(width):
                if(j%8==0):
                    resized_image[i//8][j//8] = image[i][j]
    return resized_image