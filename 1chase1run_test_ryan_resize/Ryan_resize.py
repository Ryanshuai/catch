def ys_resize(image):
    import numpy as np
    resized_image = np.zeros(shape=[84,84])
    for i in range(672):
        if(i%8==0):
            for j in range(672):
                if(j%8==0):
                    resized_image[i//8][j//8] = image[i][j]
    return resized_image