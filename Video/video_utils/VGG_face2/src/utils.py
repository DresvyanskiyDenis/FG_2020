import PIL
import numpy as np
mean=(91.4953, 103.8827, 131.0912)

def load_data(path=''):

    img = PIL.Image.open(path)
    im_shape = np.array(img.size)    # in the format of (width, height, *)
    img = img.convert('RGB')
    x = np.array(img)  # image has been transposed into (height, width)
    x = x[:, :, ::-1] - mean
    return x

