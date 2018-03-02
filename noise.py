
# coding: utf-8
# !conda install -y opencv
import numpy as np
import cv2


def add_noise(image, mode, mean = 0,
             sigma = 0.1, s_vs_p = 0.5,
            num=None, random_state = None):
    """Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    ndim = len(image.shape)
    if ndim<3:
        image = image[...,np.newaxis]        
    if mode.startswith("gaus"):
        row,col,ch= image.shape
        gauss = random_state.normal(mean,sigma,(row,col,ch))
#         gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
    elif mode == "s&p":
        row,col,ch = image.shape
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(sigma * image.size * s_vs_p)
        coords = [random_state.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        noisy[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(sigma* image.size * (1. - s_vs_p))
        coords = [random_state.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        noisy[coords] = 0
    elif mode == "poisson":
        if num is not None:
            mean = num/image.size
        if mean == 0.0:
            mean = sigma
        noisy = random_state.poisson(mean, size=image.shape)
    elif mode.startswith("spec"):
        row,col,ch = image.shape
        gauss = random_state.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)          
        noisy = image + image * gauss
    else:
        raise ValueError("Unkown noise mode: %s" % mode)
    if ndim >= 3:
        return noisy
    else:
        return noisy[:,:,0]

    
def add_blurred_noise(im, mode= 'poisson', scale=1, blur_sigma=1, 
		      mean=.008, **kwargs):

    ndim = len(im.shape)
    if ndim<3:
        im = im[...,np.newaxis]        
    nmask = np.zeros_like(im, np.uint8)
    nmask_noisy  = scale*add_noise(im, mode=mode, mean=mean, **kwargs)
    
    ksize = 2*(int(blur_sigma*2)//2) +1
    nmask_noisy_blurred = cv2.GaussianBlur(nmask_noisy.astype(float), (ksize, ksize), 1.0)
    
    noisy = (im + nmask_noisy_blurred).astype(im.dtype)
    if ndim >= 3:
        return noisy
    else:
        return noisy[:,:,0]

