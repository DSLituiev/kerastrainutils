#!/usr/bin/env python3
import numpy as np
#from image_gen_extended import histeq, threshold_wi_range
from PIL import Image
import sys

def _get_histogram_thr_(img, qmin=0.3, qmax=0.7, bins=25):

    lq = (qmin)/2.0
    uq = 1 - (1-qmax)/2.0
    #print(lq, uq)
    lthr, uthr = np.percentile(img.ravel(), np.r_[lq, uq]*100.0)
    mask0 = (img.ravel() < uthr) & (img.ravel()>lthr)

    yh, xh = np.histogram(img.ravel()[mask0], bins=bins)
    cumh = np.cumsum(yh)/sum(yh)
    cumh -= cumh[0]
    cumh /= cumh[-1]
    #
    cumh += lq
    cumh *= uq
    mask = (cumh > qmin) & (cumh < qmax)
    try:
        thr_ind = np.argmax(mask) + np.argmin(yh[mask])
    except ValueError as ee:
        print("shape", img.shape)
        print("mask", sum(mask))
        print("cumh", cumh)
        print("max", max(img.ravel()))
        raise ee

    thr = xh[thr_ind]
    return thr


def threshold_wi_range(img, qmin=0.3, qmax=0.7, bins=25):
    """threshold an image, assuming that qmin to qmax fraction of pixels are below the threshold
    qmin"""
    try:
        thr = _get_histogram_thr_(img, qmin=qmin, qmax=qmax, bins=bins)
    except ValueError as ee:
        thr = _get_histogram_thr_(img, qmin=qmin, qmax=qmax, bins=bins*64)
    return np.maximum(0, img-thr)


def get_cdf_vector(x, bitdepth=16, alpha=None):
    dtype = 'uint%d' % bitdepth
    hist = np.bincount(x.ravel())
    cdf = hist.cumsum()
    #cdf = np.asarray(hist.cumsum(), dtype='float32')
    del hist
    #cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0).astype('float32')
    del cdf
    #cdf_m = (cdf_m - cdf_m.min())*(2**bitdepth -1)/(cdf_m.max()-cdf_m.min())
    cdf_min = cdf_m.min()
    cdf_max = cdf_m.max()
    cdf_m -= cdf_min
    #print("cdf_m", cdf_m.dtype)
    if (cdf_max-cdf_min)>0:
        cdf_m *= float( (2**bitdepth -1)/(cdf_max-cdf_min) )
    #cdf_m = cdf_m * float( (2**bitdepth -1)/(cdf_max-cdf_min) )
    cdf = np.round(np.ma.filled(cdf_m, 0)).astype(dtype)
    if alpha is not None and alpha>0:
        cdf = (1-alpha)*cdf + (alpha)*np.arange(len(cdf))
    return cdf

def histeq(img, bitdepth = 32, mask=None, alpha=None):
    """ histogram equalization
    alpha -- interpolate between equalized and original intensites
             (0 -- no equalization, 1 -- full equalization)
    """
    if str(img.dtype).startswith('float'):
        dtype = 'uint%d' % bitdepth
        img = np.asarray(img*((2**bitdepth)/max(1.0,img.max())),
                         dtype=dtype)
    else:
        dtype = img.dtype

    if mask is None:
        cdf = get_cdf_vector(img, bitdepth=bitdepth, alpha=alpha)
        return cdf[img]
    else:
        cdf = get_cdf_vector(np.asarray(img[mask], dtype=dtype, alpha=alpha),
                    bitdepth=bitdepth, dtype=dtype)
        return cdf[img] * mask + img*(~mask)


def ztransform(img, mask=None,
               contrast=None, brightness=None,
               truncate_quantile = None,
               max_std = 2**-7):
    if mask is not None:
        imgvals = img[mask].ravel()
    else:
        imgvals = img.ravel()

    if truncate_quantile:
        ql, qu = np.percentile(imgvals, 
                100*np.r_[truncate_quantile, 1-truncate_quantile])
        #print("ql, qu", ql, qu)
        imgvals = imgvals[(imgvals>ql) & (imgvals<qu)]

    mean = np.mean(imgvals)
    std = np.std(imgvals)
    std = np.maximum(std, max_std)

    img_out = (img - mean)/std
    if contrast:
        img_out = img_out * contrast
    if brightness:
        img_out = img_out + brightness

    return np.asarray(img_out, dtype = img.dtype)

    dtype = 'uint%d' % bitdepth
if __name__ == '__main__':
    fn = sys.argv[1]
    if len(sys.argv)>2:
        thr = sys.argv[2]
    else:
        thr = False
    frmt = 'png'

    fnout = fn.replace( '.'+frmt, '') + '.histeq16bit.' + frmt

    im = np.asarray(Image.open(fn), dtype='float32')

    if thr:
        im = threshold_wi_range(im)

    im = histeq(im, bitdepth=16).astype('float32')

    im = Image.fromarray(im, mode='F').convert('I')
    print(np.asarray(im).max())
    im.save(fnout, "png", quality=100, optimize=True)
    #im.save(fnout)
    print(fnout, file=sys.stderr)
