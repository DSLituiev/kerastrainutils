import numpy as np
import sys

def crop_pad_center(img, size, pad_mode='reflect', centre=None, **kwargs):
    """crop and/or pad image to the specified size
    see np.pad for `mode`
    """
    assert len(img.shape)>1, "empty image"
    pad_seq, crop_seq = _get_crop_pad_seq_(img, size=size, centre=centre)
    # print("pad_seq", pad_seq)
    # print("img.shape", img.shape)
    try:
        img = img[crop_seq]
        out = np.pad(img, pad_seq, mode=pad_mode, **kwargs)
    except (IndexError,TypeError) as ee:
        print("img.shape", img.shape, sep='\t', file=sys.stderr)
        print("target size", size, sep='\t', file=sys.stderr)
        print("pad_seq", pad_seq, sep='\t', file=sys.stderr)
        print("crop_seq", crop_seq, sep='\t', file=sys.stderr)
        raise ee
    return out


def _get_crop_pad_seq_(img, size, centre = None):
    """a helper function for crop_pad_center
    IN :        |--------x--------|

    OUT:    |------------.----X-----------------|
            0           cin
    cout:       |<------>|

    """
    sz_in = img.shape
    if centre is None:
        centre = [x//2 for x in sz_in]
    pad_seq = []
    crop_seq = []
    for sin, sout, cin in zip(sz_in, size, centre):
        sz_diff = (sout - sin)
#         print("sz_diff", sz_diff)
        cout = sout//2
        # positive: pad
        # negative: crop
        if sz_diff<0:
            pad_seq.append((0,0))
            left_diff = cout - cin
            right_diff = cout + cin - sin
            #print("left_diff", left_diff, "right_diff", right_diff)
            if left_diff>0:
                right_diff += left_diff # -(-left_crop)
                left_diff = 0
            elif right_diff>0:
                left_diff += right_diff # -(-right_crop)
                right_diff = 0
            crop_seq.append(slice(-left_diff, sout-left_diff, 1))
        else:
            crop_seq.append(slice(None))
            cin = sin//2
            left_diff = cout - cin
            #right_diff = cout - (sin- cin) #+ 1
            right_diff = sz_diff - left_diff
            pad_seq.append((left_diff, right_diff))
    # adjust for difference in dimensionality
    ndim_diff = len(img.shape) - len(size)
    if ndim_diff>0:
        for nn in range(ndim_diff):
            crop_seq.append(slice(None))
            pad_seq.append((0,0))
    return pad_seq, crop_seq

