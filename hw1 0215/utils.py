import numpy as np
import skimage as sk
import skimage.io as skio
import os


def show(im):
    skio.imshow(im)
    skio.show()

def get_channels(im):
    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int64)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    return r,g,b

def crop(channel, factor=0.1):
    """Crop 2D image channel and leave just the center

    Args:
        channel (np.ndarray): image channel that will be cropped
        factor (float, optional): image will be cropped in both dimensions, leaving only the center. Defaults to 0.1.

    Returns:
        np.ndarray: cropped channel
    """
    h, w = channel.shape
    center = (1-factor)/2.0
    h_center = int(h*center)
    w_center = int(w*center)
    return channel[h_center:h-h_center, w_center:w-w_center] # only keep center 80%

def mirror_roll(channel, distance, axis):
    """Usage: shifted = mirror_roll(c1, center[0]+dh, axis=0)"""
    shape = channel.shape
    pad_width = [[0,0],[0,0]]
    direction = int(distance > 0)
    pad_width[axis][direction] = np.abs(distance)
    padded = np.pad(channel, pad_width, mode='reflect')
    cropped = padded.take(range(distance, shape[axis]+distance), axis=axis, mode='wrap')
    return cropped


def shift_image(channel, shift, shift_fn=np.roll):
    '''shift_fn can be np.roll or mirror_roll'''
    channel = shift_fn(channel, shift[0], 0)
    channel = shift_fn(channel, shift[1], 1)
    return channel


def resize_and_save():
    output_file_prefix = 'resized_original'

    for imfname in os.listdir('data'):
        # name of the input file
        impath = f'data/{imfname}'
        imtitle = os.path.splitext(imfname)[0]
        # print(imtitle)

        # read in the image
        im = skio.imread(impath)

        im_out = sk.transform.rescale(im, 1/4) # scale down to reduce search space

        # save the image
        output_dir = f'output/{output_file_prefix}/'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        fname = os.path.join(output_dir, f'{imtitle}.jpg')

        skio.imsave(fname, im_out)()