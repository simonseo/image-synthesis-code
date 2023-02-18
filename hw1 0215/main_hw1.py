# (16-726): Project 1 starter Python code
# credit to https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj1/data/colorize_skel.py
# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images



import numpy as np
import skimage as sk
import skimage.io as skio
import os
from functools import partial

# user-defined modules
from utils import get_channels, shift_image, crop, scale, show
from align import ncc_align, ncc_mirror_align, ncc_fill_align, ssd_align, oneshot_align, iterative_align



# stack_algo = partial(oneshot_align, align=ssd_align, hrange=(-10,10), wrange=(-5,40))
stack_algo = partial(iterative_align, align=ncc_align, depth=-1)
output_file_prefix = 'ncc_pyramid_crop=0.1'
crop_factor = 0.1 # 0.30
scale_factor = 1 # 1/8

for imfname in os.listdir('data'):
    # name of the input file
    impath = f'data/{imfname}'
    imtitle = os.path.splitext(imfname)[0]
    # print(imtitle)

    # read in the image
    im = skio.imread(impath)
    r_original,g_original,b_original = r,g,b = get_channels(im)
    if 0 < scale_factor < 1:
        r_original,g_original,b_original = r,g,b = [scale(c, factor=scale_factor) for c in (r,g,b)]
    if 0 < crop_factor < 1:
        r,g,b = [crop(c, factor=crop_factor) for c in (r,g,b)]

    # get image pyramids
    r_shift, g_shift = stack_algo(r,g,b)
    r_shifted = shift_image(r_original, r_shift, shift_fn=np.roll)
    g_shifted = shift_image(g_original, g_shift, shift_fn=np.roll)
    im_out = crop(np.dstack([r_shifted, g_shifted, b_original]), factor=0.8)
    
    
    # save the image
    output_dir = f'output/{output_file_prefix}/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fname = os.path.join(output_dir, f'{imtitle}.jpg')

    skio.imsave(fname, im_out)
    with open(os.path.join(output_dir, f'shifts.txt'), 'a') as f:
        f.write(f'\t{r_shift=}\t{g_shift=}\t{imtitle}\n')

    # display the image
    show(im_out)