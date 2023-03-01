# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr




def toy_recon(image):
    imh, imw = image.shape
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    A = np.zeros((imh*imw*2+1, imh*imw))
    e = 0
    s = image
    b = np.zeros(imh*imw*2+1)
    
    # objective 3: first pixel should be equal
    A[e, im2var[0, 0]] = 1
    b[e] = s[0, 0]

    for y in range(imh-1):
        for x in range(imw-1):
            # objective 1: x gradients should be same
            e += 1
            var1 = im2var[y, x + 1]
            var2 = im2var[y, x]
            A[e, var1] = 1
            A[e, var2] = -1
            b[e] = s[y, x + 1] - s[y, x]

            # objective 2: y gradients should be same
            e += 1
            var1 = im2var[y+1, x]
            var2 = im2var[y, x]
            A[e, var1] = 1
            A[e, var2] = -1
            b[e] = s[y+1, x] - s[y, x]

    # sparse matrix optimization
    A = lil_matrix(A)
    solution = lsqr(A, b)
    print(solution[1:])
    return solution[0].reshape(imh, imw)


def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    assert fg.shape == bg.shape, "source/(fg) and target(bg) have different shape"
    assert fg.shape[:2] == mask.shape[:2], "source(fg) and mask have different size"
    imh, imw, imc = fg.shape

    def neighbours(y, x):
        return [(y+dy, x+dx) for (dy, dx) in ((0,1),(1,0))]
        
    def in_mask(y,x, mask):
        return mask[y,x,0]
    
    def construct_conditions(s, S_y, S_x, t):
        im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
        A = lil_matrix((imh*imw*2+1, imh*imw))
        b = np.zeros(imh*imw*2+1)
        e = 0
        for y, x in zip(S_y, S_x):
            for ny, nx in neighbours(y,x):
                e += 1
                A[e, im2var[y, x]] = 1
                if in_mask(ny, nx, mask): # v should match s 
                    A[e, im2var[ny, nx]] = -1
                    b[e] = s[y, x] - s[ny, nx]
                else: # v should match t + s'
                    b[e] = s[y, x] - s[ny, nx] + t[ny, nx]
        return A, b

    def solve(A,b):
        # sparse matrix optimization
        solution = lsqr(A, b)
        print(solution[1:])
        v = solution[0].reshape(imh, imw)
        return v
    
    def channelwise_blending(fg, bg, mask, c):
        S_y, S_x, _ = np.where(mask)
        A, b = construct_conditions(fg[:,:,c], S_y, S_x, bg[:,:,c]) # 0.4 seconds
        v = solve(A,b) # 3.5 seconds
        return v* mask[:,:,0] + bg[:,:,c] * (1 - mask[:,:,0])
    
    return np.stack([channelwise_blending(fg, bg, mask, c) for c in range(imc)], axis=2)



def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    return fg * mask + bg * (1 - mask)


def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    return np.zeros_like(rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")
    parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    # Example script: python proj2_starter.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255.
        image_hat = toy_recon(image)

        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.show()

    # Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = poisson_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.show()

    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = mixed_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
