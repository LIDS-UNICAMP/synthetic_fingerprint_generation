# Author: André Igor Nóbrega da Silva
# email :  andreigor008@gmail.com
# date  : 2023-09-19
# Generates a synthetic latent fingerprint database, applying contrast adjustments, gaussian blur, gaussian noise, occlusion and downsampling

import sys
import os
import random
from argparse import ArgumentParser
from functools import partial



import glob
import warnings
import wsq
import cv2
import numpy as np
from PIL import Image
from skimage.measure import block_reduce



def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def gaussian_noise(img, noise_level):
    gaussian  = np.random.normal(0, noise_level, img.shape)
    noisy_img = img + gaussian

    return noisy_img

def downsampling(img, block_size):
    down    = block_reduce(img, block_size = (block_size, block_size), func = np.mean)
    resized = cv2.resize(down, (img.shape[1], img.shape[0]), fx = 0, fy = 0, interpolation = cv2.INTER_NEAREST)
    return resized 

def gamma(image = None, value = 1):
    _max = image.max()
    image = (image / _max) ** value
    image = image * _max
    image = np.clip(image, a_min=0, a_max=_max)
    return image

def parabolic_occlusion(img, thickness_range = (5, 13), opacity_range = (0.3,0.9)):
    line_color = 0 if random.uniform(0,1) < 0.5 else 255  

    # Defining parabola parameters
    ylim, xlim = img.shape
    signal     = random.choice([-1, 1])
    a          = signal * random.uniform(0.001, 0.009)
    b          = -a * random.uniform(0.1 * xlim, 0.9 * xlim)
    c          = random.uniform(0, 2 * xlim) if signal == 1 else random.uniform(-xlim, xlim)
    x          = np.linspace(0, xlim, 1000)
    y          = np.polyval([a, b, c], x)
    draw_pts   = (np.asarray([x, y]).T).astype(int)

    # Defining drawing parameters
    thickness = random.randint(*thickness_range)
    opacity   = random.uniform(*opacity_range)


    # Drawing image
    drawn     = np.ones_like(img)
    drawn     = cv2.polylines(drawn, [draw_pts], False, (line_color,line_color,line_color), thickness)

    out       = np.where(drawn == line_color, ((opacity) * drawn + (1 - opacity) * img), img).astype(int)
    
    
    return out

def linear_occlusion(img, thickness_range = (5, 13), opacity_range = (0.3, 0.9), n_lines_range = (3, 7)):
    n_lines = random.randint(*n_lines_range)
    out     = img.copy()

    for i in range(n_lines):
        line_color = 0 if random.uniform(0,1) < 0.5 else 255  

        # Defining line parameters
        xlim, ylim = img.shape

        xstart, xend = random.sample(range(0, xlim), 2)
        ystart, yend = random.sample(range(0, ylim), 2)

        walls      = {'left': (0, ystart), 'right': (xlim, yend), 'top': (xstart, 0), 'bottom': (xend, ylim)}
        start, end = random.sample(list(walls.items()), 2)

        # Defining drawing parameters
        thickness = random.randint(*thickness_range)
        opacity   = random.uniform(*opacity_range)

        # Drawing image

        drawn = (np.ones_like(img, dtype=np.uint8))
        drawn = cv2.line(drawn, start[1], end[1], (line_color,line_color,line_color), thickness)

        out       = np.where(drawn == line_color, ((opacity) * drawn + (1 - opacity) * out), out).astype(int)

    return out


def generate_synthetic_latent(img):
    low_or_high = random.choice([0, 1])
    gamma_value = random.uniform(0.3, 0.7) if low_or_high == 0 else random.uniform(3.0, 4.0)
    noise_level = random.uniform(5, 60)
    blur_size = random.choice([num for num in range(3, 7) if num % 2 != 0])
    downsample = random.choice([num for num in range(2, 4)])

    # print(gamma_value)

    degradation_block = [
        partial(parabolic_occlusion),
        partial(linear_occlusion),
        partial(gamma, value=gamma_value),
        partial(gaussian_blur, kernel_size=blur_size),
        partial(downsampling, block_size=downsample),
        partial(gaussian_noise, noise_level=noise_level),
    ]

    out = img.copy()

    for d in degradation_block:
        out = d(out)

    return out.astype(int)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("images", type = str, help = "Folder with all fingerprint images (.png)")
    parser.add_argument("n", type = str, help = "Number of synthetic latents to generate per reference image")
    parser.add_argument("output", type = str, help = "Output folder")
    

    return parser.parse_args()

def create_output_dir(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        print('Warning. Output patches folder already exists. May overwrite files.')


def main(args):
    
    # reading input args
    images = sorted(glob.glob(args.images + '/*.png'))

    # Creating output directories
    output_dir = args.output + '/'
    
    create_output_dir(output_dir)
    # Generating synthetic latents
    for i in range(len(images)):
        if i % 10 == 0:
            print("Processing image {}/{}".format(i + 1, len(images)))
        img = np.array(Image.open(images[i]))
        for j in range(int(args.n)):
            out      = generate_synthetic_latent(img)
            basename = images[i].split('/')[-1]
            filename = output_dir + basename.replace('.png', '_aug{}.png'.format(j))

            cv2.imwrite(filename, out)
        


        

if __name__ == '__main__':
    main(parse_args())
