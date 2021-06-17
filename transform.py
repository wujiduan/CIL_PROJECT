import random
import numpy as np
import cv2

import tensorflow as tf

scale_up_factor = 1.2
scale_down_factor = 0.8

# flip the image horizaontally with probability = prob
def horizontal_flip(train_img, label_img, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        return tf.image.random_flip_left_right(train_img), tf.image.random_flip_left_right(label_img)
    else:
        return train_img, label_img

# flip the image vertically with probability = prob
def vertical_flip(train_img, label_img, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        return tf.image.random_flip_up_down(train_img), tf.image.random_flip_up_down(label_img)
    else:
        return train_img, label_img

# rotate the image by k*90 degree with probability = prob
def rotate_90s(train_img, label_img, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        # 1<= k <= 3, rotate clockwise by 90/180/270 degree
        k = np.random.randint(low=1, high=4, size=1)
        return tf.image.rot90(train_img, k), tf.image.rot90(label_img, k), 
    else:
        return train_img, label_img

# rotate the image by a minor degree in [+25, -25] with probability = prob
def random_rotate(train_img, label_img, prob=0.75):
    rdn = np.random.random()
    pass

# scale up by 1.2 with probability = prob1(then randomly crop), scale down by 0.8 with probability = prob2(with padding), keep unchanged with probability = prob3.
# prob1 + prob2 + prob3 = 1
def scale_up_down(img, probs=[0.65,0.9, 1]):
    rdn = np.random.random()
    cur_height = img.shape[0]
    cur_width = img.shape[1]
    if rdn < probs[0]:
        # scale up
        img = 
        return img
    elif rdn < probs[1]:
        # scale down
        img = tf.image.resize_with_pad(img, cur_height*scale_down_factor, cur_width*scale_down_factor)
        return img
    # else remain unchanged
    return img
        


# adjust the hue of an RGB image by random factor in [1, 1.1] with probability = prob
def hue_image(img, hue_factor, seed=248, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        return img
    else:
        return tf.image.random_hue(img, hue_factor, seed=seed)

# adjust the brightness of an RGB image by random factor in [1, 1.1] with probability = prob
def brightness_image(img, brightness_factor, seed=248, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        return img
    else:
        return tf.image.random_brightness(img, brightness_factor, seed=seed)

# adjust the saturation of an RGB image by random factor in [1, 1.1] with probability = prob
def saturation_image(img, saturation_factor, seed=248, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        return img
    else:
        return tf.image.random_saturation(img, saturation_factor, seed=seed)

# adjust the contrast of an RGB image by random factor in [1, 1.1] with probability = prob
def contrast_image(img, contrast_factor, seed=248, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        return img
    else:
        return tf.image.random_contrast(img, contrast_factor, seed=seed)
