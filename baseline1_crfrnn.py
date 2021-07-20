import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import copy
from glob import glob
from random import sample
from PIL import Image

# !unzip /content/drive/MyDrive/cil-road-segmentation-2021.zip -d /content/drive/MyDrive/cil-road-segmentation-2021
'''
preparation for CRF-trivial version
'''

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

# from crfrnn_layer.permutohedral.gfilt import gfilt
# from crfrnn_layer.crfrnn.crf import CRF as CRF_LAYER
# from crfasrnn_pytorch.crfasrnn.crfasrnn_model import CrfRnnNet
from crfasrnn_pytorch.crfasrnn.crfrnn import CrfRnn
'''
eval_img is the original test images in RGB format, and the eval_mask is the output(prediction) of the trained model (UNet)
0 is a label instead of unknown in our case
eval_mask can be read from the output image in grayscale format or be a 0/1 matrix
'''


def CRF(eval_img, eval_mask, use_label=False):
    height, width = eval_mask.shape[0], eval_mask.shape[1]
    n_labels = 2
    d = dcrf.DenseCRF2D(width, height, n_labels)
    # print(np.max(eval_mask))
    if use_label:
        unary = unary_from_labels(eval_mask,
                                  n_labels,
                                  gt_prob=0.7,
                                  zero_unsure=False)
    else:
        prob = eval_mask  #/ 255.0
        y = np.zeros((height, width, 2))
        y[:, :, 1] = prob
        y[:, :, 0] = 1 - y[:, :, 1]
        U = unary_from_softmax(y.transpose((2, 0, 1)))
        unary = np.ascontiguousarray(U)
        # unary = unary_from_softmax(eval_mask)
    d.setUnaryEnergy(unary)

    # d.addPairwiseGaussian(sxy=(3, 3),
    #                       compat=10,
    #                       kernel=dcrf.DIAG_KERNEL,
    #                       normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d.addPairwiseBilateral(sxy=(10, 10),
    #                        srgb=(3, 3, 3),
    #                        rgbim=eval_img,
    #                        compat=10,
    #                        kernel=dcrf.DIAG_KERNEL,
    #                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # the parameter is hard-coded and copied for now
    # sigma_smooth = 3

    # sigma_color = 3
    # sigma_color_pos = 10
    # sigma_smooth = 4.883141908483301
    # sigma_color = 1.6935536588645517
    # sigma_color_pos = 10

    # compat_smooth = 10
    # compat_smooth = np.array([[-0.4946432, 1.27117338],
    #                           [0.59452892, 0.23182234]])
    # compat_smooth = np.array([[0.57842984, 0.95404739],
    #                           [0.52004501, 0.40422152]])
    # compat_appearance = 10
    # compat_appearance = np.array([[-0.30571318, 0.83015124],
    #                               [1.3217825, -0.13046645]])
    # compat_appearance = np.array([[-0.31741694, 1.18494974],
    #                               [1.12442186, 0.24456809]])

    sigma_smooth = 3
    compat_smooth = 3
    sigma_color_pos = 10
    sigma_color = 13
    compat_appearance = 20

    # add color-independent term, features are the locations
    d.addPairwiseGaussian(
        sxy=(sigma_smooth, sigma_smooth),
        #   compat=compat_smooth.astype(np.float32),
        compat=compat_smooth,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # add color-dependent term, features are (x, y, r, g, b)
    d.addPairwiseBilateral(
        sxy=(sigma_color_pos, sigma_color_pos),
        srgb=(sigma_color, sigma_color, sigma_color),
        rgbim=eval_img,
        #    compat=compat_appearance.astype(np.float32),
        compat=compat_appearance,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # inference and MAP
    infer_num = 10
    Q = d.inference(infer_num)

    # MAP = np.argmax(Q, axis=0)
    proba = np.array(Q)
    return proba[1].reshape((height, width))


# mean and std of training set
mean1 = np.array([0.330, 0.327, 0.293])
std1 = np.array([0.183, 0.176, 0.175])


# flip the image horizaontally with probability = prob
def horizontal_flip(train_img, label_img, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        return cv2.flip(train_img, 1), cv2.flip(label_img, 1)
    else:
        return train_img, label_img


# flip the image vertically with probability = prob
def vertical_flip(train_img, label_img, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        return cv2.flip(train_img, 0), cv2.flip(label_img, 0)
    else:
        return train_img, label_img


# rotate the image by k*90 degree with probability = prob
def rotate_90s(train_img, label_img, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        # 1<= k <= 3, rotate clockwise by 90/180/270 degree
        k = np.random.randint(low=1, high=4, size=1)[0]
        return np.rot90(train_img, k), np.rot90(label_img, k)
    else:
        return train_img, label_img


# adjust the hue of an RGB image by random factor in [-10, 10] with probability = prob
def hue_image(train_img,
              label_img,
              min_hue_factor=-10,
              max_hue_factor=10,
              prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        hsv = cv2.cvtColor(train_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        delta = np.random.randint(low=min_hue_factor,
                                  high=max_hue_factor,
                                  size=1)[0]
        h = np.clip(h + delta, 0, 180).astype(h.dtype)
        final_hsv = cv2.merge((h, s, v))
        new_train_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return new_train_img, label_img
    else:
        return train_img, label_img


# adjust the saturation of an RGB image by random factor in [-20, 20] with probability = prob
def saturation_image(train_img,
                     label_img,
                     min_saturation_factor=-20,
                     max_saturation_factor=20,
                     prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        hsv = cv2.cvtColor(train_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        delta = np.random.randint(low=min_saturation_factor,
                                  high=max_saturation_factor,
                                  size=1)[0]
        s = np.clip(s + delta, 0, 255).astype(h.dtype)
        final_hsv = cv2.merge((h, s, v))
        # print(h.shape)
        # print(v.shape)
        # print(s.shape)
        new_train_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return new_train_img, label_img
    else:
        return train_img, label_img


# adjust the brightness of an RGB image by random delta in [-30, 30] with probability = prob
def brightness_image(train_img,
                     label_img,
                     min_brightness_factor=-30,
                     max_brightness_factor=30,
                     prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        brightness_factor = np.random.uniform(min_brightness_factor,
                                              max_brightness_factor, 1)[0]
        new_train_img = _contrast_and_brightness(train_img, 1,
                                                 brightness_factor)
        return new_train_img, label_img
    else:
        return train_img, label_img


# adjust the contrast of an RGB image by random factor in [1, 1.5] with probability = prob
def contrast_image(train_img,
                   label_img,
                   min_contrast_factor=1,
                   max_contrast_factor=1.5,
                   prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        contrast_factor = np.random.uniform(min_contrast_factor,
                                            max_contrast_factor, 1)[0]
        new_train_img = _contrast_and_brightness(train_img, contrast_factor, 0)
        return new_train_img, label_img
    else:
        return train_img, label_img


def _contrast_and_brightness(img, contrast_factor, brightness_factor):
    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, contrast_factor, blank, 1 - contrast_factor,
                          brightness_factor)
    return dst


def random_scale(train_img, label_img, pad_reflect=False, probs=[0.7, 0.9, 1]):
    rdn = np.random.random()
    if rdn < probs[0]:
        return _crop_and_scale_up(train_img, label_img)
    elif rdn < probs[1]:
        return _random_shift(train_img, label_img)
    else:
        return _shrink_and_pad(train_img, label_img, pad_reflect)


def _crop_and_scale_up(train_img,
                       label_img,
                       crop_size=[(200, 200), (250, 250), (300, 300),
                                  (350, 350)]):
    original_shape = train_img.shape[:2]

    # cropping
    random_crop_shape = random.choice(crop_size)
    train_img, label_img = _random_crop(train_img, label_img,
                                        random_crop_shape)

    # scalse up
    train_img = cv2.resize(train_img,
                           original_shape,
                           interpolation=cv2.INTER_LINEAR)
    label_img = cv2.resize(label_img,
                           original_shape,
                           interpolation=cv2.INTER_LINEAR)

    return train_img, label_img


def _random_crop(train_img, label_img, crop_shape):
    original_shape = train_img.shape[:2]

    crop_h = original_shape[0] - crop_shape[0]
    crop_w = original_shape[1] - crop_shape[1]
    nh = random.randint(0, crop_h)
    nw = random.randint(0, crop_w)
    train_crop = train_img[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    label_crop = label_img[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return train_crop, label_crop


def _random_shift(train_img, label_img):
    original_shape = train_img.shape[:2]
    max_translation = np.multiply(0.15, original_shape).astype(np.int64)

    delta_h = random.randint(-max_translation[0], max_translation[0])
    delta_w = random.randint(-max_translation[1], max_translation[1])

    train_img = _shift(_shift(train_img, delta_h, height=True),
                       delta_w,
                       height=False)
    label_img = _shift(_shift(label_img, delta_h, height=True),
                       delta_w,
                       height=False)

    return train_img, label_img


def _shift(img, delta, height):
    if delta == 0:
        return img
    translated_img = np.empty_like(img)
    if height:
        if delta >= 0:
            translated_img[:delta] = 0
            translated_img[delta:] = img[:-delta]
        elif delta < 0:
            translated_img[:delta] = img[-delta:]
            translated_img[delta:] = 0
        return translated_img
    else:
        if delta >= 0:
            translated_img[:, :delta] = 0
            translated_img[:, delta:] = img[:, :-delta]
        elif delta < 0:
            translated_img[:, :delta] = img[:, -delta:]
            translated_img[:, delta:] = 0
        return translated_img


def _shrink_and_pad(train_img,
                    label_img,
                    pad_reflect,
                    shrink_range=(0.6, 0.95)):
    original_shape = train_img.shape[:2]

    random_ratio = np.random.uniform(shrink_range[0], shrink_range[1])
    train_img, label_img = _random_shrink(train_img, label_img, random_ratio)
    train_img, label_img = _random_pad(train_img, label_img, original_shape,
                                       pad_reflect)

    return train_img, label_img


def _random_shrink(train_img, label_img, ratio):
    original_shape = train_img.shape[:2]
    shrink_shape = (int(original_shape[0] * ratio),
                    int(original_shape[1] * ratio))
    # shrink
    train_img = cv2.resize(train_img,
                           shrink_shape,
                           interpolation=cv2.INTER_LINEAR)
    label_img = cv2.resize(label_img,
                           shrink_shape,
                           interpolation=cv2.INTER_LINEAR)
    return train_img, label_img


def _random_pad(train_img, label_img, target_shape, pad_reflect):
    original_shape = train_img.shape[:2]

    # put to center and padding
    margin = np.subtract(target_shape, original_shape)

    # random translation: limited by max_ratio and remained margin
    max_translation = np.multiply(0.15, original_shape)
    max_translation = np.minimum((margin // 2), max_translation)
    max_translation = max_translation.astype(np.int64)

    # place image with random translation
    pad_top = margin[0] // 2 + random.randint(-max_translation[0],
                                              max_translation[0])
    pad_left = margin[1] // 2 + random.randint(-max_translation[1],
                                               max_translation[1])
    pad_bottom = margin[0] - pad_top
    pad_right = margin[1] - pad_left

    # padding to original size
    if pad_reflect:
        train_img = cv2.copyMakeBorder(train_img, pad_top, pad_bottom,
                                       pad_left, pad_right, cv2.BORDER_REFLECT)
        label_img = cv2.copyMakeBorder(label_img, pad_top, pad_bottom,
                                       pad_left, pad_right, cv2.BORDER_REFLECT)
    else:
        train_img = cv2.copyMakeBorder(train_img,
                                       pad_top,
                                       pad_bottom,
                                       pad_left,
                                       pad_right,
                                       cv2.BORDER_CONSTANT,
                                       value=0)
        label_img = cv2.copyMakeBorder(label_img,
                                       pad_top,
                                       pad_bottom,
                                       pad_left,
                                       pad_right,
                                       cv2.BORDER_CONSTANT,
                                       value=0)

    return train_img, label_img


# rotate the image by a minor degree in [+25, -25] with probability = prob
def random_rotate(train_img,
                  label_img,
                  min_angle=-25,
                  max_angle=25,
                  prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        random_angle = np.random.uniform(min_angle, max_angle, 1)[0]
        return _rotate_image(train_img, random_angle), _rotate_image(
            label_img, random_angle)
    else:
        return train_img, label_img
        # return tf.convert_to_tensor(train_img), tf.convert_to_tensor(label_img)


def _rotate_image(img, angle):
    if -1 < angle < 1:
        return img
    shape_2d = (img.shape[1], img.shape[0])
    center_2d = (img.shape[1] / 2, img.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_2d, angle, 1.0)
    img = cv2.warpAffine(img,
                         rotation_matrix,
                         shape_2d,
                         flags=cv2.INTER_LINEAR)
    return img


def normalize(img):
    img = img.astype(np.float32) / 255.0
    img = img - mean1
    img = img / std1
    return img


def discretize(gt, threshold=40):
    # The order matters
    gt[gt < threshold] = 0
    gt[gt >= threshold] = 1
    return gt


def get_edge_mask(image):
    """ Accept image before binarization """
    edge_mask = cv2.Canny(image, 0, 255)
    edge_mask[image < 40] = 0
    edge_mask[edge_mask != 0] = 1
    return edge_mask


# from google.colab.patches import cv2_imshow

# train_img = cv2.imread('/content/drive/MyDrive/cil-project/cil-road-segmentation-2021/training/images/satImage_001.png')
# label_img = cv2.imread('/content/drive/MyDrive/cil-project/cil-road-segmentation-2021/training/groundtruth/satImage_001.png')
# imgs = np.hstack([train_img,label_img])
# cv2_imshow(imgs)
# !unzip /content/drive/MyDrive/cil-project/cil-road-segmentation-2021.zip -d /content/drive/MyDrive/cil-project/cil-road-segmentation-2021
# some constants
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 275  # size of the validation set (number of images)(nearly 20 percent of the training set)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road
# !rm -rf /content/validation
# !rm -rf /content/training
# unzip the dataset, split it and organize it in folders
# if not os.path.isdir('validation'):  # make sure this has not been executed yet
#     try:
#         #   !unzip /content/drive/MyDrive/cil-project/cil-road-segmentation-2021.zip -d cil-road-segmentation-2021
#         #   !mkdir training
#         #   !mkdir training/images
#         #   !mkdir training/groundtruth
#         #   !mv /content/cil-road-segmentation-2021/training/training/* training
#         # !cp -r /content/cil-road-segmentation-2021/training/training/* training
#         # put the additional images into the training set
#         #   !cp -r /content/drive/MyDrive/cil-project/additional_images/images/* training/images
#         #   !cp -r /content/drive/MyDrive/cil-project/additional_images/masks/* training/groundtruth
#         # !rm -rf /content/cil-road-segmentation-2021/training/training
#         #   !mkdir validation
#         #   !mkdir validation/images
#         #   !mkdir validation/groundtruth
#         for img in sample(glob("training/images/*.png"), VAL_SIZE):
#             os.rename(img, img.replace('training', 'validation'))
#             mask = img.replace('images', 'groundtruth')
#             os.rename(mask, mask.replace('training', 'validation'))
#     except:
#         print('Please upload a .zip file containing your datasets.')

# for img in sample(
#         glob("/cluster/home/jiduwu/CIL_PROJECT/training/images/*.png"),
#         VAL_SIZE):
#     os.rename(img, img.replace('training', 'validation'))
#     mask = img.replace('images', 'groundtruth')
#     os.rename(mask, mask.replace('training', 'validation'))


def load_all_from_path_255(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([
        np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))
    ]).astype(np.float32)


def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    print(path)
    return np.stack(
        [np.array(Image.open(f))
         for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.


# paths to training and validation datasets
train_path = '/cluster/home/jiduwu/CIL_PROJECT/training'
val_path = '/cluster/home/jiduwu/CIL_PROJECT/validation'

train_images = load_all_from_path_255(os.path.join(train_path, 'images'))
train_masks = load_all_from_path_255(os.path.join(train_path, 'groundtruth'))
val_images = load_all_from_path_255(os.path.join(val_path, 'images'))
val_masks = load_all_from_path_255(os.path.join(val_path, 'groundtruth'))
print(train_images.shape)
# !mkdir '/content/training/new_images'
# !mkdir '/content/training/new_groundtruth'
# !mkdir '/content/validation/new_images'
# !mkdir '/content/validation/new_groundtruth'
# !rm -rf '/content/training/new_images'
# !rm -rf '/content/training/new_groundtruth'
import tensorflow.compat.v1 as tf
# from google.colab.patches import cv2_imshow


def preprocess_saveimages(train_images, label_images, path_suffix,
                          name_suffix):
    cnt = 0
    for train_image, label_image in zip(train_images, label_images):
        train_image, label_image = horizontal_flip(train_image, label_image)
        train_image, label_image = vertical_flip(train_image, label_image)
        train_image, label_image = rotate_90s(train_image, label_image)
        train_image, label_image = random_rotate(train_image, label_image)
        train_image, label_image = random_scale(train_image, label_image)
        train_image, label_image = contrast_image(train_image, label_image)
        train_image, label_image = hue_image(train_image, label_image)
        train_image, label_image = saturation_image(train_image, label_image)
        train_image, label_image = brightness_image(train_image, label_image)
        save_img_path = path_suffix + '/images/'
        save_label_path = path_suffix + '/groundtruth/'
        cv2.imwrite(save_img_path + name_suffix + str(cnt) + '.png',
                    train_image)
        cv2.imwrite(save_label_path + name_suffix + str(cnt) + '.png',
                    label_image)
        cnt += 1


# print()
# preprocess_saveimages(train_images, train_masks, '/training', 'st')
# preprocess_saveimages(train_images, train_masks, '/content/training', 'nd')
# preprocess_saveimages(val_images, val_masks, '/validation', 'st')


# preprocess_saveimages(val_images, val_masks, '/content/validation', 'nd')
# !mv /content/training/new_images/* /content/training/images
# !mv /content/training/new_groundtruth/* /content/training/groundtruth
# !mv /content/validation/new_images/* /content/validation/images
# !mv /content/validation/new_groundtruth/* /content/validation/groundtruth
# !rm -rf '/content/training/new_images'
# !rm -rf '/content/training/new_groundtruth'
# !rm -rf '/content/validation/new_images'
# !rm -rf '/content/validation/new_groundtruth'
# import tensorflow.compat.v1 as tf
# from google.colab.patches import cv2_imshow
# def pre_process_and_save_images(train_images, label_images):
#   transformed_train_images = np.zeros(train_images.shape, dtype=np.float32)
#   transformed_label_images = np.zeros(label_images.shape, dtype=np.float32)
#   cnt = 0
#   for train_image, label_image in zip(train_images, label_images):
#       # train_image, label_image = horizontal_flip(train_image, label_image)
#       # train_image, label_image = vertical_flip(train_image, label_image)
#       # train_image, label_image = rotate_90s(train_image, label_image)
#       train_image, label_image = random_rotate(train_image, label_image)
#       # train_image, label_image = random_scale(train_image, label_image)
#       # train_image, label_image = hue_image(train_image, label_image)
#       # train_image, label_image = saturation_image(train_image, label_image)
#       # train_image, label_image = brightness_image(train_image, label_image)
#       # train_image, label_image = contrast_image(train_image, label_image)
#       transformed_train_images[cnt] = train_image
#       transformed_label_images[cnt] = label_image
#       cnt += 1
#   return np.vstack((train_images, transformed_train_images)), np.vstack((label_images, transformed_label_images))
# train_images, train_masks = pre_process_and_save_images(train_images, train_masks)
# # val_images_0, val_masks_0 = pre_process_and_save_images(val_images, val_masks)
# val_images, val_masks = pre_process_and_save_images(val_images, val_masks)
# print(val_images.shape)
# val_images, val_masks = pre_process_and_save_images(val_images, val_masks)
# print(val_images.shape)
# folder_name = '/content/drive/MyDrive/cil-project/cil-road-segmentation-2021/original_random_rotate'
# num_val = str(val_images.shape[0])
# # !mkdir $folder_name
# # np.save(folder_name+'/train_images.npy',train_images)
# # np.save(folder_name+'/train_masks.npy',train_masks)
# np.save(folder_name+'/val_images.npy',val_images)
# np.save(folder_name+'/val_masks.npy',val_masks)
# # !mkdir $folder_name/model
# # !mkdir $folder_name/predict
# folder_name = '/content/drive/MyDrive/cil-project/cil-road-segmentation-2021/original_random_rotate'
# num_val = str(val_images.shape[0])
# # !rm -rf $folder_name
# !mkdir $folder_name
# np.save(folder_name+'/train_images.npy',train_images)
# np.save(folder_name+'/train_masks.npy',train_masks)
# np.save(folder_name+'/val_images.npy',val_images)
# np.save(folder_name+'/val_masks.npy',val_masks)
# !mkdir $folder_name/model
# !mkdir $folder_name/predict
# def from_array_to_pictures(pictures_array, path_suffix):
#   cnt = 0
#   for a in pictures_array:
#     path = folder_name+path_suffix
#     cv2.imwrite(path + str(cnt) + '.png', a)
#     cnt += 1
# # !rm -rf $folder_name/training
# # !rm -rf $folder_name/training/images
# # !rm -rf $folder_name/training/groundtruth
# !rm -rf $folder_name/validation
# # !rm -rf $folder_name/validation/images
# # !rm -rf $folder_name/validation/groundtruth
# # !mkdir $folder_name/training
# # !mkdir $folder_name/training/images
# # !mkdir $folder_name/training/groundtruth
# !mkdir $folder_name/validation
# !mkdir $folder_name/validation/images
# !mkdir $folder_name/validation/groundtruth
# # from_array_to_pictures(train_images, '/training/images/')
# # from_array_to_pictures(train_masks, '/training/groundtruth/')
# from_array_to_pictures(val_images, '/validation/images/')
# from_array_to_pictures(val_masks, '/validation/groundtruth/')
# pip install tensorboard
# import tensorflow.compat.v1 as tf
# from google.colab.patches import cv2_imshow
# def pre_process_images(train_images, label_images):
#   transformed_train_images = np.zeros(train_images.shape, dtype=np.float32)
#   transformed_label_images = np.zeros(label_images.shape, dtype=np.float32)
#   cnt = 0
#   with tf.Session() as sess:
#     for train_image, label_image in zip(train_images, label_images):
#       # train_image, label_image = horizontal_flip(train_image, label_image)
#       # train_image, label_image = vertical_flip(train_image, label_image)
#       # train_image, label_image = rotate_90s(train_image, label_image)
#       # train_image, label_image = random_rotate(train_image, label_image)
#       train_image, label_image = random_scale(train_image, label_image)
#       # train_image, label_image = hue_image(train_image, label_image)
#       # train_image, label_image = saturation_image(train_image, label_image)
#       # train_image, label_image = brightness_image(train_image, label_image)
#       # train_image, label_image = contrast_image(train_image, label_image)
#       transformed_train_images[cnt] = train_image
#       transformed_label_images[cnt] = label_image
#       cnt += 1
#   return transformed_train_images, transformed_label_images
# def squeeze_patches(images):
#   new_train_image = np.empty([50000, 3], dtype=float)
#   out_cnt = 0
#   for img in train_image_patches:
#     sum = [0, 0, 0]
#     cnt = 0
#     for i in range(16):
#       for j in range(16):
#         sum[0] += img[i][j][0]
#         sum[1] += img[i][j][1]
#         sum[2] += img[i][j][2]
#         cnt += 1
#     sum[0] /= cnt
#     sum[1] /= cnt
#     sum[2] /= cnt
#     new_train_image[out_cnt] = sum
#     out_cnt += 1
#   return new_train_image
def image_to_patches(images, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % PATCH_SIZE) + (
        w % PATCH_SIZE) == 0  # make sure images can be patched exactly

    h_patches = h // PATCH_SIZE
    w_patches = w // PATCH_SIZE
    patches = images.reshape(
        (n_images, h_patches, PATCH_SIZE, h_patches, PATCH_SIZE, -1))
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape(
        (n_images, h_patches, PATCH_SIZE, h_patches, PATCH_SIZE, -1))
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    shape = masks.shape

    #patches = squeeze_patches(patches)

    return patches, labels


import torch
import sys
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import matplotlib

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(
            device=device, non_blocking=True)


def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400)):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, 'images'))
        self.y = load_all_from_path(os.path.join(self.path, 'groundtruth'))
        # self.x, self.y = pre_process_images(self.x, self.y)
        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1],
                                self.x.shape[2]):  # resize images
            self.x = np.stack(
                [cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack(
                [cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1,
                             1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        return x, y

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device),
                                np_to_tensor(self.y[[item]], self.device))

    def __len__(self):
        return self.n_samples


class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,
                      out_channels=out_ch,
                      kernel_size=3,
                      padding=1), nn.ReLU(), nn.BatchNorm2d(out_ch),
            nn.Conv2d(in_channels=out_ch,
                      out_channels=out_ch,
                      kernel_size=3,
                      padding=1), nn.ReLU())

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList([
            Block(in_ch, out_ch)
            for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])
        ])  # encoder blocks
        self.pool = nn.MaxPool2d(
            2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
            for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
        ])  # deconvolution
        self.dec_blocks = nn.ModuleList([
            Block(in_ch, out_ch)
            for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
        ])  # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1),
            nn.Sigmoid())  # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs,
                                          enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


class UNet_crfrnn(nn.Module):
    def __init__(self, trained_UNet):
        super().__init__()
        self.trained_UNet = trained_UNet
        self.crfrnn = CrfRnn(num_labels=2,
                             num_iterations=5).to(device).to(device)

    def forward(self, x):
        # print("input x requires_grad:", x.requires_grad) # - False
        image = x * 255
        x.requires_grad = True
        prob = self.trained_UNet(x)
        # print("prob shape:", prob.shape) # - (1, 1, 384, 384)
        # print("image shape:", image.shape) # - (1, 3, 384, 384)

        prob_np = prob.detach().numpy()[0][0]
        height, width = prob_np.shape[0], prob_np.shape[1]
        y = np.zeros((height, width, 2))
        y[:, :, 1] = prob_np
        y[:, :, 0] = 1 - y[:, :, 1]
        y = np_to_tensor(y.transpose((2, 0, 1)), device).unsqueeze(0).float()
        y.requires_grad = True
        # print("y requires_grad:", y.requires_grad) # - False
        # print("y shape:", y.shape) # - 1, 2, 384, 384
        # print("y:", y)
        unary = self.crfrnn(image, y)  #.detach().numpy()[0]
        # unary shape - 1, 2, 384, 384
        # labels = torch.argmax(unary, 1, True).to(device=device).float() # - not differentiable
        # print("unary requires_grad:", unary.requires_grad)
        sm = torch.nn.functional.softmax(unary, dim=1)
        labels = sm[:, 1, :, :]
        # print("crfrnn output dtype:", labels.dtype)
        # labels.requires_grad = True
        # print("output shape:", output.shape)  # - 2, 384, 384
        # labels = output.argmax(axis=0).astype('uint8')

        # return type of argmax is long
        # labels = torch.argmax(labels, 1, True).to(device=device)
        # print("labels requires_grad:", labels.requires_grad)
        labels = labels.unsqueeze(0)
        # print("labels shape:", labels.shape)  # - 384, 384
        # print("labels dtype:", labels.dtype)
        # labels = np_to_tensor(labels, device).unsqueeze(0)
        # labels = labels.unsqueeze(0).float()
        # labels = labels.float()
        # labels.requires_grad = True
        # print("labels requires_grad:", labels.requires_grad)  # - False
        return labels


# class UNet_CRF(nn.Module):
#     def __init__(self, trained_UNet):
#         super().__init__()
#         self.trained_UNet = trained_UNet

#         # for child in self.trained_UNet.children():
#         #     for param in child.parameters():
#         #         param.requires_grad = False
#         # trainable_kstd = True can make the training less stable
#         # self.crf = CRF_LAYER(3, 2).to(device)

#     def forward(self, x):

#         # prob = eval_mask  #/ 255.0
#         # y = np.zeros((height, width, 2))
#         # y[:, :, 1] = prob
#         # y[:, :, 0] = 1 - y[:, :, 1]
#         # U = unary_from_softmax(y.transpose((2, 0, 1)))
#         # unary = np.ascontiguousarray(U)
#         # print("x:", x)

#         print(x.shape)
#         input = x * 255
#         # print("input:", input)
#         # sys.exit()
#         x = self.trained_UNet(x)
#         x_np = x.cpu().numpy()[0][0]
#         height, width = x_np.shape[0], x_np.shape[1]
#         print("numpy:")
#         print(x_np.shape)
#         y = np.zeros((height, width, 2))

#         y[:, :, 1] = x_np
#         y[:, :, 0] = 1 - y[:, :, 1]
#         y = np.ascontiguousarray(y)
#         x = self.crf(
#             np_to_tensor(y.transpose((2, 0, 1)), device).unsqueeze(0).float(),
#             input)
#         return x[:, 1, :, :].unsqueeze(0)


def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    patches_hat = y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches,
                                PATCH_SIZE).mean((-1, -3)) > CUTOFF
    patches = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches,
                        PATCH_SIZE).mean((-1, -3)) > CUTOFF
    return (patches == patches_hat).float().mean()


# folder_name = '/content/drive/MyDrive/cil-project'
folder_name = '/cluster/home/jiduwu/CIL_PROJECT'
# folder_name = ''
num_val = VAL_SIZE * 2
import gc


def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns,
          optimizer, n_epochs):
    # training loop
    logdir = '/cluster/home/jiduwu/CIL_PROJECT/tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_' + k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            # x.requires_grad = True
            # y.requires_grad = True
            y_hat = model(x)  # forward pass
            # print("mask y type:", y.dtype) # - torch.float32
            # print("y shape:", y.shape) # - 1, 1, 384, 384
            # print("y:", y)
            # print("y hat requires_grad:", y_hat.requires_grad)

            loss = loss_fn(y_hat, y)
            # loss = Variable(loss, requires_grad=True)
            # print(loss)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix(
                {k: sum(v) / len(v)
                 for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y)

                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_' + k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
            writer.add_scalar(k, v, epoch)
        print(' '.join([
            '\t- ' + str(k) + ' = ' + str(v) + '\n '
            for (k, v) in history[epoch].items()
        ]))
        #show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

        # deep copy the model
        if history[epoch]['val_acc'] > best_acc:
            # print(history[epoch]['val_acc'])
            best_acc = history[epoch]['val_acc']
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),
                       folder_name + '/model/model_e_temp.pt')

        gc.collect()

    print('Finished Training')
    print(best_acc)
    torch.save(
        model.state_dict(), folder_name + '/model/model_e' + str(n_epochs) +
        '_val' + str(num_val) + '.pt')
    model.load_state_dict(best_model_wts)
    torch.save(
        model.state_dict(), folder_name + '/model/best_val_acc_model_e' +
        str(n_epochs) + '_val' + str(num_val) + '.pt')
    # plot loss curves
    plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    plt.plot([v['val_loss'] for k, v in history.items()],
             label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    return model


# # Garbage Collector - use it like gc.collect()
# import gc

# # Custom Callback To Include in Callbacks List At Training Time
# class GarbageCollectorCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         gc.collect()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = UNet().to(device)
# model.load_state_dict(torch.load('/content/drive/MyDrive/cil-project/model/model_e_temp.pt'))

# reshape the image to simplify the handling of skip connections and maxpooling
# train_dataset = ImageDataset(folder_name+'/training', device, use_patches=False, resize_to=(384, 384))
# val_dataset = ImageDataset(folder_name+'/validation', device, use_patches=False, resize_to=(384, 384))
train_dataset = ImageDataset('/cluster/home/jiduwu/CIL_PROJECT/training',
                             device,
                             use_patches=False,
                             resize_to=(384, 384))
val_dataset = ImageDataset('/cluster/home/jiduwu/CIL_PROJECT/validation',
                           device,
                           use_patches=False,
                           resize_to=(384, 384))
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=True)

unet_model = UNet().to(device)
unet_model.load_state_dict(
    torch.load(
        '/cluster/home/jiduwu/CIL_PROJECT/model/best_val_acc_model_e60_val800_4100.pt'
    ))
# '/cluster/home/jiduwu/CIL_PROJECT/model/model_jiduan60_val550.pt'))
loss_fn = nn.BCELoss()
metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
# optimizer = torch.optim.Adam(model.parameters())

n_epochs = 2
# train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer,
#   n_epochs)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = UNet().to(device)
# model.load_state_dict(torch.load('/content/drive/MyDrive/cil-project/model/model_e_temp.pt'))
# model.load_state_dict(torch.load(folder_name+'/model/best_val_acc_model_e'+str(n_epochs)+'_val'+num_val+'.pt'))
# torch.save(
#     model.state_dict(), folder_name + '/model/model_jiduan' + str(n_epochs) +
#     '_val' + str(num_val) + '.pt')

test_path = '/cluster/home/jiduwu/CIL_PROJECT/test_images'


def create_submission(labels, test_filenames, submission_filename):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), labels):
            img_number = int(re.findall(r"\d+", fn)[-1])
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number,
                                                       i * PATCH_SIZE,
                                                       j * PATCH_SIZE,
                                                       int(patch_array[j, i])))


# predict on test set
test_filenames = sorted(glob(test_path + '/*.png'))
test_images = load_all_from_path(test_path)
batch_size = test_images.shape[0]
size = test_images.shape[1:3]
# we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
test_images = np.stack(
    [cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
# label_images = np.zeros(test_images.shape, dtype=np.float32)
# test_images, label_images = contrast_image(test_images, label_images)
test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
test_pred = [
    unet_model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)
]
test_pred = np.concatenate(test_pred, 0)
test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred],
                     0)  # resize to original shape
ori_test_images = [cv2.imread(tmp_name) for tmp_name in test_filenames]

# now compute labels without crf post-processing
test_pred_ncrf = test_pred.reshape(
    (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
test_pred_ncrf = np.moveaxis(test_pred_ncrf, 2, 3)
test_pred_ncrf = np.round(np.mean(test_pred_ncrf, (-1, -2)) > CUTOFF)
create_submission(test_pred_ncrf,
                  test_filenames,
                  submission_filename=folder_name +
                  '/predict/unet_submission_e' + str(n_epochs) + '_val' +
                  str(num_val) + '_best.csv')

test_pred_crf = [
    CRF(ori_test_images[i],
        test_pred[i])  #for i in range(1)  # to debug faster
    for i in range((len(ori_test_images)))
]

# test = torch.tensor([[1., -1.], [1., -1.]]).to(device)
# test.to(device)

crfrnn_model = UNet_crfrnn(unet_model).to(device)
for param in crfrnn_model.trained_UNet.parameters():
    param.requires_grad = False
# print(crfrnn_model.parameters())
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, crfrnn_model.parameters()))

train(train_dataloader, val_dataloader, crfrnn_model, loss_fn, metric_fns,
      optimizer, n_epochs)

test_pred_crfrnn = [
    crfrnn_model(t).detach().cpu().numpy().round()
    for t in test_images.unsqueeze(1)
]
test_pred_crfrnn = np.concatenate(test_pred_crfrnn, 0)
test_pred_crfrnn = np.moveaxis(test_pred_crfrnn, 1, -1)  # CHW to HWC
test_pred_crfrnn = np.stack(
    [cv2.resize(img, dsize=size) for img in test_pred_crfrnn],
    0)  # resize to original shape

for i in range(len(test_pred_crfrnn)):
    tmp_pred_fig = test_pred[i]
    tmp_pred_figname = test_filenames[i][:-4] + '_pred.png'
    tmp_pred_figname = tmp_pred_figname.replace('test_images', 'pred_images')
    matplotlib.image.imsave(tmp_pred_figname, tmp_pred_fig)

    tmp_crfrnn_fig = test_pred_crfrnn[i]
    tmp_crfrnn_figname = test_filenames[i][:-4] + '_crfrnn.png'
    # tmp_figname = tmp_figname.replace('')
    # print("temp filename:", tmp_figname)
    tmp_crfrnn_figname = tmp_crfrnn_figname.replace('test_images',
                                                    'crfrnn_images')
    # print("temp filename:", tmp_figname)
    matplotlib.image.imsave(tmp_crfrnn_figname, tmp_crfrnn_fig)

    tmp_fig = test_pred[i]

test_pred_crfrnn = np.stack(test_pred_crfrnn)

test_pred_crfrnn = test_pred_crfrnn.reshape(
    (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
test_pred_crfrnn = np.moveaxis(test_pred_crfrnn, 2, 3)
test_pred_crfrnn = np.round(np.mean(test_pred_crfrnn, (-1, -2)) > CUTOFF)
create_submission(test_pred_crfrnn,
                  test_filenames,
                  submission_filename=folder_name +
                  '/predict/unet_submission_e' + str(n_epochs) + '_val' +
                  str(num_val) + '_best_crfrnn.csv')

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input, input.shape)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target, target.shape)
output = loss(input, target)
print(output)
output.backward()
