import random
import numpy as np
import cv2

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
def hue_image(train_img, label_img, min_hue_factor=-10, max_hue_factor=10, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        hsv = cv2.cvtColor(train_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        delta = np.random.randint(low=min_hue_factor, high=max_hue_factor, size=1)[0]
        h = np.clip(h+delta, 0, 180).astype(h.dtype)
        final_hsv = cv2.merge((h, s, v))
        new_train_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return new_train_img, label_img
    else:
        return train_img, label_img


# adjust the saturation of an RGB image by random factor in [-20, 20] with probability = prob
def saturation_image(train_img, label_img, min_saturation_factor=-20, max_saturation_factor=20, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        hsv = cv2.cvtColor(train_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        delta = np.random.randint(low=min_saturation_factor, high=max_saturation_factor, size=1)[0]
        s = np.clip(s+delta, 0, 255).astype(h.dtype)
        final_hsv = cv2.merge((h, s, v))
        # print(h.shape)
        # print(v.shape)
        # print(s.shape)
        new_train_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return new_train_img, label_img
    else:
        return train_img, label_img


# adjust the brightness of an RGB image by random delta in [-30, 30] with probability = prob
def brightness_image(train_img, label_img, min_brightness_factor=-30, max_brightness_factor=30, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        brightness_factor = np.random.uniform(min_brightness_factor, max_brightness_factor, 1)[0]
        new_train_img = _contrast_and_brightness(train_img, 1, brightness_factor)
        return new_train_img, label_img
    else:
        return train_img, label_img


# adjust the contrast of an RGB image by random factor in [0.6, 2.5] with probability = prob
def contrast_image(train_img, label_img, min_contrast_factor=0.6, max_contrast_factor=2.5, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        contrast_factor = np.random.uniform(min_contrast_factor, max_contrast_factor, 1)[0]
        new_train_img = _contrast_and_brightness(train_img, contrast_factor, 0)
        return new_train_img, label_img
    else:
        return train_img, label_img


def _contrast_and_brightness(img, contrast_factor, brightness_factor):
    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, contrast_factor, blank, 1-contrast_factor, brightness_factor)
    return dst


def random_scale(train_img, label_img, pad_reflect=False, probs=[0.7, 0.9, 1]):
    rdn = np.random.random()
    if rdn < probs[0]:
        return _crop_and_scale_up(train_img, label_img)
    elif rdn < probs[1]:
        return _random_shift(train_img, label_img)
    else:
        return _shrink_and_pad(train_img, label_img, pad_reflect)


def _crop_and_scale_up(train_img, label_img, crop_size=[(200, 200), (250, 250), (300, 300), (350, 350)]):
    original_shape = train_img.shape[:2]

    # cropping
    random_crop_shape = random.choice(crop_size)
    train_img, label_img = _random_crop(train_img, label_img, random_crop_shape)

    # scalse up
    train_img = cv2.resize(train_img, original_shape, interpolation=cv2.INTER_LINEAR)
    label_img = cv2.resize(label_img, original_shape, interpolation=cv2.INTER_LINEAR)

    return train_img, label_img


def _random_crop(train_img, label_img, crop_shape):
    original_shape = train_img.shape[:2]

    crop_h = original_shape[0]-crop_shape[0]
    crop_w = original_shape[1]-crop_shape[1]
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

    train_img = _shift(_shift(train_img, delta_h, height=True), delta_w, height=False)
    label_img = _shift(_shift(label_img, delta_h, height=True), delta_w, height=False)

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


def _shrink_and_pad(train_img, label_img, pad_reflect, shrink_range=(0.6, 0.95)):
    original_shape = train_img.shape[:2]

    random_ratio = np.random.uniform(shrink_range[0], shrink_range[1])
    train_img, label_img = _random_shrink(train_img, label_img, random_ratio)
    train_img, label_img = _random_pad(train_img, label_img, original_shape, pad_reflect)

    return train_img, label_img


def _random_shrink(train_img, label_img, ratio):
    original_shape = train_img.shape[:2]
    shrink_shape = (int(original_shape[0]*ratio), int(original_shape[1]*ratio))
    # shrink
    train_img = cv2.resize(train_img, shrink_shape, interpolation=cv2.INTER_LINEAR)
    label_img = cv2.resize(label_img, shrink_shape, interpolation=cv2.INTER_LINEAR)
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
    pad_top = margin[0] // 2 + random.randint(-max_translation[0], max_translation[0])
    pad_left = margin[1] // 2 + random.randint(-max_translation[1], max_translation[1])
    pad_bottom = margin[0] - pad_top
    pad_right = margin[1] - pad_left

    # padding to original size
    if pad_reflect:
        train_img = cv2.copyMakeBorder(train_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        label_img = cv2.copyMakeBorder(label_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    else:
        train_img = cv2.copyMakeBorder(train_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        label_img = cv2.copyMakeBorder(label_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    return train_img, label_img


# rotate the image by a minor degree in [+25, -25] with probability = prob
def random_rotate(train_img, label_img, min_angle=-25, max_angle=25, prob=0.75):
    rdn = np.random.random()
    if rdn < prob:
        random_angle = np.random.uniform(min_angle, max_angle, 1)[0]
        return _rotate_image(train_img, random_angle), _rotate_image(label_img, random_angle)
    else:
        return train_img, label_img
        # return tf.convert_to_tensor(train_img), tf.convert_to_tensor(label_img)


def _rotate_image(img, angle):
    if -1 < angle < 1:
        return img
    shape_2d = (img.shape[1], img.shape[0])
    center_2d = (img.shape[1] / 2, img.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_2d, angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, shape_2d, flags=cv2.INTER_LINEAR)
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


if __name__ == "__main__":
    train_img = cv2.imread('E:\\ETH\\2021 Spring\\CIL\\cil-road-segmentation-2021\\training\\training\\images\\satImage_001.png')
    label_img = cv2.imread('E:\\ETH\\2021 Spring\\CIL\\cil-road-segmentation-2021\\training\\training\\groundtruth\\satImage_001.png')
    imgs = np.hstack([train_img,label_img])
    cv2.imshow("before", imgs)

    img_t, img_l = saturation_image(train_img, label_img,-20,-10,1)
    res_imgs = np.hstack([img_t,img_l])
    cv2.imshow("after", res_imgs)

    # img_yuv = cv2.cvtColor(train_img, cv2.COLOR_BGR2YUV)
    # # equalize the histogram of the Y channel
    # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #
    # # convert the YUV image back to RGB format
    # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # cv2.imshow("after", img_output)


    cv2.waitKey(0)
