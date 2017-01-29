#/usr/bin/python
# --------------------------------------------------------
# floor_recog
# Written by Sai Prabhakar
# CMU-RI Masters
# --------------------------------------------------------

import numpy as np
import cv2
from pythonlayers.helpers import *


def _combine_images(img1, mask, img2):
    """Copy masked regions of img2 to img1
    img2 - original image
    img1 - blur image
    mask - visualized mask
    """
    n_mask = abs(1 - mask)
    n_mask = np.repeat(n_mask[:, :, np.newaxis], 3, axis=2)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    return mask * img2 + n_mask * img1


def _analyse_heat_maps(heat_map1, heat_map2):
    ovlap_count = (heat_map1 & heat_map2).sum().astype(float)
    union_count = (heat_map1 | heat_map2).sum()
    iou_percent = 100 * ovlap_count / union_count
    #print ovlap_count, union_count, iou_percent
    return iou_percent


def _load_mean_binaryproto(
        fileName='placesOriginalModel/places205CNN_mean.binaryproto',
        im_target_size=227):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(fileName, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    # changing order of rows, colmn, channel, batch_size
    arr = np.squeeze(arr.transpose((2, 3, 1, 0)))
    im_scaley = float(im_target_size) / float(256)
    im_scalex = float(im_target_size) / float(256)
    meanarr = cv2.resize(
        arr,
        None,
        None,
        fx=im_scalex,
        fy=im_scaley,
        interpolation=cv2.INTER_LINEAR)
    return meanarr


def _find_threshold(h_map, ratio):
    assert ratio <= 1.0
    temp = np.sort(
        h_map.reshape(h_map.shape[0] * h_map.shape[1]), kind='mergesort')
    return temp[int((1 - ratio) * len(temp))]


def _round_image(img):
    img[img > 255] = 255
    img[img < 0] = 0
    return img


def _get_image_blob(img_name, meanarr, im_target_size):
    im = _load_image(img_name, im_target_size)
    processed_ims = im - meanarr
    blob = im_to_blob(processed_ims)
    return blob


def _get_image_blob_from_image(im, meanarr, im_target_size):
    processed_ims = im - meanarr
    blob = im_to_blob(processed_ims)
    return blob


def _load_image(img_name, im_target_size):
    im = cv2.imread(img_name)
    im = im.astype(np.float32, copy=True)

    min_curr_size = min(im.shape[:2])
    im_scale = float(im_target_size) / float(min_curr_size)

    #im_scaley = float(im_target_size) / float(im_size[0])
    #im_scalex = float(im_target_size) / float(im_size[1])
    im = cv2.resize(
        im[:min_curr_size, :min_curr_size, :],
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR)
    return im.copy()


def _get_occluded_image_blob(im, size_patch, cR, cC, meanarr, im_target_size):
    l_blob = []
    l_occ_map = []
    #import IPython
    #IPython.embed()
    occluded_image, occ_map = _occlude_image(im.copy(), cR, cC, size_patch,
                                             im_target_size)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image',occluded_image.astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    processed_ims = occluded_image - meanarr
    l_blob = im_to_blob(processed_ims)
    l_occ_map = occ_map
    return l_blob, l_occ_map


def _get_coordinates(cR, cC, size_patch, maxRow, maxCol):
    r1 = cR
    r2 = cR + size_patch
    c1 = cC
    c2 = cC + size_patch
    if r1 < 0:
        r1 = 0
    if r2 - 1 > maxRow - 1:
        r2 = maxRow
    if c1 < 0:
        c1 = 0
    if c2 - 1 > maxCol - 1:
        c2 = maxCol

    return r1, r2, c1, c2


def _occlude_image(im, cR, cC, size_patch, im_target_size):
    """creates gray patches in image."""
    r1, r2, c1, c2 = _get_coordinates(cR, cC, size_patch, im.shape[0],
                                      im.shape[1])
    im[r1:r2, c1:c2, :] = 127.5
    occ_map = np.ones((im_target_size, im_target_size))
    occ_map[r1:r2, c1:c2] = 0
    return im, occ_map
