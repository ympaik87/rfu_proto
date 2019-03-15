# %%
import pathlib
import skimage
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
%matplotlib inline


# %%
im_dir_li = []
for i in range(1, 4):
    for j in ['m', 's']:
        im_dir_li.append(
            pathlib.Path(f'F:/device/optics/images/190308_96w/{str(i)+j}'))
# %%
res_dir = pathlib.Path('F:/device/optics/results/96well')
# %%
temp_li = ['Low Temp', 'High Temp']
x_range = slice(500, 1800)
y_range = slice(500, 1800)
colors_li = [plt.cm.get_cmap('hsv', 30)(i) for i in range(30)]
ch_dict = {
    'c': 'CalRed',
    'f': 'FAM',
    'q6': 'Q670',
    'q7': 'Q705',
    'h': 'HEX'
}
# %%


def get_fpath(im_dir):
    fpath_li = []
    for i in range(45):
        fpath_li.append(im_dir/f'{i}_0_f.jpg')
    return fpath_li

# %%


x_range = slice(600, 2000)
y_range = slice(600, 1800)
for channel in ['c', 'f', 'q6', 'h', 'q7']:
    _li = []
    for im_dir in im_dir_li:
        _li.append(im_dir/f'44_0_{channel}.jpg')
    fig, ax = plt.subplots(3, 2, figsize=(12, 18))
    for i in range(len(_li)):
        col, row = divmod(i, 2)
        im = np.array(Image.open(_li[i]))
        im_sum = im.sum(axis=2)
        if '3' in _li[i].parent.name:
            im_sum = skimage.transform.rotate(im_sum, 90)
        elif 's' in _li[i].parent.name:
            im_sum = skimage.transform.rotate(im_sum, 180)

        im_cropped = im_sum[y_range, x_range]
        thresh_sum = skimage.filters.threshold_otsu(im_cropped)
        threshed_im_sum = im_cropped > thresh_sum
        cleared = skimage.segmentation.clear_border(threshed_im_sum)
        bw = skimage.morphology.closing(cleared, skimage.morphology.disk(3))
        bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
        im_labeled = skimage.measure.label(bw2)
        im_res = skimage.color.label2rgb(im_labeled, bg_label=0,
                                         colors=colors_li)

        ax[col, row].imshow(im_res)
        ax[col, row].set_title(f'{_li[i].parent.name} - {_li[i].name}')
    fig.savefig(res_dir/f'step2_otsu_{channel}_channel_45c_600_2000.png')

# %%
