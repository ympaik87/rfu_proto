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
im_dir = pathlib.Path('F:/device/optics/images/190308_96w/2m')
im_dir
# %%
temp_li = ['Low Temp', 'High Temp']
x_range = slice(400, 1900)
y_range = slice(400, 1900)
colors_li = [plt.cm.get_cmap('hsv', 30)(i) for i in range(30)]
ch_dict = {
    'c': 'CalRed',
    'f': 'FAM',
    'q6': 'Q670',
    'q7': 'Q705',
    'h': 'HEX'
}
# %%
fpath_li = []
for dir in im_dir_li:
    for i in [0, 44]:
        fpath_li.append(dir/f'{i}_0_q7.jpg')

# %%
plt.figure(figsize=(12, 12))
im = np.array(Image.open(fpath_li[0]))
plt.imshow(im)

# %%
im_res_dir = pathlib.Path('F:/device/optics/results/96well')
# %%
for img in fpath_li:
    title = img.parents[0].name + ' - ' + img.name
    im = np.array(Image.open(img))
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im_gray = im_hsv[:, :, 2]

    thresh = skimage.filters.threshold_otsu(im_gray)
    threshed_im = im_gray > thresh

    bw = skimage.morphology.closing(threshed_im, skimage.morphology.disk(3))
    bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
    cleared = skimage.segmentation.clear_border(bw2)
    im_labeled = skimage.measure.label(cleared)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im_labeled)
    rect = matplotlib.patches.Rectangle(
        (500, 500), 1500, 1500, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f'{title} {im_labeled.max()}')
    plt.savefig(im_res_dir/f'step1_{title}_{im_labeled.max()}.png')
# %%
x_range = slice(500, 2000)
y_range = slice(500, 2000)
for img in fpath_li:
    title = img.parents[0].name + ' - ' + img.name
    im = np.array(Image.open(img))
    im_cropped = im[x_range, y_range]
    im_hsv = cv2.cvtColor(im_cropped, cv2.COLOR_BGR2HSV)
    im_gray = im_hsv[:, :, 2]

    thresh = skimage.filters.threshold_otsu(im_gray)
    threshed_im = im_gray > thresh

    bw = skimage.morphology.closing(threshed_im, skimage.morphology.disk(3))
    bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
    cleared = skimage.segmentation.clear_border(bw2)
    im_labeled = skimage.measure.label(cleared)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im_labeled)
    ax.set_title(f'{title} {im_labeled.max()}')
    plt.savefig(im_res_dir/f'step2_{title}_{im_labeled.max()}.png')
# %%
img = pathlib.Path('F:/device/optics/images/190308_96w/1s/0_0_q7.jpg')
im = np.array(Image.open(img))
im_cropped = im[x_range, y_range]

# %%
plt.imshow(im_cropped.sum(axis=2))

# %%
im_sum = im_cropped.sum(axis=2)
thresh = skimage.filters.threshold_otsu(im_sum)
threshed_im = im_sum > thresh

bw = skimage.morphology.closing(threshed_im, skimage.morphology.disk(3))
bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
cleared = skimage.segmentation.clear_border(bw2)
im_labeled = skimage.measure.label(cleared)
plt.figure(figsize=(12, 12))
plt.imshow(im_labeled)
# %%
divmod(5, 3)
# %%
thresh_li = [
    skimage.filters.threshold_otsu,
    skimage.filters.threshold_isodata,
    skimage.filters.threshold_li,
    skimage.filters.threshold_yen,
    skimage.filters.threshold_mean,
    skimage.filters.threshold_triangle,
]
im_hsv = im_cropped[:, :, 2]
fig, ax = plt.subplots(3, 2, figsize=(12, 18))
i = 0
for filter in thresh_li:
    thresh = filter(im_hsv)
    threshed_im = im_hsv > thresh

    bw = skimage.morphology.closing(threshed_im, skimage.morphology.disk(3))
    bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
    cleared = skimage.segmentation.clear_border(bw2)
    im_labeled = skimage.measure.label(cleared)
    image_label_overlay = skimage.color.label2rgb(
        im_labeled, bg_label=0, colors=colors_li)
    c, r = divmod(i, 3)
    ax[r, c].imshow(image_label_overlay)
    ax[r, c].set_title(filter.__name__)
    i += 1
plt.tight_layout()
plt.show()
# %%
skimage.filters.threshold_isodata.__name__
# %%
image_label_overlay = label2rgb(im_labeled)

# %%



# fig, ax = plt.subplots(figsize=(10, 6))
# ax.imshow(image_label_overlay)
# for region in skimage.measure.regionprops(im_labeled, intensity_image=im_gray):
