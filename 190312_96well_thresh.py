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
img = pathlib.Path('F:/device/optics/images/190308_96w/1s/0_0_h.jpg')
im = np.array(Image.open(img))
im_cropped = im[x_range, y_range]

# %%
plt.imshow(im_cropped.sum(axis=2))
plt.title(f'{img.parent.name} - {img.name}')

# %%
im_sum = im_cropped.sum(axis=2)
im_hsv = cv2.cvtColor(im_cropped, cv2.COLOR_BGR2HSV)
im_gray = im_hsv[:, :, 2]

thresh_sum = skimage.filters.threshold_mean(im_sum)
threshed_im_sum = im_sum > thresh_sum
thresh_val = skimage.filters.threshold_mean(im_gray)
threshed_im_val = im_gray > thresh_val

# %%
_li = []
for threshed_im in [threshed_im_sum, threshed_im_val]:
    cleared = skimage.segmentation.clear_border(threshed_im)
    bw = skimage.morphology.closing(cleared, skimage.morphology.disk(3))
    bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
    im_labeled = skimage.measure.label(bw2)
    _li.append(skimage.color.label2rgb(
        im_labeled, bg_label=0, colors=colors_li))

fig, ax = plt.subplots(3, 2, figsize=(12, 12))
ax[0, 0].imshow(im_sum)
ax[0, 1].imshow(threshed_im_sum)
ax[1, 0].imshow(im_gray)
ax[1, 1].imshow(threshed_im_val)
ax[2, 0].imshow(_li[0])
ax[2, 1].imshow(_li[1])
plt.show()


# %%
plt.figure(figsize=(12, 6))
sum_dist = [i for i in im_sum.ravel() if i > 0]
val_dist = [i for i in im_gray.ravel() if i > 0]
plt.hist(sum_dist, bins=100, alpha=0.5,
         label=f'sum (total {len(sum_dist)} pixel)')
plt.axvline(x=thresh_sum, color='b', alpha=1,
            label=f'Sum Thresh={round(thresh_sum, 2)}')
plt.hist(val_dist, bins=100, alpha=0.5,
         label=f'val (total {len(val_dist)} pixel)')
plt.axvline(x=thresh_val, color='r', alpha=1,
            label=f'Val Thresh={round(thresh_val, 2)}')
plt.legend(loc='upper right')
plt.show()

# %%
print(len(sum_dist))
print(len(val_dist))

# %%

thresh_sum
# %%
arr = im_sum.ravel()
new_arr = [i for i in arr if i > 0]
plt.hist(new_arr, bins=100)
# %%

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
    ax[r, c].set_title(f'{filter.__name__} - {round(thresh, 4)}')
    i += 1
fig.savefig('thresh_with_val_in_hsv.png')
plt.tight_layout()
plt.show()
# %%
im_gray = im_cropped.sum(axis=2)
fig, ax = plt.subplots(3, 2, figsize=(12, 18))
i = 0
for filter in thresh_li:
    thresh = filter(im_gray)
    threshed_im = im_gray > thresh

    bw = skimage.morphology.closing(threshed_im, skimage.morphology.disk(3))
    bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
    cleared = skimage.segmentation.clear_border(bw2)
    im_labeled = skimage.measure.label(cleared)
    image_label_overlay = skimage.color.label2rgb(
        im_labeled, bg_label=0, colors=colors_li)
    c, r = divmod(i, 3)
    ax[r, c].imshow(image_label_overlay)
    ax[r, c].set_title(f'{filter.__name__} - {round(thresh, 4)}')
    i += 1
fig.savefig('thresh_with_sum.png')
plt.tight_layout()
plt.show()
# %%
skimage.filters.threshold_isodata.__name__
# %%
threshed_im = im_cropped.sum(axis=2) > 0

bw = skimage.morphology.closing(threshed_im, skimage.morphology.disk(3))
bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
cleared = skimage.segmentation.clear_border(bw2)
im_labeled = skimage.measure.label(cleared)
image_label_overlay = skimage.color.label2rgb(
    im_labeled, bg_label=0, colors=colors_li)
plt.figure(figsize=(12, 12))
plt.imshow(image_label_overlay)
# %%
fig, ax = plt.subplots(3, 2, figsize=(12, 18))
i = 0
for k, v in ch_dict.items():
    img = im_dir/f'0_0_{k}.jpg'
    im = np.array(Image.open(img))
    im_cropped = im[x_range, y_range]

    im_gray = im_cropped.sum(axis=2)
    thresh = skimage.filters.threshold_otsu(im_gray)
    threshed_im = im_gray > thresh

    bw = skimage.morphology.closing(threshed_im, skimage.morphology.disk(3))
    bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
    cleared = skimage.segmentation.clear_border(bw2)
    im_labeled = skimage.measure.label(cleared)
    image_label_overlay = skimage.color.label2rgb(
        im_labeled, bg_label=0, colors=colors_li)
    c, r = divmod(i, 3)
    ax[r, c].imshow(image_label_overlay)
    ax[r, c].set_title(f'{img.name} - {round(thresh, 4)}')
    i += 1
fig.savefig(res_dir/'otsu_thresh_with_sum_by_channels.png')
plt.tight_layout()
plt.show()
# %%
fig, ax = plt.subplots(3, 2, figsize=(12, 18))
i = 0
for k, v in ch_dict.items():
    img = im_dir/f'0_0_{k}.jpg'
    im = np.array(Image.open(img))
    im_cropped = im[x_range, y_range]

    im_gray = im_cropped.sum(axis=2)
    block_size = 11
    thresh = skimage.filters.threshold_local(im_gray, block_size)
    threshed_im = im_gray > thresh

    bw = skimage.morphology.closing(threshed_im, skimage.morphology.disk(3))
    bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
    cleared = skimage.segmentation.clear_border(bw2)
    im_labeled = skimage.measure.label(cleared)
    image_label_overlay = skimage.color.label2rgb(
        im_labeled, bg_label=0, colors=colors_li)
    c, r = divmod(i, 3)
    ax[r, c].imshow(image_label_overlay)
    ax[r, c].set_title(f'{img.name} - gaussian {block_size}')
    i += 1
fig.savefig(res_dir/'local_thresh_gaussian_with_sum_by_channels.png')
plt.tight_layout()
plt.show()
# %%


# fig, ax = plt.subplots(figsize=(10, 6))
# ax.imshow(image_label_overlay)
# for region in skimage.measure.regionprops(im_labeled, intensity_image=im_gray):
