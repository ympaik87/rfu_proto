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
im_dir = pathlib.Path('F:/device/optics/images/32w_4th_RTtest_data/2')
im_dir
# %%
temp_li = ['Low Temp', 'High Temp']
x_range = slice(400, 1900)
y_range = slice(400, 1900)
colors_li = [plt.cm.get_cmap('hsv', 30)(i) for i in range(30)]
# %%
ch_dict = {
    0: 'CalRed',
    1: 'FAM',
    2: 'Q670',
    3: 'HEX'
}
# %%
ch_dict
# %%
im_dict = {}
for ch in range(4):
    ch_name = ch_dict[ch]
    im_dict[ch_name] = {}
    for t in range(2):
        im_dict[ch_name][temp_li[t]] = {}
        for c in range(45):
            im_dict[ch_name][temp_li[t]][c] = {}
            fpath = im_dir/f'{c}_{t}_{ch}.jpg'
            im = np.array(Image.open(fpath))
            im_cropped = im[x_range, y_range]

            im_hsv = cv2.cvtColor(im_cropped, cv2.COLOR_BGR2HSV)
            im_gray = im_hsv[:, :, 2]

            thresh = skimage.filters.threshold_otsu(im_gray)
            threshed_im = im_gray > thresh
            im_dict[ch_name][temp_li[t]][c]['thresh'] = threshed_im

            bw = skimage.morphology.closing(threshed_im, skimage.morphology.disk(3))
            bw2 = skimage.morphology.opening(bw, skimage.morphology.disk(3))
            cleared = skimage.segmentation.clear_border(bw2)
            im_labeled = skimage.measure.label(cleared)
            im_dict[ch_name][temp_li[t]][c]['labeled'] = im_labeled

            region_raw_dict = {}
            for region in skimage.measure.regionprops(im_labeled, intensity_image=im_gray):
                region_raw_dict[region.area] = region
            im_dict[ch_name][temp_li[t]][c]['regions'] = region_raw_dict

# %%
result_dict = {}
for temp in temp_li:
    result_dict[temp] = {}
    for ch in ch_dict.values():
        result_dict[temp][ch] = {}
# %%
im_dict[ch][temp][44]['regions']
# %%
plt.figure(figsize=(12, 12))
plt.imshow(im_dict[ch][temp][44]['labeled'])
# %%
row = list('ABCD')[::-1]
result_dict = {'well_grid': {}}
for temp in temp_li:
    result_dict['well_grid'][temp] = {}

box_li = []
for ch in ch_dict.values():
    for temp in temp_li:
        regions = im_dict[ch][temp][44]['regions']

        areas_li = []
        for area, region in regions.items():
            areas_li.append(area)

        sorted_region_key = list(areas_li)
        sorted_region_key.sort(reverse=True)

        minr_li = []
        minc_li = []
        maxr_li = []
        maxc_li = []
        for key in sorted_region_key[:16]:
            region = regions[key]
            minr, minc, maxr, maxc = region.bbox
            minr_li.append(minr)
            minc_li.append(minc)
            maxr_li.append(maxr)
            maxc_li.append(maxc)
        minr_li.sort()
        top = minr_li[0]
        minc_li.sort()
        left = minc_li[0]
        maxr_li.sort()
        bottom = maxr_li[-1]
        maxc_li.sort()
        right = maxc_li[-1]
        box_li.append([top, left, bottom, right])

box_arr = np.array(box_li)
top = box_arr[:, 0].mean()
left = box_arr[:, 1].mean()
bottom = box_arr[:, 2].mean()
right = box_arr[:, 3].mean()

well_box = [top-50, left-50, bottom+50, right+50]
y_li = np.linspace(well_box[0], well_box[2], 5, endpoint=True)
x_li = np.linspace(well_box[1], well_box[3], 5, endpoint=True)
pts_x = []
pts_y = []
for x in x_li:
    for y in y_li:
        pts_x.append(x)
        pts_y.append(y)
pts_li = list(zip(pts_x, pts_y))

i = 0
well_location_dict = {}
for x in range(4):
    for y in range(4):
        key = row[y]+str(x+1)
        top_left_pt = pts_li[i+y+x]
        bottom_right_pt = pts_li[i+y+x+6]
        well_location_dict[key] = [top_left_pt[1], top_left_pt[0], bottom_right_pt[1], bottom_right_pt[0]]
    i += 4
result_dict['well_grid'] = well_location_dict


# %%
result_dict['well_grid']
# %%
def get_well_loc(x, y, pts_center, well_location_dict):
    for well in well_location_dict.keys():
        y_min, x_min, y_max, x_max = well_location_dict[well]
        if y_min < y < y_max and x_min < x < x_max:
            radius = (x_max-x_min)/2 - 50
            pts_given = np.array([x, y])
            distance = np.linalg.norm(pts_given-pts_center)
            if distance < radius:
                return well
# %%
def get_grid_loc(x, y, well_location_dict):
    for well in well_location_dict.keys():
        y_min, x_min, y_max, x_max = well_location_dict[well]
        if y_min < y < y_max and x_min < x < x_max:
            return well
# %%
region_sum_dict = {}
for t in range(2):
    temp = temp_li[t]
    region_sum_dict[temp] = {}
    for ch in ch_dict.values():
        region_sum_dict[temp][ch] = {}
        for well in well_location_dict.keys():
            region_sum_dict[temp][ch][well] = [0]*45

        for c in range(45):
            sorted_region_key = list(im_dict[ch][temp][c]['regions'].keys())
            sorted_region_key.sort(reverse=True)

            center_at_cycle = {}
            for key in sorted_region_key:
                region_obj = im_dict[ch][temp][c]['regions'][key]
                y, x = region_obj.centroid
                grid = get_grid_loc(x, y, result_dict['well_grid'])
                if grid is None:
                    continue

                if grid not in center_at_cycle.keys():
                    center = [x, y]
                    center_at_cycle[grid] = center
                else:
                    center = center_at_cycle[grid]
                well = get_well_loc(x, y, center, result_dict['well_grid'])

                if well is not None:
                    val = region_sum_dict[temp][ch][well][c]
                    val += region_obj.intensity_image.sum()
                    region_sum_dict[temp][ch][well][c] = val

result_dict['rfu_sum'] = region_sum_dict

# %%
ch_dict.values()
# %%
dict_before = pd.read_excel('F:/device/optics/32w_Realtime_test_5th/5차 2번카메라.xlsx', sheet_name=None, header=None)
dict_before.keys()
# %%
df = dict_before['fam60']
df
# %%
len(df)
# %%
map_dict = {
    'CalRed': 'cal',
    'FAM': 'fam',
    'HEX': 'hex',
    'Q670': 'qua',
    'L': '60',
    'H': '72'
}
row = list('ABCD')[::-1]
col_li = []
for y in range(4):
    for x in range(4):
        key = row[y]+str(x+1)
        col_li.append(key)
col_li
# %%
result_dict['rfu_sum']['Low Temp'].keys()
# %%
pd.DataFrame(result_dict['rfu_sum']['High Temp']['Q670'])
# %%
colors_li2 = [plt.cm.get_cmap('tab20c')(i) for i in range(20)]
# %%
for temp in result_dict['rfu_sum'].keys():
    for ch, _dict in result_dict['rfu_sum'][temp].items():
        df = pd.DataFrame(_dict)
        df.index = range(1, 46)
        df = df.reindex(sorted(df.columns), axis=1)
        fig = df.plot(figsize=(12, 6), color=colors_li2, title=f'{temp} {ch}').get_figure()
        fig.savefig(f'camera2_hsv_otsu2_{temp}_{ch}.png')

        key_before = map_dict[ch]+map_dict[temp[0]]
        df_before = dict_before[key_before]
        df_before.columns = col_li
        df_before.index = range(1, len(df_before)+1)

        df_before_norm = (df_before - df_before.loc[10:20].mean())/(df_before.max()-df_before.loc[10:20].mean())
        df_norm = (df-df.loc[10:20].mean())/(df.max()-df.loc[10:20].mean())

        fig, axes = plt.subplots(4, 4, figsize=(24, 12))
        n = 0
        for col in df_norm.columns:
            i = n%4
            j = n//4
            axes[i, j].plot(df_before_norm[col], label='before')
            axes[i, j].plot(df_norm[col], label='after')
            axes[i, j].set_title(f'{temp}-{ch}-{col}')
            n += 1
            axes[i, j].legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'camera2_hsv_otsu2_compare_{temp}_{ch}.png')
# %% markdown
# # investigate High Q670
# %%
import matplotlib.patches as _patches
# %%
fig, ax = plt.subplots(10, 9, figsize=(24*5, 12*5))
region_intensity_dict = {}
for t in range(2):
    region_intensity_dict[temp_li[t]] = {}
    for well in well_location_dict.keys():
        region_intensity_dict[temp_li[t]][well] = [0]*45

    for c in range(45):
        im_labeled = im_dict['Q670'][temp_li[t]][c]['labeled']
        image_label_overlay = skimage.color.label2rgb(im_labeled, bg_label=0, colors=colors_li)
        ax_x_coord = c//9+(t*5)
        ax_y_coord = c%9

        ax[ax_x_coord, ax_y_coord].imshow(image_label_overlay)
        ax[ax_x_coord, ax_y_coord].set_title(f'{temp_li[t]} - cycle {c+1}')
        ax[ax_x_coord, ax_y_coord].scatter(pts_x, pts_y, c='g')

        sorted_region_key = list(im_dict['Q670'][temp_li[t]][c]['regions'].keys())
        sorted_region_key.sort(reverse=True)

        center_at_cycle = {}
        circle_li = []
        for key in sorted_region_key:
            region_obj = im_dict['Q670'][temp_li[t]][c]['regions'][key]
            y, x = region_obj.centroid
            grid = get_grid_loc(x, y, result_dict['well_grid'][temp_li[t]]['Q670'])
            if grid is None:
                continue

            if grid not in center_at_cycle.keys():
                center = [x, y]
                center_at_cycle[grid] = center

                y_min, x_min, y_max, x_max = well_location_dict[grid]
                grid_center_x = (x_max-x_min)/2 + x_min
                rad = grid_center_x - x_min - 50
                circle_li.append(_patches.Circle(center, radius=rad, color='r', fill=False, linewidth=1))
            else:
                center = center_at_cycle[grid]
            well = get_well_loc(x, y, center, result_dict['well_grid'][temp_li[t]]['Q670'])

            if well is not None:
                val = region_intensity_dict[temp_li[t]][well][c]
                val += region_obj.intensity_image.sum()
                region_intensity_dict[temp_li[t]][well][c] = val

                ax[ax_x_coord, ax_y_coord].plot(x, y, color='white', marker='*')
                ax[ax_x_coord, ax_y_coord].text(x, y, well, color='gray')
            else:
                ax[ax_x_coord, ax_y_coord].plot(x, y, color='b', marker='x')
            for circle in circle_li:
                ax[ax_x_coord, ax_y_coord].add_artist(circle)

plt.tight_layout()
plt.savefig('hsv_otsu_camera2_q670_circle.png')
plt.show()
# %%
df_high_q = pd.DataFrame(region_intensity_dict['High Temp'])
# %%
pd.DataFrame(region_intensity_dict['High Temp']).plot()
# %%
df_before = dict_before['qua72']
df_before.columns = col_li
df_before.index = range(1, len(df_before)+1)
# %%
df_high_q.index = range(1, 46)
df_high_q = df_high_q.reindex(sorted(df_high_q.columns), axis=1)

df_before_norm = (df_before - df_before.loc[10:20].mean())/(df_before.max()-df_before.loc[10:20].mean())
df_high_q_norm = (df_high_q-df_high_q.loc[10:20].mean())/(df_high_q.max()-df_high_q.loc[10:20].mean())
# %%
fig, axes = plt.subplots(4, 4, figsize=(24, 12))
n = 0
for col in df_before_norm.columns:
    i = n%4
    j = n//4
    axes[i, j].plot(df_before_norm[col], label='before')
    axes[i, j].plot(df_high_q_norm[col], label='after')
    axes[i, j].set_title(col)
    n += 1
    axes[i, j].legend(loc='best')
plt.tight_layout()
plt.show()
# %% markdown
# # Low Cal Red
# %%
pts_x = []
pts_y = []
for key, val in result_dict['well_grid'].items():
    y1, x1, y2, x2 = val
    pts_x.extend([x1, x2])
    pts_y.extend([y1, y2])
# %%
fig, ax = plt.subplots(10, 9, figsize=(24*5, 12*5))
region_intensity_dict = {}
for t in range(2):
    region_intensity_dict[temp_li[t]] = {}
    for well in well_location_dict.keys():
        region_intensity_dict[temp_li[t]][well] = [0]*45

    for c in range(45):
        im_labeled = im_dict['CalRed'][temp_li[t]][c]['labeled']
        image_label_overlay = skimage.color.label2rgb(im_labeled, bg_label=0, colors=colors_li)
        ax_x_coord = c//9+(t*5)
        ax_y_coord = c%9

        ax[ax_x_coord, ax_y_coord].imshow(image_label_overlay)
        ax[ax_x_coord, ax_y_coord].set_title(f'{temp_li[t]} - cycle {c+1}')
        ax[ax_x_coord, ax_y_coord].scatter(pts_x, pts_y, c='g')

        sorted_region_key = list(im_dict['CalRed'][temp_li[t]][c]['regions'].keys())
        sorted_region_key.sort(reverse=True)

        center_at_cycle = {}
        circle_li = []
        for key in sorted_region_key:
            region_obj = im_dict['CalRed'][temp_li[t]][c]['regions'][key]
            y, x = region_obj.centroid
            grid = get_grid_loc(x, y, result_dict['well_grid'])
            if grid is None:
                continue

            if grid not in center_at_cycle.keys():
                center = [x, y]
                center_at_cycle[grid] = center

                y_min, x_min, y_max, x_max = well_location_dict[grid]
                grid_center_x = (x_max-x_min)/2 + x_min
                rad = grid_center_x - x_min - 50
                circle_li.append(_patches.Circle(center, radius=rad, color='r', fill=False, linewidth=1))
            else:
                center = center_at_cycle[grid]
            well = get_well_loc(x, y, center, result_dict['well_grid'])

            if well is not None:
                val = region_intensity_dict[temp_li[t]][well][c]
                val += region_obj.intensity_image.sum()
                region_intensity_dict[temp_li[t]][well][c] = val

                ax[ax_x_coord, ax_y_coord].plot(x, y, color='white', marker='*')
                ax[ax_x_coord, ax_y_coord].text(x, y, well, color='gray')
            else:
                ax[ax_x_coord, ax_y_coord].plot(x, y, color='b', marker='x')
            for circle in circle_li:
                ax[ax_x_coord, ax_y_coord].add_artist(circle)

plt.tight_layout()
plt.savefig('hsv_otsu_camera2_calRed_circle2.png')
plt.show()
# %%
fpath = im_dir/'7_0_0.jpg'
im = np.array(Image.open(fpath))
plt.imshow(im)
# %%
im_cropped = im[x_range, y_range]

im_hsv = cv2.cvtColor(im_cropped, cv2.COLOR_BGR2HSV)
im_gray = im_hsv[:, :, 2]
plt.imshow(im_gray)
# %%
im_cropped = im[x_range, y_range]
plt.imshow(im_cropped[:, :, 0])
# %%
plt.imshow(im_gray)
# %%
otsu = skimage.filters.threshold_otsu(im_gray)
mean = skimage.filters.threshold_mean(im_gray)
print(otsu, mean)
# %%
plt.hist(im_gray.ravel(), bins=100)
# %%
plt.imshow(im_dict['CalRed'][temp_li[0]][7]['thresh'])
# %% markdown
# # High HEX
# %%
pts_low_x = []
pts_low_y = []
for key, val in result_dict['well_grid']['Low Temp']['HEX'].items():
    y1, x1, y2, x2 = val
    pts_low_x.extend([x1, x2])
    pts_low_y.extend([y1, y2])
pts_high_x = []
pts_high_y = []
for key, val in result_dict['well_grid']['High Temp']['HEX'].items():
    y1, x1, y2, x2 = val
    pts_high_x.extend([x1, x2])
    pts_high_y.extend([y1, y2])
pts_x = [pts_low_x, pts_high_x]
pts_y = [pts_low_y, pts_high_y]
# %%
fig, ax = plt.subplots(10, 9, figsize=(24*5, 12*5))
region_intensity_dict = {}
for t in range(2):
    region_intensity_dict[temp_li[t]] = {}
    for well in well_location_dict.keys():
        region_intensity_dict[temp_li[t]][well] = [0]*45

    for c in range(45):
        im_labeled = im_dict['HEX'][temp_li[t]][c]['labeled']
        image_label_overlay = skimage.color.label2rgb(im_labeled, bg_label=0, colors=colors_li)
        ax_x_coord = c//9+(t*5)
        ax_y_coord = c%9

        ax[ax_x_coord, ax_y_coord].imshow(image_label_overlay)
        ax[ax_x_coord, ax_y_coord].set_title(f'{temp_li[t]} - cycle {c+1}')
        ax[ax_x_coord, ax_y_coord].scatter(pts_x[t], pts_y[t], c='g')

        sorted_region_key = list(im_dict['HEX'][temp_li[t]][c]['regions'].keys())
        sorted_region_key.sort(reverse=True)

        center_at_cycle = {}
        circle_li = []
        for key in sorted_region_key:
            region_obj = im_dict['HEX'][temp_li[t]][c]['regions'][key]
            y, x = region_obj.centroid
            grid = get_grid_loc(x, y, result_dict['well_grid'][temp_li[t]]['HEX'])
            if grid is None:
                continue

            if grid not in center_at_cycle.keys():
                center = [x, y]
                center_at_cycle[grid] = center

                y_min, x_min, y_max, x_max = well_location_dict[grid]
                grid_center_x = (x_max-x_min)/2 + x_min
                rad = grid_center_x - x_min - 50
                circle_li.append(_patches.Circle(center, radius=rad, color='r', fill=False, linewidth=1))
            else:
                center = center_at_cycle[grid]
            well = get_well_loc(x, y, center, result_dict['well_grid'][temp_li[t]]['HEX'])

            if well is not None:
                val = region_intensity_dict[temp_li[t]][well][c]
                val += region_obj.intensity_image.sum()
                region_intensity_dict[temp_li[t]][well][c] = val

                ax[ax_x_coord, ax_y_coord].plot(x, y, color='white', marker='*')
                ax[ax_x_coord, ax_y_coord].text(x, y, well, color='white')
            else:
                ax[ax_x_coord, ax_y_coord].plot(x, y, color='b', marker='x')
            for circle in circle_li:
                ax[ax_x_coord, ax_y_coord].add_artist(circle)

plt.tight_layout()
plt.savefig('hsv_otsu_camera2_HEX_circle.png')
plt.show()
# %%
get_grid_loc(626, 104, result_dict['well_grid'][temp_li[1]]['HEX'])
# %%
result_dict['well_grid'][temp_li[1]]['HEX']
# %%
fpath = im_dir/'18_1_3.jpg'
im = np.array(Image.open(fpath))
# %%
rect = _patches.Rectangle((400, 400), 1500, 1500,linewidth=1,edgecolor='r',facecolor='none')
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(im)
ax.add_patch(rect)
# %%
im_cropped = im[x_range, y_range]
plt.imshow(im_cropped[:, :, 0])
# %%
