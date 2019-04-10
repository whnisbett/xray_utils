import data_utils as du
import csv
import imageproc_utils as ipu
from texture import texture_analysis as ta

import matplotlib.pyplot as plt

img_dir = '/Users/billy/Research/project-data/Texture/Texture Analysis Data/dbt_imgs/60deg_19proj'
lattice_list = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
square_lattice_list = []
img_list = du.import_data(img_dir)

# for every image in list, split into a couple of different lattices of rois

img_list.quantize_imgs(range_mode='auto')
roi_list = []
for datum in img_list:
    rois = ipu.lattice_roi_splitter(datum.quant_img, lattice_dims=(16,16))
    roi_list.append(rois)
feature_list = {'entropy': [], 'energy': [], 'inverse difference moment':
    [], 'correlation': [], 'busyness': [], 'complexity': [], 'coarseness': [], 'contrast': []}
texture_datum_list = []
for roi_set in roi_list:

    texture_datum = ta.TextureDatum(roi_set)
    texture_datum_list.append(texture_datum)

for texture_datum in texture_datum_list:
    for feature in feature_list.keys():
        feature_list[feature].append(texture_datum.feature_list[feature])

for feature in feature_list.keys():
    feature_list[feature] = [value for sublist in feature_list[feature] for value in sublist]

with open('scripts/feature_gaussianity_19x16.csv','w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(feature_list.keys())
    writer.writerows(zip(*feature_list.values()))