import os
from importlib import reload
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import matplotlib as mpl

import data_utils as du
import plotting_utils as pu
import imageproc_utils as ipu
from texture import texture_analysis as ta

ff_dir = '/home/billy/Research/Projects/08_30_airGap_thickerSlabs/1000umCdTe/40kvp/ff/'
img_dir = '/home/billy/Research/Projects/08_30_airGap_thickerSlabs/1000umCdTe/40kvp/pos_4cm/thk_51mm/'

ff_data = du.import_data(ff_dir)
img_data = du.import_data(img_dir)

ff_data = ff_data[0:2]
img_data = img_data[0:2]
img_data.show_imgs()
img_data.ff_corr(ff_data)
img_data.quantize_imgs(range_mode='manual')
img_data.show_imgs(img_type='quantized')
# img_data.ff_corr(ff_data)
#
# img_data.show_imgs()
# TH0 = ipu.ThresholdHistogram(ff_data[0],mode='mask')
# TH1 = ipu.ThresholdHistogram(ff_data[1],mode='mask')
# mask0 = TH0.mask
# mask1 = TH1.mask
#
# bad_mask = np.logical_or(mask0,mask1)
# print(bad_mask)
# np.savetxt('bad_pixel_mask_cdte',bad_mask)

# img_data.ff_corr(ff_data)
# slides = pu.ImageSlideshow(img_data)