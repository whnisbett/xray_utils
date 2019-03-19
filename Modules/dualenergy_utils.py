import numpy as np
from matplotlib import pyplot as plt
import glob
import image_utils as iu
from general import roipoly as roi


def sumFrames(TH_delta, n_frames = 'all', dir = './'):
    # get file prefix, initialize empty image matrix, import
    files = glob.glob(dir + '*.txt')
    files.sort()
    file1 = files[0]
    file_parts = file1.split('_')
    th_min = int(file_parts[-2])

    file_last = files[-1]
    file_parts = file_last.split('_')
    maxframe = file_parts[-1].split('.')[0]
    maxframe = int(maxframe.lstrip('0'))
    THmax = int(file_parts[-2])
    n_TH = (THmax -     th_min) / TH_delta + 1

    files = np.array(files)
    files = np.reshape(files, (n_TH, maxframe + 1))
    summed_images = np.loadtxt(files[0, 0])
    r, c = summed_images.shape
    summed_images = np.zeros((n_TH, r, c))
    if n_frames == 'all': n_frames = maxframe + 1
    for i_TH in np.arange(n_TH):
        for i_frame in np.arange(n_frames):
            img = np.loadtxt(files[i_TH, i_frame])
            summed_images[i_TH] += img
    return summed_images


def getBinImage(THL, THH):  # tuples with ff and img entries for each threshold

    THL_ff = THL[0]
    THL_img = THL[1]
    THH_ff = THH[0]
    THH_img = THH[1]

    bin_img = THL_img - THH_img
    bin_ff = THL_ff - THH_ff
    bin_img_ffcorr = iu.ff_correct(bin_img, bin_ff)
    return bin_img_ffcorr


def getMueff(img, thickness):
    plt.imshow(img, cmap = 'viridis', vmin = np.percentile(img, 1), vmax = np.percentile(img, 99))
    roi_samp = roi.roipoly()
    wait = raw_input('PRESS ENTER TO CONTINUE')
    roi_samp = img[roi_samp.getMask(img)]
    mu_eff = -np.log(roi_samp) / thickness
    mu_mean = np.nanmean(mu_eff)
    mu_std = np.std(mu_eff)

    return (mu_mean, mu_std)


def deSubtract(THL_img, THH_img, thickness):
    mu_low = getMueff(THL_img, thickness)
    mu_high = getMueff(THH_img, thickness)
    print(mu_high[0], mu_low[0])
    R = mu_high[0] / mu_low[0]
    sub_img = np.log(THL_img) * R - np.log(THH_img)
    print(R)
    return sub_img
