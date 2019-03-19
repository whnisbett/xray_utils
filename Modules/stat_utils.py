import numpy as np
import copy
from matplotlib import pyplot as plt
import roipoly as roi
import os


def filter_outliers(points, thresh = 15):
    med = np.median(points)
    diff = (points - med)**2
    diff = np.sqrt(diff)
    med_dev = np.median(diff)
    s = diff / med_dev

    return points[s < thresh]


def wls_estimate(output_mat, coeff_mat, w_mat):
    weight_mat = copy.copy(w_mat)
    weight_mat = weight_mat / float(np.sum(weight_mat))

    term2 = np.matmul(weight_mat, output_mat)
    term2 = np.matmul(np.transpose(coeff_mat), term2)

    term1 = np.matmul(weight_mat, coeff_mat)
    term1 = np.matmul(np.transpose(coeff_mat), term1)

    term1 = np.linalg.inv(term1)

    input_mat = np.matmul(term1, term2)

    return input_mat


def getcnr(directory = os.getcwd(), img = np.array([])):
    if len(img) == 0:
        cnr_list = []
        filelist = os.listdir(directory)
        for fname in reversed(filelist):
            if fname.endswith('txt'):
                img = np.loadtxt(fname)
                cnr_list.append(getcnr(img = img))
            else:
                filelist.remove(fname)
                continue
        return cnr_list, filelist
    else:
        plt.imshow(img, cmap = 'magma')
        plt.title('Select Sample ROI')
        sample_roi = roi.roipoly()
        # wait = raw_input('PRESS ENTER TO CONTINUE')
        input('PRESS ENTER TO CONTINUE')
        plt.imshow(img, cmap = 'magma')
        plt.title('Select Background ROI')
        background_roi = roi.roipoly()
        # wait = raw_input('PRESS ENTER TO CONTINUE')
        input('PRESS ENTER TO CONTINUE')
        sample_roi = img[sample_roi.getMask(img)]
        background_roi = img[background_roi.getMask(img)]

        sample_mean = np.mean(sample_roi)
        background_mean = np.mean(background_roi)
        background_std = np.std(background_roi)

    cnr = (sample_mean - background_mean) / background_std
    return cnr


def getsnr(img):
    plt.imshow(img, cmap = 'magma')
    plt.title('Select Sample ROI')
    sample_roi = roi.roipoly()
    # wait = raw_input('PRESS ENTER TO CONTINUE')
    input('PRESS ENTER TO CONTINUE')
    plt.imshow(img, cmap = 'magma')
    plt.title('Select Background ROI')
    background_roi = roi.roipoly()
    # wait = raw_input('PRESS ENTER TO CONTINUE')
    input('PRESS ENTER TO CONTINUE')
    sample_roi = img[sample_roi.getMask(img)]
    background_roi = img[background_roi.getMask(img)]

    sample_mean = np.mean(sample_roi)
    background_mean = np.mean(background_roi)
    background_std = np.sqrt(np.std(background_roi)**2 + np.std(sample_roi)**2)
    print(sample_mean, background_mean, background_std)
    snr = (sample_mean - background_mean) / background_std
    return snr
