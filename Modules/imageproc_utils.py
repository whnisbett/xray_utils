import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

import global_variables
import stat_utils as su


class ThresholdHistogram:
    def __init__(self, img_datum, cmap='viridis', mode=''):
        # initialize variables
        self.img_datum = img_datum
        self.color_map = cmap
        self.img = self.img_datum.img
        self.mode = mode
        self.is_picked = False
        self.curr_line = None
        self.mask = None
        self.masked_img = None

        # prepare data to create histogram
        r, c = self.img.shape
        self.img[np.isnan(self.img)] = 0
        img1d = np.reshape(self.img, r * c)
        # get rid of outliers **** and determine starting threshold for image viewing/histogram
        self.hist_data = img1d

        self.thresh_low, self.thresh_high = (np.percentile(su.filter_outliers(self.hist_data), 0.5),
                                             np.percentile(su.filter_outliers(self.hist_data), 99.5))

        # make plot interactive
        plt.ioff()
        # create figures and axes subplots
        self.fig = plt.figure(figsize=(16, 7))
        self.fig.suptitle(self.img_datum.filename)
        self.hist_ax = self.fig.add_subplot(121)
        self.img_ax = self.fig.add_subplot(122)
        # draw the histogram and images
        self.draw_hist()
        self.draw_image()
        # format plotting domain
        xmin = np.percentile(self.hist_data, 0.5)
        xmax = np.percentile(self.hist_data, 99.5)
        xmin -= 0.1 * (xmax - xmin)
        xmax += 0.1 * (xmax - xmin)
        self.hist_ax.set_xlim(xmin, xmax)
        # plot threshold lines and make them pickable in 5 pixel radius
        ymin, ymax = self.hist_ax.get_ylim()
        self.line_high = self.hist_ax.vlines(x=[self.thresh_high], ymin=ymin, ymax=ymax)
        self.line_high.set_picker(5)
        self.line_low = self.hist_ax.vlines(x=[self.thresh_low], ymin=ymin, ymax=ymax)
        self.line_low.set_picker(5)

        # connect all event listeners
        self.connect()
        plt.show(block=True)

    def connect(self):
        # begins listening for artist picks, button releases, and motion events
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.fig.canvas.mpl_connect('close_event', self.onclose)

    def onclose(self, event):
        plt.close(self.fig)

    def onpick(self, event):
        # when an artist is picked (i.e. a threshold line), make it red and change state of object plot to "is pressed"
        self.curr_line = event.artist
        self.curr_line.set_color((1, 0, 0, 1))
        self.is_picked = True
        self.fig.canvas.draw()

    def onrelease(self, event):
        # when the button is released, if curr_line is assigned then turn it back to black, change state back to "is
        # not pressed" and clear curr_line
        if isinstance(self.curr_line, mpl.collections.LineCollection):
            self.curr_line.set_color((0, 0, 0, 1))

        self.is_picked = False
        self.curr_line = None
        self.fig.canvas.draw()

    def onmove(self, event):
        # updates image and histogram when an artist is picked and mouse is moving
        if not self.is_picked:
            return
        else:
            # if the mouse is inside of the axes bounds, update move curr_line to mouse x position
            mouse_x = event.xdata
            if self.hist_ax.get_xlim()[0] < mouse_x < self.hist_ax.get_xlim()[1]:
                seg = self.curr_line.get_segments()
                seg[0][0][0], seg[0][1][0] = (mouse_x, mouse_x)
                self.curr_line.set_segments(seg)
            # sort lines based on x coordinates and assign first line to line_low and second to line_high
            lines = self.hist_ax.collections
            lines.sort(key=lambda line: line.get_segments()[0][0][0])
            self.line_low = lines[0]
            self.line_high = lines[1]

            # update the threshold values and redraw the image with new thresholds
            self.update_thresholds()
            self.img_ax.clear()
            self.draw_image()
            self.fig.canvas.draw()

    def update_thresholds(self):
        # update threshold values based on position of threshold lines
        self.thresh_low = self.line_low.get_segments()[0][0][0]
        self.thresh_high = self.line_high.get_segments()[0][0][0]

    def update_mask(self):
        # create masked image based on threshold values
        mask_low = self.img < self.thresh_low
        mask_high = self.img > self.thresh_high
        self.mask = np.logical_or(mask_high, mask_low)
        self.masked_img = np.ma.masked_array(self.img, self.mask)
        self.masked_img.set_fill_value(0)

    def draw_image(self):
        # plot the img on img_ax using the designated mode and current threshold values
        if self.mode != 'mask':
            self.img_plot = self.img_ax.imshow(self.img, cmap=self.color_map, vmin=self.thresh_low,
                                               vmax=self.thresh_high)
        else:
            self.update_mask()
            self.img_plot = self.img_ax.imshow(self.masked_img, cmap=self.color_map, vmin=self.thresh_low,
                                               vmax=self.thresh_high)

    def draw_hist(self):
        # plot histogram on hist_ax based on hist_data
        self.img_hist = self.hist_ax.hist(self.hist_data, bins=256)


def quantize_img(img_datum, n_grey=256, range_mode='auto', quant_method='uniform'):
    img = img_datum.img
    if quant_method == 'uniform':
        if range_mode == 'manual':
            th = ThresholdHistogram(img_datum)
            quant_min = th.thresh_low
            quant_max = th.thresh_high
        elif range_mode == 'auto':
            quant_min = np.percentile(su.filter_outliers(img.flatten()), 0.5)
            quant_max = np.percentile(su.filter_outliers(img.flatten()), 99.5)

        relative_bins = (img - quant_min) / (quant_max - quant_min)

        # FIND ALL VALUES THAT ARE LESS THAN THE MIN THRESHOLD
        nonzero_values = relative_bins >= 0
        nonzero_values = nonzero_values.astype(np.int)
        nonzero_img = np.floor(relative_bins * nonzero_values * (n_grey - 2))
        max_values = nonzero_img >= (n_grey - 2)
        quantized_img = np.ma.array(nonzero_img, mask=max_values)
        quantized_img = np.ma.filled(quantized_img, fill_value=n_grey - 1)

        return quantized_img.astype(np.int)


def ff_correct(img_datum, ff_img_datum, mode='auto'):
    if mode == 'manual':
        ff_hist = ThresholdHistogram(ff_img_datum, mode='mask')
        img_hist = ThresholdHistogram(img_datum, mode='mask')

        ff_mask = ff_hist.mask
        img_mask = img_hist.mask
        mask = np.logical_or(img_mask, ff_mask)

    elif mode == 'auto':
        r, c = ff_img_datum.img.shape
        ff_1d = np.reshape(ff_img_datum.img, r * c)
        ff_1d_no_outliers = ff_1d  # su.filter_outliers(ff_1d)
        high_thresh = np.percentile(ff_1d_no_outliers, 99)
        low_thresh = np.percentile(ff_1d_no_outliers, 1)
        high_mask = ff_img_datum.img > high_thresh
        low_mask = ff_img_datum.img < low_thresh
        ff_mask = np.logical_or(high_mask, low_mask)

        r, c = img_datum.img.shape
        img_1d = np.reshape(img_datum.img, r * c)
        img_1d_no_outliers = img_1d
        high_thresh = np.percentile(img_1d_no_outliers, 99)
        low_thresh = np.percentile(img_1d_no_outliers, 0.5)
        high_mask = img_datum.img > high_thresh
        low_mask = img_datum.img < low_thresh
        img_mask = np.logical_or(high_mask, low_mask)

        mask = np.logical_or(ff_mask, img_mask)

    mask = np.logical_or(mask, global_variables.cdte_pix_mask)
    ff_interp = correctBadPixels(ff_img_datum.img, mask)
    img_interp = correctBadPixels(img_datum.img, mask)
    # img_interp = correctBadPixels(img,img_mask)
    # crop both of the images to the same size
    img_ffcorr = img_interp.astype('float') / ff_interp.astype('float')
    img_ffcorr[np.isnan(img_ffcorr)] = 1E-10
    img_ffcorr[img_ffcorr <= 0.0] = 1E-10

    return img_ffcorr


def correctBadPixels(img, mask):  # runs prior to flatfield correction and interpolates all of the data
    # img_hist = thresholdingHistogram(img,mode = 'mask')
    # img_mask = img_hist.mask
    img_mask = mask
    x_mesh, y_mesh = np.mgrid[0:256, 0:256]
    # x_mesh,y_mesh = np.mgrid[0:246,0:246]
    mask_inv = img_mask != True
    points = np.vstack((x_mesh[mask_inv], y_mesh[mask_inv]))
    points = points.T
    values = img[mask_inv]
    values = values.T

    grid = griddata(points, values, (x_mesh, y_mesh), method='linear')
    # plt.imshow(grid,cmap = 'viridis',vmin = np.percentile(grid,1),vmax = np.percentile(grid,99))
    return grid
    # plt.imshow(grid,cmap = 'magma',vmin = np.percentile(grid,5),vmax = np.percentile(grid,95))


def select_roi(img, ul_coordinates):
    # ul_coordinates are the upper left coordinates and dimensions of the roi in the form
    # [r,c,vertical_width,horizontal_width]
    r_start = ul_coordinates[0]
    c_start = ul_coordinates[1]
    v_span = ul_coordinates[2]
    h_span = ul_coordinates[3]
    roi = img[r_start:r_start + v_span, c_start:c_start + h_span]

    return roi
