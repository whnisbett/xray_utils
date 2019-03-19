import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import stat_utils as su


class threshHist:
    # IMPORTANT PARAMETERS:
    # img - raw image
    # thresh_low/thresh_high - the low and high thresholds
    # mask - the mask basked on the thresholds
    # masked_img - the masked_array of the image
    def __init__(self, img, mode = '', cmap = 'viridis'):
        # CREATE FIGURE AND PLOT HISTOGRAM
        plt.ion()

        self.color_map = cmap
        self.mode = mode
        self.img = img
        r, c = self.img.shape
        self.is_pressed = False

        self.img[np.isnan(self.img)] = 0. # replace nan with 0
        # replace outliers with 0???
        img1d = np.reshape(self.img, r * c)
        self.outlier_free_data = img1d #su.filter_outliers(img1d)
        # set initial threshold
        self.thresh_low, self.thresh_high = (np.percentile(self.outlier_free_data, 1),
                                             np.percentile(self.outlier_free_data, 99))

        self.hist_fig = plt.figure(1)
        plt.show()
        self.hist_ax = self.hist_fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # self.img_hist = self.hist_ax.hist(self.img1d, bins = 256)
        self.img_hist = self.hist_ax.hist(self.outlier_free_data, bins = 256)

        # DRAW LINES AND MAKE THEM PICKABLE
        ymin, ymax = self.hist_ax.get_ylim()
        self.line_low = self.hist_ax.vlines([self.thresh_low], ymin, ymax)
        self.line_low.set_picker(10)
        self.line_high = self.hist_ax.vlines([self.thresh_high], ymin, ymax)
        self.line_high.set_picker(10)

        # DISPLAY IMAGE
        self.img_fig = plt.figure(2)
        plt.show()
        self.img_ax = self.img_fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.img_obj = self.img_ax.imshow(self.img, cmap = self.color_map, vmin = self.thresh_low,
                                          vmax = self.thresh_high)

        # BEGIN LISTENING FOR EVENTS
        self.connect()
        # wait = raw_input('PRESS ENTER TO CONFIRM')
        input('PRESS ENTER TO CONFIRM')
        plt.close('all')

    def __call__(self):
        # REDRAW FIGURE
        self.hist_fig = plt.figure(1)
        self.hist_ax = self.hist_fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.img_hist = self.hist_ax.hist(self.outlier_free_data, bins = 'auto')

        # REDRAW LINES AND MAKE THEM PICKABLE
        ymin, ymax = self.hist_ax.get_ylim()
        self.line_low = self.hist_ax.vlines([self.thresh_low], ymin, ymax)
        self.line_low.set_picker(10)
        self.line_high = self.hist_ax.vlines([self.thresh_high], ymin, ymax)
        self.line_high.set_picker(10)

        # TELLS WHETHER OR NOT MOUSE IS BEING PRESSED
        self.is_pressed = False

        self.img_fig = plt.figure(2)
        self.img_ax = self.img_fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.img_obj = self.img_ax.imshow(self.img, cmap = self.color_map, vmin = self.thresh_low,
                                          vmax = self.thresh_high)

        # BEGIN LISTENING FOR EVENTS
        self.connect()

    def connect(self):
        # CONNECT ALL LISTENERS
        self.pick_id = self.hist_fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.release_id = self.hist_fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.move_id = self.hist_fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_pick(self, event):
        # IDENTIFY LINE BEING CLICKED, MAKE IT RED, AND TURN PRESS ON
        self.curr_line = event.artist
        if self.curr_line == self.line_low:
            self.lineid = 'low'
        else:
            self.lineid = 'high'
        self.curr_line.set_color((1, 0, 0, 1))
        self.is_pressed = True

    def on_move(self, event):
        # IF PRESS IS ON, UPDATE LINE COORDINATES BASED ON MOUSE POSITION
        if not self.is_pressed:
            return
        else:
            x = event.xdata
            if self.hist_ax.get_xlim()[1] > x > self.hist_ax.get_xlim()[0]:
                seg = self.curr_line.get_segments()
                seg[0][0][0], seg[0][1][0] = (x, x)
                self.curr_line.set_segments(seg)
                seg = self.curr_line.get_segments()
                x = seg[0][0][0]

                #####There has to be a more elegent way to do this
                if self.lineid == 'low' and x < self.thresh_low:
                    self.thresh_low = x
                    self.line_low = self.curr_line
                elif self.lineid == 'low' and x > self.thresh_high:
                    self.thresh_low = self.thresh_high
                    self.thresh_high = x
                    self.line_low = self.line_high
                    self.line_high = self.curr_line
                elif self.lineid == 'low' and self.thresh_low < x < self.thresh_high:
                    self.thresh_low = x
                    self.line_low = self.curr_line
                elif self.lineid == 'high' and x > self.thresh_high:
                    self.thresh_high = x
                    self.line_high = self.curr_line
                elif self.lineid == 'high' and x < self.thresh_low:
                    self.thresh_high = self.thresh_low
                    self.thresh_low = x
                    self.line_high = self.line_low
                    self.line_low = self.curr_line
                elif self.lineid == 'high' and self.thresh_low < x < self.thresh_high:
                    self.thresh_high = x
                    self.line_high = self.curr_line

                self.img_obj.remove()
                if self.mode != 'mask':
                    self.img_obj = self.img_ax.imshow(self.img, cmap = self.color_map, vmin = self.thresh_low,
                                                      vmax = self.thresh_high)
                else:
                    mask_low = self.img < self.thresh_low
                    mask_high = self.img > self.thresh_high
                    self.mask = np.array(mask_low) != np.array(mask_high)
                    self.masked_img = np.ma.masked_array(self.img, self.mask)
                    self.masked_img.set_fill_value(0)
                    self.img_obj = self.img_ax.imshow(self.masked_img, cmap = self.color_map, vmin = self.thresh_low,
                                                      vmax = self.thresh_high)

    def on_release(self, event):
        # ON RELEASE CHANGE LINE BACK TO BLACK, CHANGE PRESS STATE, UPDATE THRESHOLDS AND LINE ID'S
        self.curr_line.set_color((0, 0, 0, 1))
        self.is_pressed = False


def ff_correct(img, ff_img):
    # ff_hist = threshHist(ff_img, mode = 'mask')
    # wait = raw_input('PRESS ENTER TO CONTINUE')
    # input('PRESS ENTER TO CONTINUE')
    # plt.close('all')
    # img_hist = thresholdingHistogram(img,mode = 'mask')
    # wait = raw_input('PRESS ENTER TO CONTINUE')
    # plt.close('all')
    #ff_mask = ff_hist.mask
    # img_mask = img_hist.mask
    # SHOULD WE ENSURE THAT THE SAME PIXELS ARE MASKED?

    r,c = ff_img.shape
    ff_1d = np.reshape(ff_img,r*c)
    ff_1d_no_outliers = su.filter_outliers(ff_1d)
    high_threshold = np.max(ff_1d_no_outliers)
    low_threshold = np.min(ff_1d_no_outliers)

    high_mask = ff_img > high_threshold
    low_mask = ff_img < low_threshold
    zero_mask = ff_img < 5.0
    ff_mask = np.logical_or(high_mask,low_mask)
    ff_mask = np.logical_or(ff_mask,zero_mask)


    ff_interp = correctBadPixels(ff_img, ff_mask)
    img_interp = correctBadPixels(img, ff_mask)
    # img_interp = correctBadPixels(img,img_mask)
    # crop both of the images to the same size
    img_ffcorr = img_interp.astype('float') / ff_interp.astype('float')
    img_ffcorr[np.isnan(img_ffcorr)] = 1E-10
    img_ffcorr[img_ffcorr <= 0.0] = 1E-10

    return img_ffcorr


def correctBadPixels(img, mask): # runs prior to flatfield correction and interpolates all of the data
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

    grid = griddata(points, values, (x_mesh, y_mesh), method = 'linear')
    # plt.imshow(grid,cmap = 'viridis',vmin = np.percentile(grid,1),vmax = np.percentile(grid,99))
    return grid
    # plt.imshow(grid,cmap = 'magma',vmin = np.percentile(grid,5),vmax = np.percentile(grid,95))

def select_roi(img,ul_coordinates):
    # ul_coordinates are the upper left coordinates and dimensions of the roi in the form
    # [r,c,vertical_width,horizontal_width]
    r_start = ul_coordinates[0]
    c_start = ul_coordinates[1]
    v_span = ul_coordinates[2]
    h_span = ul_coordinates[3]
    roi = img[r_start:r_start+v_span,c_start:c_start+h_span]

    return roi