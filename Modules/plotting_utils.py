import matplotlib.widgets as widgets
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button


class PlottingFormat:
    small_size = 10
    medium_size = 15
    large_size = 18
    axes_labelpad = 10

    title_font = {'fontname': 'Arial', 'fontweight': 'normal', 'fontsize': large_size, 'color': 'black',
                  'horizontalalignment': 'center'}
    axes_label_font = {'fontname': 'Arial', 'fontweight': 'normal', 'fontsize': medium_size, 'color': 'black',
                       }  # https://matplotlib.org/api/text_api.html#matplotlib.text.Text for all options
    x_axes_ticks = {'labelsize': medium_size, 'colors': 'red', 'width': 2,
                    'direction': 'inout', 'pad': 1}  #
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    # for options
    y_axes_ticks = {'labelsize': medium_size, 'colors': 'red', 'width': 2,
                    'direction': 'inout', 'pad': 1}
    colorbar_ticks = {'labelsize': small_size, 'colors': 'black', 'width': 1,
                      'direction': 'out'}
    legend = {}
    plot_2d_params = {'marker': '^', 'markersize': 10, 'markerfacecolor': 'black', 'markerfacecoloralt': 'red',
                      'linestyle': '--', 'linewidth': 3, 'color': 'black'}
    # https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_color

    # need to also format the figure size here
    # add legend formatting too


def display_img(img, title=None):
    pformat = PlottingFormat()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title(title, **pformat.title_font)

    ax.set_axis_off()
    im = ax.imshow(img, vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))

    cbar = fig.colorbar(im)
    cbar.ax.tick_params(**pformat.colorbar_ticks)  #

    return fig


def plot_data_2d(xdata, ydata, title=None, xlabel=None, ylabel=None):
    pformat = PlottingFormat
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(xdata, ydata, **pformat.plot_2d_params)

    ax.set_title(title, **pformat.title_font)
    ax.set_xlabel(xlabel, labelpad=pformat.axes_labelpad, **pformat.axes_label_font)
    ax.set_ylabel(ylabel, labelpad=pformat.axes_labelpad, **pformat.axes_label_font)
    ax.tick_params(axis='x', **pformat.x_axes_ticks)
    ax.tick_params(axis='y', **pformat.y_axes_ticks)
    return fig


def line_profile():
    """take in a single image or a dataloader object and give line profile(s) for all images. Maybe have some
    functionality where it's an image slideshow for the dataloader, each slide has the line profile on the right as
    well as the raw image on left so that you can where the line profile is being taken and you can adjust this as
    well"""
    return


# class OldImageSlideshow:
#
#     def __init__(self, axes_creators):
#         # pass a list of functions into this constructor, each of these functions will take an input figure and
#         # manipulate it to plot something specific. What will be specifically plotted is determined by the object
#         # that the function is created by.
#         plt.ion()
#         self.curr_img = 0
#         self.axes_creators = axes_creators
#         self.fig = plt.figure()
#         self.axes_creators[0](self.fig)
#         plt.show()
#         self.start_listening_press()
#
#         self.curr_roi = np.array([])
#
#     def get_current_img_axes(self):
#         return
#
#     def append_axes(self, axes_creator):
#         # add figure to the end of the rotation
#         self.axes_creators.append(axes_creator)
#         self.update_axes()
#
#     def remove_axes(self, ind):
#         # clear the canvas, remove the item, redraw
#         try:
#             if ind < len(self.axes_creators):
#                 self.axes_creators = self.axes_creators[:ind] + self.axes_creators[(ind + 1):]
#                 self.curr_img = self.curr_img % len(self.axes_creators)
#                 self.update_axes()
#             else:
#                 print('Index out of range, please select a number between 0 and ' + str(len(self.axes_creators) - 1))
#         except:
#             raise Exception('Cannot remove since only 1 image remains')
#
#     def goto_axes(self, ind):
#         # go to specific figure in the rotation
#         if ind < len(self.axes_creators):
#             self.curr_img = ind
#             self.update_axes()
#         else:
#             print('Index out of range, please select a number between 0 and ' + str(len(self.axes_creators) - 1))
#
#     def update_axes(self):
#         self.fig.clear()
#         self.axes_creators[self.curr_img](self.fig)
#         plt.draw()
#
#     def goto_next_axes(self):
#
#         self.curr_img = (self.curr_img + 1) % len(self.axes_creators)
#         self.update_axes()
#
#     def goto_previous_axes(self):
#         self.curr_img = (self.curr_img - 1) % len(self.axes_creators)
#         self.update_axes()
#
#     def start_listening_press(self):
#         self.fig.canvas.mpl_connect('key_press_event', self.on_press)
#
#     def on_press(self, event):
#         if event.key == 'left':
#             self.goto_previous_axes()
#         elif event.key == 'right':
#             self.goto_next_axes()


class ImageSlideshow:
    def __init__(self, image_datalist,img_type = 'img'):
        self.img_list = image_datalist
        self.img_type = img_type
        self.n_imgs = len(image_datalist)
        self.curr_img = 0
        self.curr_ax = None
        self.curr_roi = np.array([])
        self.roi_list = [[] for _ in range(self.n_imgs)]

        plt.ioff()
        self.fig = plt.figure(figsize=(10, 10))
        self.plot_curr_img()

        self.start_listening_press()
        plt.show()

    def clear_roi_list(self):
        self.roi_list = [[] for _ in range(self.n_imgs)]

    def plot_curr_img(self):
        self.fig.clear()

        curr_imgdatum = self.img_list[self.curr_img]
        self.curr_ax = plt.axes()

        self.curr_ax.set_title(curr_imgdatum.filename)
        # check what image we want to plot here and then do it

        if self.img_type == 'img':
            img = curr_imgdatum.img
        elif self.img_type == 'raw':
            img = curr_imgdatum.raw_img
        elif self.img_type == 'ff':
            img = curr_imgdatum.ff_img
        elif self.img_type == 'corrected':
            img = curr_imgdatum.ff_corr_img
        elif self.img_type == 'quantized':
            img = curr_imgdatum.quant_img
        else:
            raise Exception('Not a valid image type')



        im = self.curr_ax.imshow(img, vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        self.fig.colorbar(im)

        self.make_buttons()
        self.fig.canvas.draw()

    def make_buttons(self):
        self.button_active = False

        selector_button_ax = self.fig.add_axes([0.21, 0.05, 0.1, 0.075])
        self.selector_button = Button(selector_button_ax, 'ROI Selector')
        self.selector_button.on_clicked(self.roi_selector)

        append_button_ax = self.fig.add_axes([0.41, 0.05, 0.1, 0.075])
        self.append_button = Button(append_button_ax, 'Append ROI')
        self.append_button.on_clicked(self.append_roi)

        remove_button_ax = self.fig.add_axes([0.61, 0.05, 0.1, 0.075])
        self.remove_button = Button(remove_button_ax, 'Remove ROI')
        self.remove_button.on_clicked(self.remove_last_roi)

    def roi_selector(self, event):
        if not self.button_active:
            self.selector_button.color = '#ff8080'
            self.button_active = True
            self.rs = widgets.RectangleSelector(self.curr_ax, self.onclick, drawtype='box',
                                                rectprops=dict(facecolor='#ff4d4d', edgecolor='black', alpha=0.3,
                                                               fill=True),
                                                interactive=True)
            self.fig.canvas.draw()
        else:
            self.selector_button.color = '0.85'
            self.button_active = False
            self.rs.set_visible(False)

    def append_roi(self, event):
        self.roi_list[self.curr_img].append(self.curr_roi)

    def remove_last_roi(self, event):
        self.roi_list[self.curr_img] = self.roi_list[self.curr_img][:-1]

    def onclick(self, click, release):
        x0 = int(click.xdata)
        x1 = int(release.xdata)
        y0 = int(click.ydata)
        y1 = int(release.ydata)

        curr_imgdatum = self.img_list[self.curr_img]

        if curr_imgdatum.ff_corrected:
            self.curr_roi = curr_imgdatum.ff_corr_img[y0:y1, x0:x1]
        else:
            self.curr_roi = curr_imgdatum.img[y0:y1, x0:x1]

        self.fig.canvas.draw()

    def goto_next_img(self):

        self.curr_img = (self.curr_img + 1) % len(self.img_list)
        self.plot_curr_img()

    def goto_previous_img(self):

        self.curr_img = (self.curr_img - 1) % len(self.img_list)
        self.plot_curr_img()

    def start_listening_press(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def on_press(self, event):
        if event.key == 'left':
            self.goto_previous_img()
        elif event.key == 'right':
            self.goto_next_img()
