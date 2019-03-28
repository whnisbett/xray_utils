from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.widgets as widgets


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
    format = PlottingFormat()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title(title, **format.title_font)

    ax.set_axis_off()
    im = ax.imshow(img, vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))

    cbar = fig.colorbar(im)
    cbar.ax.tick_params(**format.colorbar_ticks)  #

    return fig


def plot_data_2d(xdata, ydata, title=None, xlabel=None, ylabel=None):
    format = PlottingFormat
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(xdata, ydata, **format.plot_2d_params)

    ax.set_title(title, **format.title_font)
    ax.set_xlabel(xlabel, labelpad=format.axes_labelpad, **format.axes_label_font)
    ax.set_ylabel(ylabel, labelpad=format.axes_labelpad, **format.axes_label_font)
    ax.tick_params(axis='x', **format.x_axes_ticks)
    ax.tick_params(axis='y', **format.y_axes_ticks)
    return fig


def line_profile():
    """take in a single image or a dataloader object and give line profile(s) for all images. Maybe have some
    functionality where it's an image slideshow for the dataloader, each slide has the line profile on the right as
    well as the raw image on left so that you can where the line profile is being taken and you can adjust this as
    well"""
    return


class oldImageSlideshow:

    def __init__(self, axes_creators):
        # pass a list of functions into this constructor, each of these functions will take an input figure and
        # manipulate it to plot something specific. What will be specifically plotted is determined by the object
        # that the function is created by.
        plt.ion()
        self.curr_img = 0
        self.axes_creators = axes_creators
        self.fig = plt.figure()
        self.axes_creators[0](self.fig)
        plt.show()
        self.start_listening_press()

        self.curr_roi = np.array([])

    def get_current_img_axes(self):
        return
    def append_axes(self, axes_creator):
        # add figure to the end of the rotation
        self.axes_creators.append(axes_creator)
        self.update_axes()

    def remove_axes(self, ind):
        # clear the canvas, remove the item, redraw
        try:
            if ind < len(self.axes_creators):
                self.axes_creators = self.axes_creators[:ind] + self.axes_creators[(ind + 1):]
                self.curr_img = self.curr_img % len(self.axes_creators)
                self.update_axes()
            else:
                print('Index out of range, please select a number between 0 and ' + str(len(self.axes_creators) - 1))
        except:
            print('Error: Cannot remove since only 1 image remains')

    def goto_axes(self, ind):
        # go to specific figure in the rotation
        if ind < len(self.axes_creators):
            self.curr_img = ind
            self.update_axes()
        else:
            print('Index out of range, please select a number between 0 and ' + str(len(self.axes_creators) - 1))

    def update_axes(self):
        self.fig.clear()
        self.axes_creators[self.curr_img](self.fig)
        plt.draw()

    def goto_next_axes(self):

        self.curr_img = (self.curr_img + 1) % len(self.axes_creators)
        self.update_axes()

    def goto_previous_axes(self):
        self.curr_img = (self.curr_img - 1) % len(self.axes_creators)
        self.update_axes()

    def start_listening_press(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def on_press(self, event):
        if event.key == 'left':
            self.goto_previous_axes()
        elif event.key == 'right':
            self.goto_next_axes()



class ImageSlideshow:
    def __init__(self,image_datalist):
        self.img_list = image_datalist
        self.curr_img = 0
        self.curr_ax = None

        plt.ion()
        self.fig = plt.figure()
        self.plot_curr_img()
        plt.show()
        self.start_listening_press()

        self.curr_roi = np.array([])

    def plot_curr_img(self):
        curr_imgdatum = self.img_list[self.curr_img]

        self.fig.clear()
        ax = plt.axes()
        self.fig.add_axes(ax)
        # ax = fig.add_subplot(111)
        # ax.set_axis_off()
        ax.set_title(curr_imgdatum.filename)

        if curr_imgdatum.ff_corrected:
            try:
                im = ax.imshow(curr_imgdatum.ff_corr_img, vmin=np.percentile(curr_imgdatum.ff_corr_img, 1), vmax=np.percentile(
                    curr_imgdatum.ff_corr_img, 99))
            except TypeError:
                im = ax.imshow(curr_imgdatum.img)
        else:
            im = ax.imshow(curr_imgdatum.img, vmin=np.percentile(curr_imgdatum.img, 1), vmax=np.percentile(curr_imgdatum.img, 99))
        self.fig.colorbar(im)
        plt.draw()
        self.curr_ax = ax
    def roi_selector(self):

        rs = widgets.RectangleSelector(self.curr_ax, self.onclick, drawtype='box',
                                       rectprops=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True),
                                       interactive=True)

        return rs

    def onclick(self,click,release):
        x0 = int(click.xdata)
        x1 = int(release.xdata)
        y0 = int(click.ydata)
        y1 = int(release.ydata)

        curr_imgdatum = self.img_list[self.curr_img]

        if curr_imgdatum.ff_corrected:
            self.curr_roi = curr_imgdatum.ff_corr_img[y0:y1,x0:x1]
        else:
            self.curr_roi = curr_imgdatum.img[y0:y1,x0:x1]

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


def get_img_from_axes(axes):

    children = axes.get_children()
    img_axes = [img_ax for img_ax in children if isinstance(img_ax,mpl.image.AxesImage)]
    if len(img_axes) == 1:
        return img_axes[0].get_array()
    else:
        raise Exception('More than one AxesImage in the axes, unsure which to retrieve.')
