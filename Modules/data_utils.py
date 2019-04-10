import csv
import os
from typing import Any, Union

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import imageproc_utils as ipu
import plotting_utils as pu
from texture import texture_analysis as ta


class ImgDatum:
    def __init__(self, img, filename):
        self.img = img
        self.raw_img = img
        self.filename = filename

        self.ff_img = None
        self.ff_filename = None
        self.ff_corr_img = None
        self.ff_corrected = False

    def get_filename(self):
        return self.filename

    def get_img(self):
        return self.img

    def get_ff_corr_img(self):
        return self.ff_corr_img

    def axes_creator(self, fig, ff_corrected=True):
        ax = plt.axes()
        fig.add_axes(ax)
        # ax = fig.add_subplot(111)
        # ax.set_axis_off()
        ax.set_title(self.filename)

        if ff_corrected:
            try:
                im = ax.imshow(self.ff_corr_img, vmin=np.percentile(self.ff_corr_img, 1), vmax=np.percentile(
                    self.ff_corr_img, 99))
            except TypeError:
                im = ax.imshow(self.img)
        else:
            im = ax.imshow(self.img, vmin=np.percentile(self.img, 1), vmax=np.percentile(self.img, 99))
        fig.colorbar(im)
        # replace this with a more robust plotting routine in the future that will allow me
        #  to window the image (like my threshHist class)

    def ff_correct(self, ff_datum, mode = 'manual'):

        self.ff_img = ff_datum.img
        self.ff_corr_img = ipu.ff_correct(self, ff_datum,mode = mode)
        self.ff_filename = ff_datum.filename
        self.ff_corrected = True

        self.img = self.ff_corr_img

    def quantize_img(self, range_mode='auto'):
        self.quant_img = ipu.quantize_img(self, range_mode=range_mode)


class DataList:
    def __init__(self, data):
        self.data = data
        self.slides = None
        self.roi_list = []

    def __getitem__(self, ind):
        if isinstance(ind, slice):
            list = self.data[ind.start:ind.stop]
            new_data = DataList(list)
            return new_data
        else:
            return self.data[ind]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self):
            curr_datum = self[self.i]
            self.i += 1
            return curr_datum
        else:
            raise StopIteration

    def append_datum(self, img_datum, position=None):
        if position == None:
            self.data.append(img_datum)
        else:
            self.data.insert(img_datum, position)

    def analyze_texture(self):
        try:
            self.texture_data = [ta.TextureDatum(rois) for rois in self.roi_list]
        except TypeError:
            raise TypeError

    def select_rois(self):
        self.slides = pu.ImageSlideshow(self)
        if not all(roi == [] for roi in self.slides.roi_list):
            self.roi_list = self.slides.roi_list

    def show_quant_imgs(self):
        # Show all images in a slideshow type format
        self.slides = pu.ImageSlideshow(self)

    def show_imgs(self, img_type='img'):
        # Show all images in a slideshow type format
        self.slides = pu.ImageSlideshow(self,img_type = img_type)

    def get_threshold_list(self):
        threshold_list = []
        for spectral_datum in self:
            threshold_list.append(spectral_datum.threshold)
        return threshold_list

    def get_img_list(self):  # won't work for frame list
        imgs = []
        for datum in self:
            imgs.append(datum.img)
        return imgs

    def get_raw_img_list(self):
        raw_imgs = []
        for datum in self:
            raw_imgs.append(datum.raw_img)
        return raw_imgs

    def get_ff_corr_img_list(self):
        ff_corr_imgs = []
        for datum in self:
            ff_corr_imgs.append(datum.ff_corr_img)
        return ff_corr_imgs

    def get_filename_list(self):
        filenames = []
        for datum in self.data:
            filenames.append(datum.filename)
        return filenames

    def ff_corr(self, ff_datalist, img_ff_mapping=None , mode = 'manual'):
        # img_ff_mapping is a mapping of which data corresponds to which flatfield
        if img_ff_mapping == None:
            if len(ff_datalist) == len(self):
                for i, datum in enumerate(self):
                    datum.ff_correct(ff_datalist[i],mode = mode)
            else:
                raise ValueError(
                    'Too few or too many flatfields provided. Please provide one for each datum or provide a image-flat-field mapping.')
        else:
            for i, datum in enumerate(self):  # i-th data corresponds to img_ff_mapping[i]-th flatfield
                datum.ff_correct(ff_datalist[img_ff_mapping[i]], mode = mode)

    def quantize_imgs(self,range_mode = 'auto'):
        self.quant_imgs = [img_datum.quantize_img(range_mode = range_mode) for img_datum in self]


class SpectralImgDatum(ImgDatum):  # need to modify ImgDataLoader to make it data type agnostic
    def __init__(self, img, filename, threshold, frame=-1):  # or should frame = -1?
        super().__init__(img, filename)
        self.threshold = threshold
        self.frame = frame  # -1 will indicate that it is integral data


class SpectralFrameList:  # this is a list of spectral
    def __init__(self, frames_list, threshold):
        self.frames_list = frames_list  # a list of SpectralImgDatum objects with frames specified for each. THe
        # importer will determine the frames when parsing the data.
        self.threshold = threshold
        self.ff_corr_list = None

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, ind):
        return self.frames_list[ind]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self):
            curr_frame = self[self.i]
            self.i += 1
            return curr_frame
        else:
            raise StopIteration

    # def axes_creator(self, fig):
    #     axes = []
    #     for frame in self:

    def append_frame(self, frame_datum):
        self.frames_list.append(frame_datum)

    def convert_to_integral(self, frame_range=None):

        if frame_range is None:
            start = 0
            stop = len(self)
            frame_slice = slice(start, stop)  # this might not work....Can my list be indexed using slices?? It
            # should work
        else:
            start = frame_range[0] - 1
            stop = frame_range[1]  # will include the stop number in this case
            frame_slice = slice(start, stop)

        #
        integral_dir = 'summed_frames' + str(start + 1) + '_' + str(stop)
        #
        # try:
        #     os.chdir(integral_dir)
        # except FileNotFoundError:
        #     os.mkdir(integral_dir)
        # make a new folder to store the integral data, make all of the integral data, write to files,
        # create spectralintegralloader object for it

        frame_subset = self[frame_slice]
        integral_filename = (integral_dir + '_th-' + str(self.threshold))
        frame_sum = np.zeros(frame_subset[0].shape)
        for frame in self:
            frame_sum += frame

        spectral_integral_datum = SpectralImgDatum(frame_sum, integral_filename, self.threshold, frame=-1)
        return spectral_integral_datum

        # create a spectralimgdatum from this

    def get_img(self):
        return self.frames_list

    def ff_correct(self, ff_frames_list):

        for i, frame in enumerate(self):
            frame.ff_correct(ff_frames_list[i].img)


#######################Import Functions#############################
def get_file_delimiter(file):
    """determine delimiter for csv/txt files"""
    with open(file, 'r') as data:
        dialect = csv.Sniffer().sniff(data.read())

    return dialect.delimiter


def get_threshold_from_filename(filename):
    # filename_parts = filename.split('_')
    # Ivan's Data
    filename_parts = filename.split('-')

    try:
        threshold = int(filename_parts[-2])
        print('Warning: ' + filename + ' appears to be frame data. Use SpectralFrameLoader instead.')
    except ValueError:
        try:
            threshold = int(filename_parts[-1])
        except ValueError:
            threshold = -1
            print('Error: Cannot parse threshold from ' + filename)

    return threshold


def get_frame_from_filename(filename):
    filename_parts = filename.split('_')

    try:
        # a filename like "test_1_020.txt" could masquerade as framed data in this implementation.
        threshold = int(filename_parts[-2])
        frame = int(filename_parts[-1])
    except ValueError:
        frame = -1
        print('Error: ' + filename + ' appears to be integral data. Use SpectralIntegralLoader instead.')
    return frame


def open_img(filename):
    file, file_ext = os.path.splitext(filename)
    # add data to img list

    if file_ext in ['.txt', '.csv', '.pmf']:
        img = np.loadtxt(filename, delimiter=get_file_delimiter(filename))

    elif file_ext == '.img':
        img = np.fromfile(filename, dtype='f')
        img = np.transpose(img.reshape(240, 760))

    elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
        img = cv.imread(filename, -1)  # returns color images in reverse order (B,G, then R last)

    else:
        print(file + ' is not in a valid format. File will be skipped.')
        img = None

    return img


def import_data(directory=os.getcwd(), data_type='integral'):
    data = []
    file_list = os.listdir(directory)
    file_list.sort()

    for file in file_list:
        filename, file_ext = os.path.splitext(file)
        # add data to img list
        file_path = directory + '/' + file

        if file_ext in ['.txt', '.csv', '.pmf']:
            img = np.loadtxt(file_path, delimiter=get_file_delimiter(file_path))

        elif file_ext == '.img':
            img = np.fromfile(file_path, dtype='f')
            img = np.transpose(img.reshape(240, 760))

        elif file_ext == '.hdf':
            print(file + ' is in hdf5 format. File will be skipped.')
            continue

        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
            img = cv.imread(file_path, -1)  # returns color images in reverse order (B,G, then R last)

        else:
            print(file + ' is not in a valid format. File will be skipped.')
            continue

        if data_type == 'integral':
            datum = ImgDatum(img, filename)
        elif data_type == 'spectral':
            threshold = get_threshold_from_filename(filename)
            datum = SpectralImgDatum(img, filename, threshold)
        elif data_type == 'frames':  # UH OH this doesn't work because it never creates the SpectralFrameList
            threshold = get_threshold_from_filename(filename)
            frame = get_frame_from_filename(filename)
            datum = SpectralImgDatum(img, filename, threshold, frame)
            if len(data) != 0:
                for frame_list in data:
                    if frame_list.threshold == threshold:
                        frame_list.append_frame(datum)
                    else:
                        datum = SpectralFrameList([datum], threshold)
            else:
                datum = SpectralFrameList([datum], threshold)

        else:
            raise NameError('Type argument must be: integral, spectral, or frames')

        data.append(datum)

    datalist = DataList(data)

    return datalist


# class DataLoader_old:
#     def __init__(self, dir=os.getcwd()):
#         # if the number of images is less than 20 (i.e. small), store as a list of images
#         self.dir = dir
#         self.data = []
#
#         self.import_data()
#
#     def __getitem__(self, ind):
#         return self.data[ind]
#
#     def __str__(self):
#         return 'Directory: ' + str(self.dir) + '\nNumber of Images: ' + str(len(self))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __iter__(self):
#         self.i = 0
#         return self
#
#     def __next__(self):
#         if self.i < len(self):
#             curr_datum = self[self.i]
#             self.i += 1
#             return curr_datum
#         else:
#             raise StopIteration
#
#     # def __add__(self, new_data):
#     #     self.data += new_data
#     #     return
#
#     def create_datum_object(self, img, filename):
#         datum = ImgDatum(img, filename)
#         return datum
#
#     def import_data(self):
#
#         file_list = os.listdir(self.dir)
#
#         for file in file_list:
#             filename, file_ext = os.path.splitext(file)
#             # add data to img list
#             file_path = self.dir + '/' + file
#
#             if file_ext in ['.txt', '.csv']:
#                 img = np.loadtxt(file_path, delimiter=self.get_file_delimiter(file_path))
#
#             elif file_ext == '.hdf':
#                 print(file + ' is in hdf5 format. File will be skipped.')
#                 continue
#
#             elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
#                 img = cv.imread(file_path, -1)  # returns color images in reverse order (B,G, then R last)
#
#             else:
#                 print(file + ' is not in a valid format. File will be skipped.')
#                 continue
#
#             datum = self.create_datum_object(img, filename)
#             self.append_datum(datum)
#
#     def get_file_delimiter(self, file):
#         """determine delimiter for csv/txt files"""
#         with open(file, 'r') as data:
#             dialect = csv.Sniffer().sniff(data.read())
#
#         return dialect.delimiter
#
#     def show_imgs(self):
#         # Show all images in a slideshow type format
#         axes_creators = self.get_axes_creators()
#         slideshow = ImageSlideshow(axes_creators)
#         return slideshow
#
#     def append_datum(self, img_datum):
#         self.data.append(img_datum)
#
#     def get_axes_creators(self):
#         axes_creators = []
#
#         for img_datum in self:
#             axes_creators.append(img_datum.axes_creator)
#
#         return axes_creators
#
#     def get_img_list(self):
#         imgs = []
#         for datum in self:
#             imgs.append(datum.img)
#         return imgs
#
#     def get_ff_corr_img_list(self):
#         ff_corr_imgs = []
#         for datum in self:
#             ff_corr_imgs.append(datum.ff_corr_img)
#         return ff_corr_imgs
#
#     def get_filename_list(self):
#         filenames = []
#         for datum in self.data:
#             filenames.append(datum.filename)
#         return filenames


#
#
# class SpectralDataLoader:
#     def __init__(self, dir = os.getcwd(), type = 'integral'):  # img_type can be: summed, frame, binned
#         self.dir = dir
#         self.data = {}
#         self.type = type
#         self.import_data(self.type)
#
#     def __getitem__(self, key):
#         return self.data[key]
#
#     def __str__(self):
#         return 'Directory: ' + str(self.dir) + '\nNumber of Images: ' + str(len(self))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __iter__(self):
#         self.i = 0
#         return self
#
#     def __next__(self):
#         if self.i < len(self):
#             curr_datum = self[self.i]
#             self.i += 1
#             return curr_datum
#         else:
#             raise StopIteration
#
#     def create_datum_object(self, img, filename):
#         datum = ImgDatum(img, filename)
#         return datum
#
#     def _get_file_delimiter(self, file):
#         """determine delimiter for csv/txt files"""
#         with open(file, 'r') as data:
#             dialect = csv.Sniffer().sniff(data.read())
#
#         return dialect.delimiter
#
#     def import_data(self, type):
#
#         file_list = os.listdir(self.dir)
#
#         for file in file_list:
#             filename, file_ext = os.path.splitext(file)
#             # add data to img list
#             file_path = self.dir + '/' + file
#
#             if file_ext in ['.txt', '.csv']:
#                 img = np.loadtxt(file_path, delimiter = self._get_file_delimiter(file_path))
#
#             elif file_ext == '.hdf':
#                 print(file + ' is in hdf5 format. File will be skipped.')
#                 continue
#
#             elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
#                 img = cv.imread(file_path, -1)  # returns color images in reverse order (B,G, then R last)
#
#             else:
#                 print(file + ' is not in a valid format. File will be skipped.')
#                 continue
#             if self.type == 'frame':
#                 frame = self.get_frame_from_filename(filename)
#                 datum = ImgDatum(img, filename, frame)
#             else:
#                 datum = ImgDatum(img, filename)
#             self.append_datum(datum)
#
#     def get_threshold_from_filename(self, filename):
#         # filename_parts = filename.split('_')
#         # Ivan's Data
#         filename_parts = filename.split('-')
#
#         try:
#             threshold = int(filename_parts[-2])
#             print('Warning: ' + filename + ' appears to be frame data. Use SpectralFrameLoader instead.')
#         except ValueError:
#             try:
#                 threshold = int(filename_parts[-1])
#             except ValueError:
#                 threshold = -1
#                 print('Error: Cannot parse threshold from ' + filename)
#
#         return threshold
#
#     def get_threshold_list(self):
#         thresholds = list(self.data.keys())
#         return thresholds
#
#     def append_datum(self, img_datum):
#         threshold = self.get_threshold_from_filename(img_datum.filename)
#         if threshold in self.data:
#             self.data[threshold].append(img_datum)
#         else:
#             self.data[threshold] = [img_datum]
#
#     def show_imgs(self):
#         # Show all images in a slideshow type format
#         axes_creators = self.get_axes_creators()
#         slideshow = ImageSlideshow(axes_creators)
#         return slideshow
#
#     def get_axes_creators(self):
#         axes_creators = []
#
#         for key in self.data:
#             for datum in self.data[key]:
#                 axes_creators.append(datum.axes_creator)
#
#         return axes_creators
#
#     def get_ff_corr_img_list(self):
#         ff_corr_imgs = []
#
#         for key in self.data:
#             for datum in self.data[key]:
#                 ff_corr_imgs.append(datum.ff_corr_img)
#
#         return ff_corr_imgs
#
#     def ff_corr_data(self, ff_data):
#         for key in self.data:
#             for i, datum in enumerate(self.data[key]):
#                 datum.ff_correct(ff_data[key][i].img)
#
#     def get_frame_from_filename(self, filename):
#         filename_parts = filename.split('_')
#
#         try:
#             # a filename like "test_1_020.txt" could masquerade as framed data in this implementation.
#             threshold = int(filename_parts[-2])
#             frame = int(filename_parts[-1])
#         except ValueError:
#             frame = -1
#             print('Error: ' + filename + ' appears to be integral data. Use SpectralIntegralLoader instead.')
#         return frame


# class SpectralFrameLoader(SpectralIntegralLoader):
#     def __init__(self, dir = os.getcwd()):
#         super().__init__(dir)
#
#     def import_data(self):
#
#         file_list = os.listdir(self.dir)
#
#         for file in file_list:
#             filename, file_ext = os.path.splitext(file)
#             # add data to img list
#             file_path = self.dir + '/' + file
#
#             if file_ext in ['.txt', '.csv']:
#                 img = np.loadtxt(file_path, delimiter = self._get_file_delimiter(file_path))
#
#             elif file_ext == '.hdf':
#                 print(file + ' is in hdf5 format. File will be skipped.')
#                 continue
#
#             elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
#                 img = cv.imread(file_path, -1)  # returns color images in reverse order (B,G, then R last)
#
#             else:
#                 print(file + ' is not in a valid format. File will be skipped.')
#                 continue
#
#             datum = self.create_datum_object(img, filename)
#             self.append_datum(datum)
#
#     def create_datum_object(self, img, filename):
#         threshold = self.get_threshold_from_filename(filename)
#         frame = self.get_frame_from_filename(filename)
#         frame_datum = SpectralImgDatum(img, filename, threshold, frame)
#         return frame_datum
#
#     def get_frame_from_filename(self, filename):
#         filename_parts = filename.split('_')
#
#         try:
#             # a filename like "test_1_020.txt" could masquerade as framed data in this implementation.
#             threshold = int(filename_parts[-2])
#             frame = int(filename_parts[-1])
#         except ValueError:
#             frame = -1
#             print('Error: ' + filename + ' appears to be integral data. Use SpectralIntegralLoader instead.')
#         return frame
#
#     def get_frame_list(self):
#         frames = []
#         for datum in self:
#             frames.append(datum.frame)
#         return frames


class Experiment:
    def __init__(self, r1, r2, detector, kvp, current, spot_size):
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.detector = detector
        self.kvp = kvp
        self.current = current
        self.spot_size = spot_size
        self.magnification = (self.r1 + self.r2) / self.r1
        self.data = None
        self.threshold_energy_maps = None

        self.get_k_vector()

    def include_data(self, data):
        self.data = data

    def add_data(self, data):
        self.data = data

    def get_k_vector(self):
        k_max = 2 * np.pi / self.detector.pixel_size
        k_nyq = k_max / 2.
        dkx = k_max / self.detector.resolution[0]
        dky = k_max / self.detector.resolution[1]

        k_x = np.linspace(-k_nyq + dkx, k_nyq, self.detector.resolution[0], endpoint=True)  # /self.magnification
        k_y = np.linspace(-k_nyq + dky, k_nyq, self.detector.resolution[1], endpoint=True)  # /self.magnification
        self.k_space = [k_x, k_y]
        self.k_2 = np.add.outer(k_x ** 2, k_y ** 2)


class PhaseContrastExperiment(Experiment):
    def __init__(self, r1, r2, detector, kvp, current, spot_size):
        super().__init__(r1, r2, detector, kvp, current, spot_size)


class DualEnergyExperiment(Experiment):
    def __init__(self, r1, r2, detector, kvp, current, spot_size, thresholds):
        super().__init__(r1, r2, detector, kvp, current, spot_size)
        self.thresholds = thresholds


class Detector:
    def __init__(self, name, pixel_size, resolution, bias=-500):
        self.name = name
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.bias = bias

    def get_fov(self):
        fov = self.pixel_size * self.resolution
        return fov


class PhotonCountingDetector(Detector):
    def __init__(self, name, pixel_size, resolution, mode, bias=-500):
        super().__init__(name, pixel_size, resolution, bias)
        self.mode = mode  # single pixel or csm
