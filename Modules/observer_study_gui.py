import csv
import os
import random
from tkinter import Tk, ttk

import numpy as np


class ObserverStudyGUI(Tk):
    def __init__(self, dir):

        super().__init__()

        self.stimuli = self.import_imgs(dir)
        self.keys, self.imgs = self.randomize_stimuli()
        self.n_imgs = len(self.imgs)
        self.curr_img_i = 0
        self.curr_rating = None
        self.curr_coordinates = None
        self.ratings = [None] * self.n_imgs
        self.coordinates = [None] * self.n_imgs

        self.img_frame = ImageFrame(self) # Have a different type of child class inheriting from
        # Frame for each special frame. Maybe combine sliders and images into one frame
        self.button_frame = ButtonFrame(self)

        self.title('Human Observer LROC Study')

        self.rating_1_button = ttk.Radiobutton(self.button_frame, text='1',
                                               variable=self.curr_rating, value=1)
        self.rating_2_button = ttk.Radiobutton(self.button_frame, text='2',
                                               variable=self.curr_rating, value=2)
        self.rating_3_button = ttk.Radiobutton(self.button_frame, text='3',
                                               variable=self.curr_rating, value=3)
        self.rating_4_button = ttk.Radiobutton(self.button_frame, text='4',
                                               variable=self.curr_rating, value=4)

        self.next_img_button = ttk.Button(self.button_frame, text='Confirm',
                                          command=self.on_click_next)
        self.back_img_button = ttk.Button(self.button_frame, text='Previous',
                                          command=self.on_click_previous)

        self.rating_1_button.pack()
        self.rating_2_button.pack()
        self.rating_3_button.pack()
        self.rating_4_button.pack()
        self.next_img_button.pack()

    def randomize_stimuli(self):
        keys = list(self.stimuli.keys())
        imgs = list(self.stimuli.values())
        stimuli = list(zip(keys, imgs))
        random.shuffle(stimuli)
        keys_rand, imgs_rand = zip(*stimuli)
        return keys_rand, imgs_rand

    def reset_states(self):
        self.curr_rating = None
        self.curr_coordinates = None
        # reset slider if there is one
        # reset radio buttons

    def on_click_image(self):
        return

    def on_click_next(self):

        if self.curr_rating in [1, 2]:
            # If user says there are no abnormalities: store rating, increment image index,
            # and reset all state variables
            self.ratings[self.curr_img_i] = self.curr_rating
            self.curr_img_i += 1
            self.reset_states()

        elif self.curr_rating in [3, 4]:
            # If user says there is an abnormality, check to see if they localized.
            if self.curr_coordinates is None:
                # If they didn't, prompt them to select one.
                raise Exception('Please localize the abnormality before proceeding')

            else:
                # Otherwise, store the coordinates and ratings, increment image index, and reset
                # allstate variables
                self.coordinates[self.curr_img_i] = self.curr_coordinates
                self.ratings[self.curr_img_i] = self.curr_rating
                self.curr_img_i += 1
                self.reset_states()
        else:
            # Otherwise (if no rating is selected), prompt user to select a rating
            raise Exception('Please select a confidence rating before proceeding')

    def on_click_previous(self):
        return

    def export_results(self):
        # Export the results as a tab delimited text file and include: filename, rating,
        # coordinates (should be x,y,z for ct and just x,y for slice/projection)

        return

    def import_imgs(self, dir):
        stimuli = {}
        file_list = os.listdir(dir)
        for file in file_list:
            filename, file_ext = os.path.splitext(file)
            # add data to img list
            file_path = dir + '/' + file
            if file_ext == '.img':
                img = np.fromfile(file_path, dtype='f')
                # flips the image to be vertical
                img = np.transpose(img.reshape(240, 760))
            stimuli[filename] = img
        return stimuli

    def get_file_delimiter(self, file):
        """determine delimiter for csv/txt files"""
        with open(file, 'r') as data:
            dialect = csv.Sniffer().sniff(data.read())

        return dialect.delimiter
class ImageFrame(ttk.Frame):
    def __init__(self,parent):
        super().__init__(parent)

class ButtonFrame(ttk.Frame):
    def __init__(self,parent):
        super().__init__(parent)