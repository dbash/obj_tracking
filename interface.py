import numpy as np
import cv2
from PIL import ImageTk, Image
import tkinter
import glob, os
import matplotlib.pyplot as plt
import detect

class Application():
    def __init__(self, root):
        self.root = root
        self.IMG_FOLDER = '/scratch2/dinka/from_tuna/scratch/cs585/HW4/data/Normalized/v1/'
        self.img_list = sorted(glob.glob(self.IMG_FOLDER + '*.jpg'))
        self.cur_idx = 0
        self.num_images = len(self.img_list)
        self.root.title = "Cell Tracking Project"
        self.cv_img = cv2.cvtColor(cv2.imread(self.img_list[self.cur_idx]), cv2.COLOR_BGR2RGB)
        self.height, self.width, no_channels = self.cv_img.shape
        self.canvas = tkinter.Canvas(self.root, width=self.width, height=self.height)
        self.kernel = np.ones((6,6), np.uint8)
        self.canvas.pack()
        self.show_image()

        self.root.bind('<Left>', self.left_key)
        self.root.bind('<Right>', self.right_key)
        self.root.bind('n', self.N_key) # negative
        self.root.bind('t', self.T_key) # thresholding
        self.root.bind(1, self.show_video_1)
        self.root.bind(2, self.show_video_2)
        self.thresholded = False
        self.negative = False

        self.root.mainloop()

    def right_key(self, event):
        if self.cur_idx < self.num_images - 1:
            self.cur_idx += 1
            self.show_image()
            self.negative = False
            self.thresholded = False


    def left_key(self, event):
        if self.cur_idx > 0:
            self.cur_idx -= 1
            self.show_image()
            self.negative = False
            self.thresholded = False

    def show_image(self):
        self.cv_img = cv2.cvtColor(cv2.imread(self.img_list[self.cur_idx]), cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def N_key(self, event):
        self.cv_img = 255 - self.cv_img
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        self.negative = not self.negative

    def T_key(self, event):
        if self.thresholded:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.cv_img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.thresholded = False
        else:
            self.labels = detect.detect(self.cv_img, negative=self.negative)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.labels))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.thresholded = True

    def show_video_1(self, event):
        self.IMG_FOLDER = '/scratch2/dinka/from_tuna/scratch/cs585/HW4/data/Normalized/v1/'
        self.img_list = sorted(glob.glob(self.IMG_FOLDER + '*.jpg'))
        self.cur_idx = 0
        self.num_images = len(self.img_list)
        self.show_image()

    def show_video_2(self, event):
        self.IMG_FOLDER = '/scratch2/dinka/from_tuna/scratch/cs585/HW4/data/Normalized/v2/'
        self.img_list = sorted(glob.glob(self.IMG_FOLDER + '*.jpg'))
        self.cur_idx = 0
        self.num_images = len(self.img_list)
        self.show_image()


window = tkinter.Tk()
Application(window)
