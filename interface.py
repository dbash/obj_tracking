import numpy as np
import cv2
from PIL import ImageTk, Image
import tkinter
import glob, os
import matplotlib.pyplot as plt
import detect, track

class Application():
    def __init__(self, root):
        self.root = root
        self.IMG_FOLDER = '/scratch2/dinka/from_tuna/scratch/cs585/HW4/data/Normalized/v1/'
        self.img_list = sorted(glob.glob(self.IMG_FOLDER + '*.jpg'))
        self.cur_idx = 0
        self.flow = None
        self.multitrack = None
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
        self.root.bind('f', self.F_key)  # flow
        self.root.bind('t', self.T_key) # thresholding
        self.root.bind('m', self.Multitracking_key)  # multitracking
        self.root.bind('d', self.next_flow_key) #flow next
        self.root.bind('a', self.prev_flow_key) # flow prev
        self.root.bind(1, self.show_video_1)
        self.root.bind(2, self.show_video_2)
        self.thresholded = False
        self.negative = False
        self.show_flow = False
        self.show_multitrack = False

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

    def next_flow_key(self, event):
        if self.show_flow:
            if self.cur_idx < self.num_images - 1:
                self.cur_idx += 1
                self.show_flow_img()
        elif self.show_multitrack:
            if self.cur_idx < self.num_images - 1:
                self.cur_idx += 1
                self.show_multitrack_img()

    def prev_flow_key(self, event):
        if self.show_flow:
            if self.cur_idx > 0:
                self.cur_idx -= 1
                self.show_flow_img()
        elif self.show_multitrack:
            if self.cur_idx > 0:
                self.cur_idx -= 1
                self.show_multitrack_img()


    def show_image(self):
        self.cv_img = cv2.cvtColor(cv2.imread(self.img_list[self.cur_idx]), cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def show_flow_img(self):
        self.cv_img = cv2.cvtColor(self.flow[..., self.cur_idx], cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def show_multitrack_img(self):
        self.cv_img = cv2.cvtColor(self.multitrack[self.cur_idx], cv2.COLOR_BGR2RGB)
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


    def F_key(self, event):
        if self.show_flow:
            self.show_image()
            self.flow = None
        else:
            self.flow = track.flow(self.img_list)
            self.show_flow_img()
        self.show_flow = not self.show_flow


    def Multitracking_key(self, event):
        if self.show_multitrack:
            self.show_image()
            self.flow = None
        else:
            self.multitrack = track.simple_multitracker_greedy(self.img_list)
            self.show_multitrack_img()
        self.show_multitrack = not self.show_multitrack



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
