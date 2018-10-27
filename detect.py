import numpy as np
import cv2

import matplotlib.pyplot as plt

def flow(img_list):
    n_frames = len(img_list)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.9,
                          minDistance=100,
                          blockSize=5)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(64, 64),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    old_frame = cv2.imread(img_list[0])
    old_gray = segment_morph(old_frame, False)


    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    cur_frame = 1
    out_img_flow = np.zeros(old_frame.shape + (n_frames,), dtype='uint8')
    out_img_flow[...,  0] = old_frame

    while(cur_frame < n_frames):
        frame = cv2.imread(img_list[cur_frame])
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_grey = segment_morph(frame, False)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_grey, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        out_img_flow[..., cur_frame] = img

        # Now update the previous frame and previous points
        old_gray = frame_grey.copy()
        p0 = good_new.reshape(-1, 1, 2)
        cur_frame = cur_frame + 1
    return out_img_flow


# #performs thresholding and closing
def segment_morph(img, negative):
    norm_img = (255*((img - img.min())/img.ptp())).astype('uint16')
    trd = (255 - int(norm_img.mean() + norm_img.var()/4), 255) if negative\
        else (int(norm_img.mean() + norm_img.var()/4), 255) #decide the thresholding boundaries

    trd_mode = cv2.THRESH_BINARY_INV if negative else cv2.THRESH_BINARY

    img_bw = np.array(np.sum(norm_img, axis=2) / 3, dtype='uint8') #rgb to bw
    img_thresholded = cv2.threshold(img_bw, trd[0], trd[1], trd_mode)[1]
    img_morph = cv2.erode(img_thresholded, np.ones((5,5)) , 3)
    for _ in range(3): #need to do dilation more times to connect the cell parts
        img_morph = cv2.dilate(img_morph, np.ones((8,8)), 5)
    img_morph = cv2.erode(img_morph, np.ones((8,8)), 1)
    return img_morph


# computes centroids of all objects
#returns the image with an additional label class for centroid (max(img))
# and the coordinated of centroids
def mark_object_centroids(img):
    classes = np.unique(img) #all object labels
    out_img = np.copy(img)
    centroids = {}
    mark_size = 4
    centroid_class = max(classes) + 1 #we will mark the centroids with this label

    for class_id in classes[1:]:
        positions = np.where(img==class_id)
        centroid = [int(positions[0].mean()), int(positions[1].mean())] #avg coordinage of object pixels
        centroids.update({class_id:centroid})
        x_start, x_end = max(0, centroid[0] - mark_size), \
                         min(img.shape[0] - 1, centroid[0] + mark_size)
        y_start, y_end = max(0, centroid[1] - mark_size), \
                         min(img.shape[1] - 1, centroid[1] + mark_size)
        out_img[x_start:x_end, y_start:y_end] = centroid_class
    return out_img, centroids


#performs thresholding by lowest values (cell body) - does not work well
# def segment_morph(img, negative):
#     norm_img = 255 - (255*((img - img.min())/img.ptp())).astype('uint16')
#     trd = (255 - int(norm_img.mean() + norm_img.var()/4), 255) if negative\
#         else (int(norm_img.mean() + norm_img.var()/4), 255) #decide the thresholding boundaries
#
#     trd_mode = cv2.THRESH_BINARY_INV if negative else cv2.THRESH_BINARY
#
#     img_bw = np.array(np.sum(norm_img, axis=2) / 3, dtype='uint8') #rgb to bw
#     img_thresholded = cv2.threshold(img_bw, trd[0], trd[1], trd_mode)[1]
#     img_morph = cv2.erode(img_thresholded, np.ones((5,5)) , 1)
#     for _ in range(3): #need to do dilation more times to connect the cell parts
#         img_morph = cv2.dilate(img_morph, np.ones((8,8)), 5)
#     # img_morph = cv2.erode(img_thresholded, kernel, 1)
#     return img_morph


def label_img(img):
    _, labels = cv2.connectedComponents(img)
    filtered_labels = filter_small_objects(labels)
    filtered_labels_w_centroids, _ = mark_object_centroids(filtered_labels)
    # Map component labels to hue val
    label_hue = np.uint8(179 * filtered_labels_w_centroids / np.max(filtered_labels_w_centroids))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img


#removes small objects from the label image
def filter_small_objects(img):
    res_img = np.copy(img)
    num_pixels = img.shape[0]*img.shape[1]
    min_num_pixels = num_pixels*0.002 #minimal number of pixels to be keep the object

    num_objects = np.max(img) + 1 #total number of detected objects
    for i in range(num_objects):
        num_obj_pixels = np.sum(img==i)
        if num_obj_pixels<min_num_pixels: #remove the object if it is too small
            res_img[img==i] = 0

    #replace the labels so that they are from 0 to n
    classes = np.unique(res_img)
    num_objects = classes.shape[0]
    for i in range(num_objects):
        res_img[res_img == classes[i]] = i

    return res_img




def detect(img, negative=True):
    segmented_img = segment_morph(img, negative)
    labeled_img = label_img(segmented_img)

    return labeled_img
