import numpy as np
import detect
import cv2
import pykalman
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
    old_labels = detect.label_img(detect.segment_morph(old_frame, False), centroids=False)
    old_gray = 255*np.array(old_labels>0, dtype='uint8')

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    cur_frame = 1
    out_img_flow = np.zeros(old_frame.shape + (n_frames,), dtype='uint8')
    out_img_flow[...,  0] = old_frame

    while(cur_frame < n_frames):
        frame = cv2.imread(img_list[cur_frame])
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = 255 * np.array(detect.label_img(
            detect.segment_morph(frame, False), centroids=False) > 0, dtype='uint8')

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if sum(st==1) > 0:
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        else:
            goow_old = good_new
            good_new = []

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        out_img_flow[..., cur_frame] = img

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        cur_frame = cur_frame + 1
    return out_img_flow



def simple_multitracker(img_list):
    n_frames = len(img_list)
    cur_frame = cv2.imread(img_list[0])
    cur_labels = detect.label_img(detect.segment_morph(cur_frame, False), centroids=False)
    trackers = cv2.MultiTracker()
    ret_images = []
    colors = []
    ret_frame = cur_frame.copy()
    for obj_label in np.unique(cur_labels): #initializing trackers for each detected object
        if obj_label == 0:
            continue
        bbox = cv2.boundingRect(np.array(cur_labels == obj_label, dtype='uint8'))
        trackers.add(cv2.TrackerCSRT_create(), cur_frame, bbox)
        cv2.rectangle(ret_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      (0, 255, 0), 2)
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255),
                       np.random.randint(0, 255)))
    ret_images.append(ret_frame)

    for frame_idx in range(1, n_frames):
        cur_frame = cv2.imread(img_list[frame_idx])
        ret_frame = cur_frame.copy()
        success, boxes = trackers.update(cur_frame)
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(ret_frame, p1, p2, colors[i], 2, 1)
        ret_images.append(ret_frame)

    return ret_images




