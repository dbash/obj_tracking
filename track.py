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
        #cur_labels = detect.label_img(detect.segment_morph(cur_frame, False), centroids=False)
        ret_frame = cur_frame.copy()
        success, boxes = trackers.update(cur_frame)
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(ret_frame, p1, p2, colors[i], 2, 1)
        ret_images.append(ret_frame)

    return ret_images


def simple_multitracker_greedy(img_list):
    def dist(point, point_dict):
        min_dist = 1000000
        min_dist_idx = -1
        for key in point_dict.keys():
            distance = np.sqrt(np.sum((np.array(point)  - np.array(point_dict[key])) ** 2))
            if distance < min_dist:
                min_dist = distance
                min_dist_idx = key
        return min_dist, min_dist_idx

    def best_overlap(bbox, labels): #finding the object that overlaps with the bbox
        max_overlap = 0
        max_overlap_idx = -1
        bbox_arr = [int(i) for i in bbox]
        for obj_idx in np.unique(labels)[1:]:
            mask = labels==obj_idx
            overlap = np.sum(mask[bbox_arr[1]: bbox_arr[1] + bbox_arr[3],
                                  bbox_arr[0]:bbox_arr[0] + bbox_arr[2]])/(bbox_arr[2]*bbox_arr[3])#/np.sum(mask)
            if overlap > max_overlap:
                max_overlap = overlap
                max_overlap_idx = obj_idx
        return max_overlap, max_overlap_idx

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
    min_overlap = 0.3

    for frame_idx in range(1, n_frames):
        cur_frame = cv2.imread(img_list[frame_idx])
        cur_labels = detect.label_img(detect.segment_morph(cur_frame, False), centroids=False)
        ref_labels = cur_labels.copy()
        _, cur_centroids = detect.mark_object_centroids(cur_labels)
        ret_frame = cur_frame.copy()
        used_box_idx = []
        success, boxes = trackers.update(cur_frame)
        found_objects = cur_centroids.copy()
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            if np.sum(cur_labels[int(newbox[1]):int(newbox[1] + newbox[3]),
                      int(newbox[0]): int(newbox[0] + newbox[2])]) > 0: #not showing the 'dead' tracks
                overlap, best_matching_obj_idx = best_overlap(newbox, ref_labels)
                if overlap>min_overlap: #asign the object with the closest centroid
                    #cx, cy = cur_centroids[best_matching_obj_idx]
                    # p1 = (int(newbox[0]), int(newbox[1]))
                    # p2 =  (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

                    cv2.rectangle(ret_frame, p1, p2, colors[i], 2, 1)
                    found_objects.pop(best_matching_obj_idx) #remove this object from the list
                    ref_labels[ref_labels==best_matching_obj_idx] = 0
                    used_box_idx.append(i)
                    #ref_labels[int(newbox[1]): int(newbox[1] + newbox[3]), int(newbox[0]):int(newbox[0] + newbox[2])] = 0
                # else: #trying to find the occluding objects
                #     overlap, best_matching_obj_idx = best_overlap(newbox, cur_labels)
                #     if distance < max_dist//4:
                #         cx, cy = cur_centroids[best_matching_obj_idx]
                #         p1 = (int(newbox[0]), int(newbox[1]))
                #         p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                #
                #         cv2.rectangle(ret_frame, p1, p2, colors[i], 2, 1)
                #         #found_objects.pop(best_matching_obj_idx)  # remove this object from the list

        if len(found_objects) > 0:
            for obj in found_objects.keys():
                bbox = cv2.boundingRect(np.array(cur_labels == obj, dtype='uint8'))
                mask = ref_labels == obj
                good_bbox = False
                for i, newbox in enumerate(boxes):
                    overlap = np.mean(mask[int(newbox[1]):int(newbox[1] + newbox[3]),
                                      int(newbox[0]): int(newbox[0] + newbox[2])])

                    if overlap > min_overlap/5:
                        if (i in used_box_idx):
                            good_bbox = True
                            break
                        else:
                            p1 = (int(newbox[0]), int(newbox[1]))
                            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                            cv2.rectangle(ret_frame, p1, p2, colors[i], 2, 1)
                            ref_labels[ref_labels == obj] = 0
                            used_box_idx.append(i)
                            good_bbox = True
                            break

                if not good_bbox:
                    trackers.add(cv2.TrackerCSRT_create(), cur_frame, bbox)
                    colors.append((np.random.randint(0, 255), np.random.randint(0, 255),
                                    np.random.randint(0, 255)))
                    cv2.rectangle(ret_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                    colors[-1], 2)
        ret_images.append(ret_frame)

    return ret_images

#assign tracks to the closest found objects
def greedy_assignment(tracks, object_centroids):
    def dist(point, point_list):
        return np.sqrt(np.sum(point ** 2 - np.array(point_list) ** 2, axis=1))








def kalman(img_list):
    n_frames = len(img_list)
    cur_frame = cv2.imread(img_list[0])
    cur_labels = detect.label_img(detect.segment_morph(cur_frame, False), centroids=False)
    cur_labels, cur_obj_centroids = detect.mark_object_centroids(cur_labels)
    out_tracking_predictions = []
    colors = []
    ret_frame = cur_frame.copy()
    trackers = {}
    i = 0
    for obj_idx in np.unique(cur_labels):  # initializing trackers for each detected object
        if obj_idx == 0:
            continue
        bbox = cv2.boundingRect(np.array(cur_labels == obj_idx, dtype='uint8'))

        cv2.rectangle(ret_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      (0, 255, 0), 2)
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255),
                       np.random.randint(0, 255)))
        trackers.update({i:([cur_obj_centroids[obj_idx]], colors[-1])})
        i+=1
    out_tracking_predictions.append(ret_frame)

    #init kalman
    cur_frame = cv2.imread(img_list[0])
    cur_labels = detect.label_img(detect.segment_morph(cur_frame, False), centroids=False)
    cur_labels, cur_obj_centroids = detect.mark_object_centroids(cur_labels)
    ret_frame = cur_frame.copy()
    i=0
    for obj_idx in np.unique(cur_labels): #need second frame to initialize kalman
        if obj_idx == 0:
            continue
        bbox = cv2.boundingRect(np.array(cur_labels == obj_idx, dtype='uint8'))
        cv2.rectangle(ret_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                      (0, 255, 0), 2)
        trackers[i][0].append(cur_obj_centroids[obj_idx])
        i += 1
    out_tracking_predictions.append(ret_frame)

    # always the same 2D case
    Transition_Matrix = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
    Observation_Matrix = [[1, 0, 0, 0], [0, 1, 0, 0]]
    initcovariance = 1.0e-3 * np.eye(4)
    transistionCov = 1.0e-4 * np.eye(4)
    observationCov = 1.0e-1 * np.eye(2)


    #greedy track assignment

