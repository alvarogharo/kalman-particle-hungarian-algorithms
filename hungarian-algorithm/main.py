from hungarian import Hungarian
import numpy as np
import os
import cv2
import random


def calculate_cost_matrix(detections_mat):
    cost_mat = np.zeros((detections_mat.shape[1], detections_mat.shape[2]))
    for i in range(detections_mat.shape[1]):
        for j in range(detections_mat.shape[1]):
            centroid1 = np.array([detections_mat[0][i][0] + round(detections_mat[0][i][2]/2),
                                  detections_mat[0][i][1] + round(detections_mat[0][i][3]/2)])
            centroid2 = np.array([detections_mat[1][j][0] + round(detections_mat[1][j][2] / 2),
                                  detections_mat[1][j][1] + round(detections_mat[1][j][3] / 2)])
            cost_mat[j][i] = np.linalg.norm(detections_mat[0][i] - detections_mat[1][j])
    return cost_mat.astype(np.int)


image_folder_path = "./images/"
detections_path = "./detections/detections.csv"

colors = [(255, 0, 0), (0, 255, 0),
          (0, 0, 255), (0, 0, 0)]

cost_matrix = []
detection_coordinates = 4

image_names = os.listdir(image_folder_path)
image_names.sort()

detections = np.genfromtxt(detections_path, delimiter=",")

num_detections_per_track = round(detections.shape[1]/detection_coordinates)

cost_matrix = np.zeros((num_detections_per_track, num_detections_per_track))

hungarian = Hungarian()

for i in range(len(image_names)):
    image = cv2.imread(image_folder_path + image_names[i])

    if i == 0:
        detects = np.array([detections[i].astype(np.int).reshape((4, 4)), detections[i].astype(np.int).reshape((4, 4))])
    else:
        detects = np.array([detections[i-1].astype(np.int).reshape((4, 4)), detections[i].astype(np.int).reshape((4, 4))])

    cost_matrix = calculate_cost_matrix(detects)
    asociations = hungarian.start(cost_matrix)

    colors_to_use = [0, 0, 0, 0]
    for c in range(len(asociations)):
        colors_to_use[asociations[c]] = colors[c]

    colors = colors_to_use


    for idx, frame_detection in enumerate(detections[i].astype(np.int).reshape((4, 4))):

        cv2.rectangle(image, (frame_detection[0], frame_detection[1]),
                  (frame_detection[0]+frame_detection[2], frame_detection[1]+frame_detection[3]), colors_to_use[idx], 2)

    cv2.imshow("Frame", image)
    cv2.waitKey(0)




