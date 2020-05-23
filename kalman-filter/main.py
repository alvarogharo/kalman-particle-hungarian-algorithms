from kalman import Kalman
import numpy as np
import os
import cv2

image_folder_path = "./images/"
measurements_path = "./measurements/ball_20.csv"


dt = 0.1
u = 0 # Acceleration
std_acc = 0.001  # Q
std_pos = 0.5 # R

image_names = os.listdir(image_folder_path)
image_names.sort()

measurements = np.genfromtxt(measurements_path, delimiter=",")

kalman = Kalman()
kalman.init(dt, u, std_acc, std_pos)

for i in range(len(image_names)):
    image = cv2.imread(image_folder_path + image_names[i])
    measurement = measurements[i].astype(np.int)
    # Given measurement color BLUE
    cv2.rectangle(image, (measurement[0], measurement[1]),
                  (measurement[0]+measurement[2], measurement[1]+measurement[3]), (255, 0, 0), 2)

    (x, y, vx, vy) = kalman.predict()
    # Predicted state color GREEN
    cv2.rectangle(image, (x, y),
                  (x + measurement[2], y + measurement[3]), (0, 255, 0), 2)

    update = np.array([[measurement[0]+(measurement[2]/2)], [measurement[1]+(measurement[3]/2)]])
    (x1, y1, vx1, vy1) = kalman.correct(update)
    # Estimated state color WHITE
    cv2.rectangle(image, (x1, y1),
                  (x1 + measurement[2], y1 + measurement[3]), (255, 255, 255), 2)

    cv2.putText(image, "Estimated velocity: " + str((vx1[0], vy1[0])), (20, 680), 0, 0.5, (255, 255, 255), 2)
    cv2.putText(image, "Predicted velocity: " + str((vx[0], vy[0])), (300, 680), 0, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", image)
    cv2.waitKey(0)

