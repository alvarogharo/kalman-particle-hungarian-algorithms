from particle import Particle
import numpy as np
import os
import cv2

image_folder_path = "./images/"

should_initialize = True
lower_color = np.array([0, 130, 230], dtype='uint8')
upper_color = np.array([40, 255, 255], dtype='uint8')

image_names = os.listdir(image_folder_path)
image_names.sort()

particle = Particle(20, 35, bbox_size=(30, 30), init_pos=np.array([0, 120]))

for image_name in image_names:
    image = cv2.imread(image_folder_path + image_name)

    if should_initialize:
        should_initialize = False
        particle.intialize(image.shape)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower_color, upper_color)

    particle.evaluate(mask)
    particles, bbox = particle.estimate()

    init_image = image.copy()

    for i in range(len(particles)):
        part = particles[i].astype(np.int)
        x = int(part[1]-(bbox[1]/2))
        w = int(x + bbox[1])
        y = int(part[0]-(bbox[0]/2))
        h = int(y + bbox[0])
        if i < 19:
            color = (0, 255, 0)
        else:
            color = (255, 0 , 0)
        cv2.rectangle(init_image, (x, y), (w, h), color, 2)
    cv2.imshow("Frame", init_image)
    cv2.waitKey(0)

    particles, bbox = particle.select()
    select_image = image.copy()

    for i in range(len(particles)):
        part = particles[i].astype(np.int)
        x = int(part[1]-(bbox[1]/2))
        w = int(x + bbox[1])
        y = int(part[0]-(bbox[0]/2))
        h = int(y + bbox[0])
        if i < 19:
            color = (0, 255, 0)
        else:
            color = (255, 0 , 0)
        cv2.rectangle(select_image, (x, y), (w, h), color, 2)
    cv2.imshow("Frame", select_image)
    cv2.waitKey(0)

    particles, bbox = particle.diffuse()
    diffuse_image = image.copy()

    for i in range(len(particles)):
        part = particles[i].astype(np.int)
        x = int(part[1] - (bbox[1] / 2))
        w = int(x + bbox[1])
        y = int(part[0] - (bbox[0] / 2))
        h = int(y + bbox[0])
        color = (0, 255, 0)
        cv2.rectangle(diffuse_image, (x, y), (w, h), color, 2)
    cv2.imshow("Frame", diffuse_image)
    cv2.waitKey(0)

    particle.predict()


