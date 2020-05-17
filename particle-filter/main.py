from particle import Particle
import numpy as np
import os
import cv2

def printParticles(image, particles, bbox, text, printEstimated=True):
    for i in range(len(particles)):
        part = particles[i].astype(np.int)
        x = int(part[1]-(bbox[1]/2))
        w = int(x + bbox[1])
        y = int(part[0]-(bbox[0]/2))
        h = int(y + bbox[0])
        color = (0, 255, 0)
        if printEstimated and i >= len(particles)-1:
            color = (255, 0, 0)
        cv2.rectangle(image, (x, y), (w, h), color, 2)
    cv2.putText(image, text, (5, 20), 0, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Particles", (5, 230), 0, 0.5, (0, 255, 0), 1)
    cv2.putText(image, "Estimation particle", (100, 230), 0, 0.5, (255, 0, 0), 1)
    cv2.imshow("Frame", image)
    cv2.waitKey(0)

image_folder_path = "./images/"

should_initialize = True
lower_color = np.array([0, 130, 230], dtype='uint8')
upper_color = np.array([40, 255, 255], dtype='uint8')
number_of_particles = 20
diffuse_dispersion = 35
bbox_size = (30, 30)
init_pos = np.array([0, 120])

image_names = os.listdir(image_folder_path)
image_names.sort()

particle = Particle(number_of_particles, diffuse_dispersion, bbox_size=bbox_size, init_pos=init_pos)

for image_name in image_names:
    image = cv2.imread(image_folder_path + image_name)

    if should_initialize:
        should_initialize = False
        particle.initialize(image.shape)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower_color, upper_color)

    particle.evaluate(mask)
    particles, bbox = particle.estimate()

    printParticles(image.copy(), particles, bbox, "Initialization/prediction and estimation")

    particles, bbox = particle.select()
    select_image = image.copy()

    printParticles(image.copy(), particles, bbox, "Selection")

    particles, bbox = particle.diffuse()
    diffuse_image = image.copy()

    printParticles(image.copy(), particles, bbox, "Diffusion", printEstimated=False)

    particle.predict()





