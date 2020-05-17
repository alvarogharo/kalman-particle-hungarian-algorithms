import numpy as np
import random

class Particle():
    def __init__(self, number_of_particles, sigma, bbox_size=(14, 14), init_pos=np.array([0, 0])):
        self.num_part = number_of_particles
        self.sigma = sigma
        self.particles = []
        self.bbox_size = bbox_size
        self.last_pos = init_pos
        self.current_estimation = np.array([0, 0])
        self.is_inside_frame = False
        self.image_shape = []

    def initialize(self, image_shape):
        self.image_shape = image_shape
        self.particles = np.random.rand(self.num_part, 3)
        self.particles[:, 0] = self.particles[:, 0] * image_shape[0]
        self.particles[:, 1] = self.particles[:, 1] * image_shape[0]
        print()

    def evaluate(self, mask_image):
        for particle in self.particles:
            y = int(max(particle[0]-(self.bbox_size[0]/2), 0))
            h = int(min(particle[0]+(self.bbox_size[0]/2), mask_image.shape[0]))
            x = int(max(particle[1] - (self.bbox_size[1] / 2), 0))
            w = int(min(particle[1] + (self.bbox_size[1] / 2), mask_image.shape[1]))
            inside_particle = mask_image[y:h, x:w]
            num_of_ones = np.count_nonzero(inside_particle)
            particle[2] = num_of_ones/(self.bbox_size[0] * self.bbox_size[1])*100
        total_weight = np.sum(self.particles[:, 2])
        if total_weight != 0.:
            self.particles[:, 2] = self.particles[:, 2] / total_weight

    def estimate(self):
        particles_copy = self.particles.copy()
        particles_copy.view('i8,i8,i8').sort(order=['f2'], axis=0)
        self.current_estimation = particles_copy[len(particles_copy)-1, 0:2]
        return particles_copy[:, 0:2], self.bbox_size

    def select(self):
        new_particles = []
        indexes = []
        accumulated = np.add.accumulate(self.particles[:, 2])
        self.is_inside_frame = np.sum(accumulated) != np.array([0])
        if self.is_inside_frame:
            for i in range(self.num_part):
                rand = random.random()
                conditions = np.argwhere(accumulated > rand)
                if conditions.size != 0:
                    indexes.append(conditions[0][0])
            for index in indexes:
                new_particles.append(self.particles[index])
            new_particles = np.array(new_particles)
            self.particles = new_particles.copy()
        else:
            new_particles = self.particles.copy()
        new_particles.view('i8,i8,i8').sort(order=['f2'], axis=0)
        return new_particles[:, 0:2], self.bbox_size

    def diffuse(self):
        if self.is_inside_frame:
            for particle in self.particles:
                rand = np.array((random.random() * self.sigma, random.random() * self.sigma))
                sign = np.array((random.random(), random.random()))
                if sign[0] < 0.5:
                    rand[0] = -rand[0]
                if sign[1] < 0.5:
                    rand[1] = -rand[1]
                particle[0] += rand[0]
                particle[1] += rand[1]
        particle_copy = self.particles.copy()
        particle_copy.view('i8,i8,i8').sort(order=['f2'], axis=0)
        return particle_copy[:, 0:2], self.bbox_size

    def predict(self):
        if self.is_inside_frame:
            self.is_inside_frame = False
            movement = self.current_estimation - self.last_pos
            for particle in self.particles:
                particle[0] += movement[0]
                particle[1] += movement[1]
            self.last_pos = self.current_estimation
        else:
            self.initialize(self.image_shape)


