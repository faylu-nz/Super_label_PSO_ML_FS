import numpy as np

class Particle:

    def __init__(self, length, pos_max, pos_min, vel_max, vel_min, w, c1, c2, problem):
        self.length = length
        self.pos_max = pos_max
        self.pos_min = pos_min
        self.vel_max = vel_max
        self.vel_min = vel_min
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.problem = problem

        self.position = pos_min + np.random.rand(length)*(pos_max-pos_min)
        self.velocity = np.zeros(length)
        self.fitness = self.problem.worst_fitness()

        self.pbest_pos = np.zeros(length)
        self.pbest_fit = self.problem.worst_fitness()

        self.gbest_pos = np.zeros(length)
        self.gbest_fit = self.problem.worst_fitness()

    def update(self):
        # Update velocity
        self.velocity = self.w * self.velocity + \
            self.c1 * np.random.rand(self.length) * (self.pbest_pos - self.position) + \
            self.c2 * np.random.rand(self.length) * \
            (self.gbest_pos - self.position)

        self.velocity[self.velocity < self.vel_min] = self.vel_min
        self.velocity[self.velocity > self.vel_max] = self.vel_max

        # update position
        self.position = self.position + self.velocity
        self.position[self.position < self.pos_min] = self.pos_min
        self.position[self.position > self.pos_max] = self.pos_max
