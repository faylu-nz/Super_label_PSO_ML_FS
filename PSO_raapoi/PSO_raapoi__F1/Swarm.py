import numpy as np
import Particle


class Swarm:

    def __init__(self, n_particles, length, pos_max, pos_min, vel_max, vel_min, problem, n_iterations):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.problem = problem

        w = 0.8
        c1 = 1.46
        c2 = 1.46
        self.population = [Particle.Particle(length=length,
                                             pos_max=pos_max, pos_min=pos_min,
                                             vel_max=vel_max, vel_min=vel_min,
                                             w=w, c1=c1, c2=c2, problem=problem)
                           for _ in range(n_particles)]

    def iterate(self):

        for i in range(self.n_iterations):
            new_w = 0.9 - i * (0.9 - 0.4) / self.n_iterations
            print('Iterate ', i, end='  ')
            gbest_fit = self.population[0].gbest_fit
            gbest_index = 0
            gbest_updated = False
            print('gbest value is ', gbest_fit)

            for index, particle in enumerate(self.population):
                # Evaluate each particle, update pbest
                particle.w = new_w
                particle.fitness = self.problem.fitness(particle.position)

                if self.problem.is_better(particle.fitness, particle.pbest_fit):
                    particle.pbest_fit = particle.fitness
                    particle.pbest_pos = np.copy(particle.position)

                if self.problem.is_better(particle.pbest_fit, gbest_fit):
                    gbest_fit = particle.pbest_fit
                    gbest_index = index
                    gbest_updated = True

            if gbest_updated:
                for particle in self.population:
                    particle.gbest_fit = self.population[gbest_index].pbest_fit
                    particle.gbest_pos = np.copy(
                        self.population[gbest_index].pbest_pos)

            # now update particle position:
            for particle in self.population:
                particle.update()

        return self.population[0].gbest_pos, self.population[0].gbest_fit
