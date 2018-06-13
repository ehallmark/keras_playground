import numpy as np


class GeneticAlgorithm:
    def __init__(self, mutate_prob=0.5, cross_over_prob=0.5, solution_generator=None):
        self.solution_generator = solution_generator
        self.mutate_prob = mutate_prob
        self.cross_over_prob = cross_over_prob

    def fit(self, data, num_solutions=30, num_epochs=100):
        # generate initial solutions
        solutions = [self.solution_generator() for _ in range(num_solutions)]

        for epoch in range(num_epochs):
            # mutate and crossover
            curr_num_solutions = len(solutions)
            for i in range(curr_num_solutions):
                if np.random.rand(1) < self.mutate_prob:
                    solutions.append(solutions[i].mutate())
                if np.random.rand(1) < self.cross_over_prob:
                    cross_idx = int(np.random.randint(0, curr_num_solutions, (1,)))
                    solutions.append(solutions[i].cross_over(solutions[cross_idx]))

            # score solutions
            solutions = [(solution, solution.score(data)) for solution in solutions]
            # sort solutions
            solutions.sort(key=lambda x: x[1], reverse=True)
            # take top n solutions
            solutions = solutions[0: num_solutions]
            print('Best solution epoch '+str(epoch)+":", solutions[0].score(data))


class Solution:
    def score(self, data):
        pass

    def mutate(self):
        pass

    def cross_over(self, other):
        pass

