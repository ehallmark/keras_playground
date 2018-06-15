import numpy as np


class GeneticAlgorithm:
    def __init__(self, mutate_prob=0.5, cross_over_prob=0.5, solution_generator=None,
                 after_epoch_func=None):
        self.after_epoch_func = after_epoch_func
        self.solution_generator = solution_generator
        self.mutate_prob = mutate_prob
        self.cross_over_prob = cross_over_prob
        self.solutions = []

    def fit(self, data, num_solutions=30, num_epochs=100):
        # generate initial solutions
        self.solutions = [self.solution_generator() for _ in range(num_solutions)]
        for epoch in range(num_epochs):
            # mutate and crossover
            curr_num_solutions = len(self.solutions)
            for i in range(curr_num_solutions):
                if np.random.rand(1) < self.mutate_prob:
                    self.solutions.append(self.solutions[i].mutate())
                if np.random.rand(1) < self.cross_over_prob:
                    cross_idx = int(np.random.randint(0, curr_num_solutions, (1,)))
                    self.solutions.append(self.solutions[i].cross_over(self.solutions[cross_idx]))

            # score solutions
            solutions_with_scores = [(solution, solution.score(data)) for solution in self.solutions]
            # sort solutions
            solutions_with_scores.sort(key=lambda x: x[1], reverse=True)
            # take top n solutions
            solutions_with_scores = solutions_with_scores[0: num_solutions]
            avg_score = 0.0
            self.solutions = []
            for solution_with_score in solutions_with_scores:
                avg_score += solution_with_score[1]
                self.solutions.append(solution_with_score[0])
            avg_score /= len(solutions_with_scores)

            print('Best solution epoch '+str(epoch)+":", solutions_with_scores[0][1])
            print('Avg solution epoch '+str(epoch)+":", avg_score)
            if self.after_epoch_func is not None:
                self.after_epoch_func()


class Solution:
    def score(self, data):
        pass

    def mutate(self):
        pass

    def cross_over(self, other):
        pass

