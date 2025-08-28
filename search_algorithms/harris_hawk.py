import numpy as np
import random
from abc import ABC, abstractmethod
from search_algorithms.base_search import HyperparameterSearch

# ------------------------------
# HHO IMPLEMENTATION
# ------------------------------
class HHO(HyperparameterSearch):
    def __init__(self, param_space, dataset_loader, device, pop_size=5, iterations=10, fitness_func=None):
        super().__init__(param_space, dataset_loader, device)
        self.pop_size = pop_size
        self.iterations = iterations
        self.fitness_func = fitness_func  # CNN train function
        self.population = self._init_population()
        self.fitness = [self.fitness_func(agent) for agent in self.population]
        best_idx = np.argmax(self.fitness)
        self.best_agent = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    # ------------------------------
    # Population Initialization
    # ------------------------------
    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            agent = {key: random.randint(*self.param_space[key]) for key in self.param_space}
            pop.append(agent)
        return pop

    # ------------------------------
    # Agent Update
    # ------------------------------
    def _update_agent(self, agent, E):
        q = random.random()
        r = random.random()
        new_agent = agent.copy()
        
        if abs(E) >= 1:  # Exploration
            if q >= 0.5:
                rand_agent = {key: random.randint(*self.param_space[key]) for key in self.param_space}
                for key in agent:
                    new_agent[key] = int(rand_agent[key] - r * abs(rand_agent[key] - 2 * r * agent[key]))
            else:
                mean_agent = {key: int(np.mean([a[key] for a in self.population])) for key in agent}
                for key in agent:
                    LB, UB = self.param_space[key]
                    new_agent[key] = int((self.best_agent[key] - mean_agent[key]) - r * (LB + r * (UB - LB)))
        else:  # Exploitation
            for key in agent:
                new_agent[key] = int(self.best_agent[key] - E * abs(self.best_agent[key] - agent[key]))
        
        # Clip to bounds
        for key in agent:
            LB, UB = self.param_space[key]
            new_agent[key] = max(LB, min(UB, new_agent[key]))
        
        return new_agent

    # ------------------------------
    # Search / Main Optimization
    # ------------------------------
    def search(self):
        for t in range(self.iterations):
            E = 2 * (1 - t / self.iterations)  # Energy decreases linearly
            for i, agent in enumerate(self.population):
                self.population[i] = self._update_agent(agent, E)
                self.fitness[i] = self.fitness_func(self.population[i])
            
            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > self.best_fitness:
                self.best_agent = self.population[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]
            
            print(f"Iteration {t+1}, Best Fitness: {self.best_fitness}")
        
        return self.best_agent, self.best_fitness
