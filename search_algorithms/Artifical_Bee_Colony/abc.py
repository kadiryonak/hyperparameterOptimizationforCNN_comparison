
# ---------------------------
# ARTIFICIAL BEE COLONY (ABC)
# ---------------------------
class ABC(HyperparameterSearch):
    """ABC optimizer. fitness_function should return a scalar objective (float) to MINIMIZE."""
    def __init__(self, conf: Config, fitness_function):
        self.conf = conf
        self.fitness_function = fitness_function
        self.foods = np.zeros((conf.FOOD_NUMBER, conf.DIMENSION))
        self.f = np.ones(conf.FOOD_NUMBER) * np.inf
        self.fitness = np.zeros(conf.FOOD_NUMBER)
        self.trial = np.zeros(conf.FOOD_NUMBER, dtype=int)
        self.prob = np.zeros(conf.FOOD_NUMBER)
        self.globalParams = np.zeros(conf.DIMENSION)
        self.globalOpt = np.inf if conf.MINIMIZE else -np.inf

        if not conf.RANDOM_SEED:
            set_global_seed(conf.SEED)

    def _obj_to_fitness(self, obj_val: float):
        if obj_val >= 0:
            return 1.0 / (1.0 + obj_val)
        else:
            return 1.0 + abs(obj_val)

    def init_food_source(self, index: int):
        for d in range(self.conf.DIMENSION):
            self.foods[index, d] = random.uniform(self.conf.LOWER_BOUND[d], self.conf.UPPER_BOUND[d])
        try:
            obj = float(self.fitness_function(self.foods[index]))
        except Exception as e:
            # if fitness function crashes, set large objective so it is treated as bad
            print(f"Error evaluating fitness at init (index {index}): {e}")
            obj = float(np.inf)
        self.f[index] = obj
        self.fitness[index] = self._obj_to_fitness(obj)
        self.trial[index] = 0

    def initial(self):
        for i in range(self.conf.FOOD_NUMBER):
            self.init_food_source(i)
        best_idx = np.argmin(self.f) if self.conf.MINIMIZE else np.argmax(self.f)
        self.globalOpt = float(self.f[best_idx])
        self.globalParams = self.foods[best_idx].copy()

    def send_employed_bees(self):
        for i in range(self.conf.FOOD_NUMBER):
            param2change = random.randint(0, self.conf.DIMENSION - 1)
            neighbour = random.randint(0, self.conf.FOOD_NUMBER - 1)
            while neighbour == i:
                neighbour = random.randint(0, self.conf.FOOD_NUMBER - 1)
            solution = self.foods[i].copy()
            phi = random.uniform(-1, 1)
            solution[param2change] = solution[param2change] + phi * (solution[param2change] - self.foods[neighbour, param2change])
            solution[param2change] = np.clip(solution[param2change], self.conf.LOWER_BOUND[param2change], self.conf.UPPER_BOUND[param2change])
            try:
                obj = float(self.fitness_function(solution))
            except Exception as e:
                print(f"Error evaluating fitness (employed) for bee {i}: {e}")
                obj = float(np.inf)
            new_fitness = self._obj_to_fitness(obj)
            if new_fitness > self.fitness[i]:
                self.foods[i] = solution
                self.f[i] = obj
                self.fitness[i] = new_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1

    def calculate_probabilities(self):
        maxfit = np.max(self.fitness)
        if maxfit == 0:
            self.prob = np.ones(self.conf.FOOD_NUMBER) * 0.1
        else:
            self.prob = 0.9 * (self.fitness / maxfit) + 0.1

    def send_onlooker_bees(self):
        i = 0
        t = 0
        while t < self.conf.FOOD_NUMBER:
            if random.random() < self.prob[i]:
                param2change = random.randint(0, self.conf.DIMENSION - 1)
                neighbour = random.randint(0, self.conf.FOOD_NUMBER - 1)
                while neighbour == i:
                    neighbour = random.randint(0, self.conf.FOOD_NUMBER - 1)
                solution = self.foods[i].copy()
                phi = random.uniform(-1, 1)
                solution[param2change] = solution[param2change] + phi * (solution[param2change] - self.foods[neighbour, param2change])
                solution[param2change] = np.clip(solution[param2change], self.conf.LOWER_BOUND[param2change], self.conf.UPPER_BOUND[param2change])
                try:
                    obj = float(self.fitness_function(solution))
                except Exception as e:
                    print(f"Error evaluating fitness (onlooker) for bee {i}: {e}")
                    obj = float(np.inf)
                new_fitness = self._obj_to_fitness(obj)
                if new_fitness > self.fitness[i]:
                    self.foods[i] = solution
                    self.f[i] = obj
                    self.fitness[i] = new_fitness
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1
                t += 1
            i = (i + 1) % self.conf.FOOD_NUMBER

    def send_scout_bees(self):
        max_trial_index = np.argmax(self.trial)
        if self.trial[max_trial_index] >= self.conf.LIMIT:
            self.init_food_source(max_trial_index)

    def memorize_best_source(self):
        if self.conf.MINIMIZE:
            min_index = np.argmin(self.f)
            if self.f[min_index] < self.globalOpt:
                self.globalOpt = float(self.f[min_index])
                self.globalParams = self.foods[min_index].copy()
        else:
            max_index = np.argmax(self.f)
            if self.f[max_index] > self.globalOpt:
                self.globalOpt = float(self.f[max_index])
                self.globalParams = self.foods[max_index].copy()

    def run(self, max_cycles: int = None):
        if max_cycles is None:
            max_cycles = self.conf.MAXIMUM_EVALUATION
        self.initial()
        history = []
        for cycle in range(max_cycles):
            self.send_employed_bees()
            self.calculate_probabilities()
            self.send_onlooker_bees()
            self.send_scout_bees()
            self.memorize_best_source()
            if self.conf.SHOW_PROGRESS:
                print(f"Cycle {cycle+1}/{max_cycles} - Best objective: {self.globalOpt:.6f}")
            history.append((cycle + 1, self.globalOpt, self.globalParams.copy()))
        return self.globalParams.copy(), self.globalOpt, history
