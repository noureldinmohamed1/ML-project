import random
import copy
# import your built in NN class

class GA_Optimization:

    def __init__(self, X_train, y_train, X_val, y_val, population_size, crossover_rate, mutation_rate, generations):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.best_individual = None
        self.best_fitness = 0

    """
    Genetic Algorithm Steps:
        Initialize population
        Evaluate fitness
        Selection
        Crossover
        Mutation
    """

    def initialize_population(self):
        population = []
        """
        Hyperparameters to Optimize
        • Hidden layers: 1-5
        • Neurons per layer: 32-512
        • Activation: relu, tanh, sigmoid
        • Learning rate: 1e-1 to 1e-5
        • Batch size: 16, 32, 64, 128
        • Optimizer: SGD, Adam, RMSProp, Adagrad
        • Epochs: 3-20
        """
        for i in range(self.population_size):
            individual = {
                "hidden_layers": random.randint(1, 5),
                "neurons_per_layer": random.randint(32, 512),
                "activation": random.choice(["relu", "tanh", "sigmoid"]),
                "learning_rate": random.uniform(1e-5, 1e-1),
                "batch_size": random.choice([16, 32, 64, 128]),
                "optimizer": random.choice(["SGD", "Adam", "RMSProp", "Adagrad"]),
                "epochs": random.randint(3, 20)
            }
            population.append(individual)
        return population

    def evaluate_fitness(self, individual):
        # change NN_builtin with the name of your class
        nn = NN_builtin(
            hidden_layers=individual["hidden_layers"],
            neurons_per_layer=individual["neurons_per_layer"],
            activation=individual["activation"],
            learning_rate=individual["learning_rate"],
            batch_size=individual["batch_size"],
            optimizer=individual["optimizer"],
            epochs=individual["epochs"]
        )
        # change train and compute_accuracy with your methods names
        nn.train(self.X_train, self.y_train)
        accuracy = nn.compute_accuracy(self.X_val, self.y_val)
        return accuracy

    def select_parents(self, population, fitnesses, k=3):
        # Tournament selection
        selected = []
        for i in range(2):
            tournament_indices = random.sample(range(len(population)), k)
            tournament_fitnesses = []
            for j in tournament_indices:
                tournament_fitnesses.append(fitnesses[j])
            winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            selected.append(population[winner_index])
        return selected[0], selected[1]

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2) # because they can be mutated later
        child1 , child2 = {} , {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    def mutate(self, individual):
        for key in individual.keys():
            if random.random() < self.mutation_rate:
                if key == "hidden_layers":
                    individual[key] = random.randint(1, 5)
                elif key == "neurons_per_layer":
                    individual[key] = random.randint(32, 512)
                elif key == "activation":
                    individual[key] = random.choice(["relu", "tanh", "sigmoid"])
                elif key == "learning_rate":
                    individual[key] = random.uniform(1e-5, 1e-1)
                elif key == "batch_size":
                    individual[key] = random.choice([16, 32, 64, 128])
                elif key == "optimizer":
                    individual[key] = random.choice(["SGD", "Adam", "RMSProp", "Adagrad"])
                elif key == "epochs":
                    individual[key] = random.randint(3, 20)
        return individual
    
    def run(self):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitnesses = []
            for individual in population:
                fitnesses.append(self.evaluate_fitness(individual))
            best_index = fitnesses.index(max(fitnesses))
            best_individual = population[best_index]
            best_fitness = fitnesses[best_index]
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_individual = best_individual
            print(f"Generation {generation+1}: Best Fitness = {self.best_fitness} with Hyperparameters = {self.best_individual}")
            new_population = []
            # Elitism
            new_population.append(copy.deepcopy(self.best_individual)) # because it can be mutated later
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            population = new_population
        return self.best_individual , self.best_fitness