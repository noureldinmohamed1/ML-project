import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# ==========================
# Reproducibility
# ==========================
np.random.seed(42)
tf.random.set_seed(42)

# ==========================
# NN_builtin
# ==========================
class NN_builtin:
    def __init__(self, **kwargs):
        self.hparams = kwargs
        self.integrator = ModelIntegrator(load_data=False) 
        self.model = None

    def train(self, X_train, y_train):
        self.integrator.X_train, self.integrator.y_train = X_train, y_train
        self.model = self.integrator.build_model(self.integrator.map_hparams(self.hparams))
        self.model.fit(X_train, y_train, 
                       epochs=int(self.hparams.get('epochs', 5)), 
                       batch_size=int(self.hparams.get('batch_size', 32)), 
                       verbose=0)

    def compute_accuracy(self, X_val, y_val):
        loss, acc = self.model.evaluate(X_val, y_val, verbose=0)
        return acc

# ==========================
# Person 4: Model Integrator
# ==========================
class ModelIntegrator:
    def __init__(self, load_data=True):
        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = self.y_test = None
        if load_data:
            self.load_and_prep_data()

    def load_and_prep_data(self):
        try:
            from part1 import FashionMNISTDataPreprocessor
            preprocessor = FashionMNISTDataPreprocessor(binary_classes=(0, 1), random_state=42)
            data = preprocessor.prepare_for_optimization(n_components=100)
            self.X_train, self.y_train = data[6], data[7]
            self.X_val, self.y_val = data[8], data[9]
            self.X_test, self.y_test = data[10], data[11]
            # print(f"Data loaded | Input dimension = {self.X_train.shape[1]}")
        except Exception as e:
            print(f"Error loading data from Person 1: {e}")

    def map_hparams(self, params):
        opt_list = ["Adam", "SGD", "RMSProp", "Adagrad"]
        opt_val = params.get("optimizer", "Adam")
        if isinstance(opt_val, (float, int)):
            opt_name = opt_list[int(min(max(opt_val, 0), len(opt_list)-1))]
        else:
            opt_name = opt_val

        return {
            "num_layers": int(params.get("hidden_layers", params.get("num_layers", 1))),
            "hidden_size": int(params.get("neurons_per_layer", params.get("hidden_size", 64))),
            "activation": params.get("activation", "relu"),
            "learning_rate": float(params.get("learning_rate", 0.001)),
            "batch_size": int(params.get("batch_size", 32)),
            "epochs": int(params.get("epochs", 5)),
            "optimizer": opt_name
        }

    def build_model(self, hparams):
        K.clear_session()
        model = models.Sequential()
        model.add(layers.Input(shape=(self.X_train.shape[1],)))
        
        for _ in range(hparams["num_layers"]):
            model.add(layers.Dense(hparams["hidden_size"], activation=hparams["activation"]))
        
        model.add(layers.Dense(1, activation="sigmoid"))

        optimizer_map = {
            "Adam": tf.keras.optimizers.Adam,
            "SGD": tf.keras.optimizers.SGD,
            "RMSProp": tf.keras.optimizers.RMSprop,
            "Adagrad": tf.keras.optimizers.Adagrad
        }

        opt_class = optimizer_map.get(hparams["optimizer"], tf.keras.optimizers.Adam)
        optimizer = opt_class(learning_rate=hparams["learning_rate"])
        
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fitness_function(self, params):
        hparams = self.map_hparams(params)
        model = self.build_model(hparams)
        history = model.fit(self.X_train, self.y_train, 
                            validation_data=(self.X_val, self.y_val),
                            epochs=hparams["epochs"], 
                            batch_size=hparams["batch_size"], 
                            verbose=0)
        return max(history.history["val_accuracy"])

# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    integrator = ModelIntegrator()
    integrator.load_and_prep_data()

    try:
        from part3 import ParticleSwarmOptimizer
        pso_bounds = {
            "learning_rate": (0.0001, 0.01),
            "hidden_size": (16, 128),
            "hidden_layers": (1, 3),
            "batch_size": (16, 64),
            "epochs": (3, 10),
            "optimizer": (0, 3) 
        }
        
        print("\n--- Starting PSO Optimization ---")
        pso = ParticleSwarmOptimizer(fitness_function=integrator.fitness_function, 
                                     hyperparameter_bounds=pso_bounds, 
                                     n_particles=5, max_iterations=5)
        best_pso_params, best_pso_score, _ = pso.optimize()
        print(f"PSO Best Score: {best_pso_score:.4f}")
    except ImportError: 
        print("PSO (part3.py) not found.")
    try:
        from part2 import GA_Optimization
        
        print("\n--- Starting GA Optimization ---")
        ga = GA_Optimization(
            X_train=integrator.X_train, 
            y_train=integrator.y_train,
            X_val=integrator.X_val, 
            y_val=integrator.y_val,
            population_size=5, 
            crossover_rate=0.8, 
            mutation_rate=0.1, 
            generations=5
        )
        
        best_ga_params, best_ga_score = ga.run()
        print(f"\nGA Best Score: {best_ga_score:.4f}")
        print(f"Best GA Params: {best_ga_params}")
        
    except ImportError as e: 
        print(f"GA Module Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n==== Integration Process Finished ====")