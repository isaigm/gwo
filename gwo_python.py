import gwo_py
import numpy as np
import time

class ProblemaEsfera(gwo_py.Problem):
    def __init__(self, setup):
        super().__init__(setup)

    def fitness_batch(self, pos_matrix: np.ndarray) -> np.ndarray:
        return np.sum(pos_matrix**2, axis=1)

config = gwo_py.Setup()
config.N = 5
config.POP_SIZE = 50
config.maxRange = [10.0] * 5
config.minRange = [-10.0] * 5

problema = ProblemaEsfera(config)


start_time = time.time()
lobo_alfa = problema.run(maxIterations=1000)
end_time = time.time()


print(f"Best fitness {lobo_alfa.savedFitness:.6f}")
print(f"Best solution:")
print(lobo_alfa.pos)
print(f"Execution time: {end_time - start_time:.4f} secs")