import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))

from game_engine import FlappyBirdEnv

class Perceptron:
    def __init__(self, n_inputs=5):
        self.weights = np.random.uniform(-1, 1, n_inputs)
        self.bias = np.random.uniform(-1, 1)

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        # Sigmoïde : ramène la sortie entre 0 et 1
        return np.tanh(z)
    def decide(self, x):
        return 1 if self.forward(x) > 0 else 0


def run(n_games=1):
    env = FlappyBirdEnv()
    nb_perceptron = 100
    mes_perceptron = []
    for numero_perceptron in range(nb_perceptron):
        numero_perceptron = Perceptron()
        mes_perceptron.append(numero_perceptron)
    scores = []

    for i in range(n_games):
        state = env.reset()
        done = False
        while not done:
            for perceptron in mes_perceptron:
                action = perceptron.decide(state)
                state, reward, done = env.step(action)
                scores.append(env.score)
                print(f"Partie {i + 1} : score = {env.score}")

    #print(f"Taille de la liste score : {len(scores)}")
    print(f"\nScore moyen des {nb_perceptron} perceptron sur {n_games} parties : {sum(scores) / len(scores):.1f}")

    best_score = max(scores)
    index_best_score = scores.index(best_score)
    #print(f"index du meilleur score dans la liste des scores = {scores.index(index_best_score)}")
    #print(f"score du meilleur perceptron perceptron = {best_score}")

    best_perceptron = mes_perceptron[index_best_score]
    
    print(f"poids du meilleur perceptron = {best_perceptron.weights}")

if __name__ == '__main__':
    run()