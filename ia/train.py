import sys
import os
import neat
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'game'))

from game_engine import FlappyBirdEnv

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
N_GENERATIONS = 100


def evaluate_genome(genome, config):
    nb_saut = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = FlappyBirdEnv()
    state = env.reset()
    done = False

    while not done:
        output = net.activate(state)
        action = 1 if output[0] > 0.5 else 0
        state, reward, done = env.step(action)
        if action == 1:
            nb_saut +=1
            

    # frames survécus + 500 * tuyaux franchis
    return env.frames + 1500 * env.score - (nb_saut * -0.1)


def eval_genomes(genomes, config):
    best_score_gen = 0
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)
        score = (genome.fitness-0)//500
        if score > best_score_gen:
            best_score_gen = score
    print(f"meilleur score = {best_score_gen}")


def run():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    os.makedirs(os.path.join(os.path.dirname(__file__), 'checkpoints'), exist_ok=True)
    checkpointer = neat.Checkpointer(
        generation_interval=5,
        filename_prefix=os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint-')
    )
    population.add_reporter(checkpointer)

    best = population.run(eval_genomes, N_GENERATIONS)

    output_path = os.path.join(os.path.dirname(__file__), 'best_genome.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(best, f)

    print(f"\nMeilleur genome sauvegarde dans {output_path}")
    print(f"Fitness du meilleur genome : {best.fitness:.1f}")


if __name__ == '__main__':
    run()