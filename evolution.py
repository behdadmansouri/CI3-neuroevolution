import copy
import random

import numpy as np

from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        sum_of_vals = 0
        least_fit = players[0].fitness
        fittest = players[0].fitness
        for player in players:
            sum_of_vals += player.fitness  # find sum of fitness
            if player.fitness < least_fit:
                least_fit = player.fitness  # find the least fit
            if player.fitness > fittest:
                fittest = player.fitness  # find fittest

        #  ROULETTE WHEEL
        random.shuffle(players)
        pick = random.uniform(0, sum_of_vals * 0.6)
        current = 0
        for player in players:
            current += player.fitness
            if current > pick:
                players.remove(player)  # remove 33% of the players

        # TODO (Additional: Learning curve)
        fitness_avg = sum_of_vals / len(players)  # find average fitness
        file = open('learning_curve.txt', 'a')  # write to file
        file.writelines(str(fitness_avg) + " " + str(fittest) + " " + str(least_fit) + "\n")
        file.close()
        # import matplotlib.pyplot as plt
        # import numpy as np
        #
        # x = np.linspace(0, 10*np.pi, 100)
        # y = np.sin(x)
        #
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # line1, = ax.plot(x, y, 'b-')
        #
        # for phase in np.linspace(0, 10*np.pi, 100):
        #     line1.set_ydata(np.sin(0.5 * x + phase))
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()
        #

        # return sorted(players, key=lambda item: item.fitness, reverse=True)[:199]
        return players[:num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        TNG = []

        for i in range(int(num_players * 0.9)):
            proto_child = self.clone_player(max(random.sample(prev_players, 40), key=lambda item: item.fitness))
            child = self.crossover(proto_child, random.sample(prev_players, 1)[0])

            rand = random.uniform(0, 1)
            if rand < 0.3:
                self.mutate(child, 0.5)
            TNG.append(child)

        for i in range(int(num_players * 0.2)):
            child = self.clone_player(max(random.sample(prev_players, 40), key=lambda item: item.fitness))
            rand = random.uniform(0, 1)
            if rand < 0.1:
                self.mutate(child, 0.2)
            TNG.append(child)

        random.shuffle(TNG)
        return TNG

    def crossover(self, p1, p2):
        p1.nn.w1 = p1.nn.w1 / 2 + p2.nn.w1 / 2
        p1.nn.w2 = p1.nn.w2 / 2 + p2.nn.w2 / 2
        p1.nn.b1 = p1.nn.b1 / 2 + p2.nn.b1 / 2
        p1.nn.b2 = p1.nn.b2 / 2 + p2.nn.b2 / 2
        return p1

    def mutate(self, child, radiation):
        sign = random.randint(-1, 1)
        child.nn.w1 += sign * radiation * np.random.normal(size=child.nn.w1.shape)
        child.nn.w2 += sign * radiation * np.random.normal(size=child.nn.w2.shape)
        child.nn.b1 += sign * radiation * np.random.normal(size=child.nn.b1.shape)
        child.nn.b2 += sign * radiation * np.random.normal(size=child.nn.b2.shape)

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
