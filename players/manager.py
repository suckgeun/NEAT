from players.worker import Worker
from globaladmin.workplace import Workplace
import random
import os
import numpy as np
from players.neuralnet import NeuralNetwork


class Manager:
    def __init__(self, n_nns, n_inputs, n_outputs, bias=None, c1=1, c2=1, c3=0.4, drop_rate=0.8,
                 weight_max=10, weight_min=-10, weight_mutate_rate=0.1, pm_weight_random=0.1):
        self.worker = None
        self.workplace = None
        self.n_nns = n_nns
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.bias = bias
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.drop_rate = drop_rate
        self.nn_best = None
        self.weight_max = weight_max
        self.weight_min = weight_min
        self.weight_mutate_rate = weight_mutate_rate
        self.pm_weight_random = pm_weight_random

    def initialize(self):
        self.workplace = Workplace(self.n_inputs, self.n_outputs, bias=None, n_nn=self.n_nns,
                                   c1=self.c1, c2=self.c2, c3=self.c3, drop_rate=self.drop_rate,
                                   weight_max=self.weight_max, weight_min=self.weight_min,
                                   weight_mutate_rate=self.weight_mutate_rate,
                                   pm_weight_random=self.pm_weight_random)
        self.worker = Worker(self.workplace)
        self.worker.initialize_workplace()

    def get_action(self, inputs, nn):
        self.worker.activate_neurons(inputs, nn)

        n_input_bias = self.workplace.n_input + self.workplace.n_bias
        n_input_bias_output = n_input_bias + self.workplace.n_output

        results = nn.results[n_input_bias: n_input_bias_output]
        result = results.index(max(results))
        return result

    def adjust_fitness(self):
        self.worker.calc_fitness_adjusted()

    def make_children(self):
        self.worker.speciate()

        for nn in self.workplace.nns:
            if nn.is_champ:
                print("champion agent: {0}".format(nn.fitness_previous))
            print("nn {0}) fitness: {1}, species: {2}, total_connects: {3}, valid_connects: {4}".format(
                self.workplace.nns.index(nn), nn.fitness, nn.species, len(nn.connect_genes),
                len(nn.connect_genes[nn.connect_genes[:, 3] == 1])))

        # choose better nns
        nns_keeping, species_keeping_nns, fitnesses_keeping = self.worker.select_better_nns()

        # keeps champions
        new_nns = []
        count = 0
        while count < self.workplace.n_champion_keeping:
            nn_best = None
            count += 1
            for nn in self.workplace.nns:
                if nn not in new_nns:
                    if nn_best is None:
                        nn_best = nn
                    else:
                        if nn_best.fitness < nn.fitness:
                            nn_best = nn
            nn_best.is_champ = True
            nn_best.fitness_previous = nn_best.fitness
            new_nns.append(nn_best)

        # get sum of adjusted fitness of each species
        sum_fitness, total_fitness = self.worker.get_sum_fitness(species_keeping_nns, fitnesses_keeping)
        print("sum_fitness: {0}".format(sum_fitness))

        # assign number of children to reproduce based on each species sum of adjusted fitness
        children_assigned = self.worker.calc_children_assign_num(sum_fitness, total_fitness)
        print("children_assigned: {0}".format(children_assigned))

        # reproduce children of each species
        for species_num, n_children in children_assigned.items():

            for i in range(n_children):
                pc = random.random()
                # crossover
                if pc <= self.workplace.pc:
                    mother = species_num
                    father = species_num
                    pc_interspecies = random.random()
                    # interspecies crossover
                    if len(children_assigned) > 1:
                        if pc_interspecies <= self.workplace.pc_interspecies:
                            species1, species2 = random.sample(list(children_assigned), 2)
                            father = species1

                            if species1 == species_num:
                                father = species2

                    mother_total_fitness = sum_fitness[mother]
                    father_total_fitness = sum_fitness[father]
                    parent1 = self.worker.choose_parent(mother, nns_keeping, mother_total_fitness)
                    parent2 = self.worker.choose_parent(father, nns_keeping, father_total_fitness)

                    child = self.worker.crossover(parent1, parent2)
                else:
                    # no crossover
                    parent1 = self.worker.choose_parent(species_num, nns_keeping, sum_fitness[species_num])
                    child = parent1.copy()

                # weight mutation
                # disable mutation also taken care of here
                pm_weight = random.random()
                if pm_weight <= self.workplace.pm_weight:
                    self.worker.mutate_weight(child)

                # node mutation
                if n_children < self.workplace.small_group:
                    pm_node_setting = self.workplace.pm_node_small
                else:
                    pm_node_setting = self.workplace.pm_node_big

                pm_node = random.random()
                if pm_node <= pm_node_setting:
                    self.worker.mutate_node(child)

                # connection mutation
                if n_children < self.workplace.small_group:
                    pm_connect_setting = self.workplace.pm_connect_small
                else:
                    pm_connect_setting = self.workplace.pm_connect_big

                pm_connect = random.random()
                if pm_connect <= pm_connect_setting:
                    self.worker.mutate_connect(child)

                # disable mutation
                pm_disable = random.random()
                if pm_disable <= self.workplace.pm_disable:
                    self.worker.mutate_disable(nn)

                # enable mutation
                pm_enable = random.random()
                if pm_enable <= self.workplace.pm_enable:
                    self.worker.mutate_enable(nn)

                new_nns.append(child)

        assert len(new_nns) == self.workplace.n_nn, "children number does not match n_nn." + str(len(new_nns))

        return new_nns

    def create_next_generation(self):
        self.adjust_fitness()
        self.workplace.nns = self.make_children()

    def write_best_nn(self, filename):

        assert self.nn_best is not None, "nn best is not ready yet"

        dir_path = os.path.join(os.getcwd(), "results")
        file_path = os.path.join(dir_path, filename)

        f = open(file_path, "w")
        f.write("connect_genes\n")
        for gene in self.nn_best.connect_genes[:, :5]:
            f.write(str(gene))
            f.write("\n")

        f.write("node_indices\n")
        f.write(str(self.nn_best.node_indices))
        f.write("\n")

        f.write("results\n")
        f.write(str(self.nn_best.results))

        f.write("\n")
        f.write("fitness: {0}".format(str(self.nn_best.fitness)))
        f.close()

    def remember_best_nn(self, nn=None):

        if self.nn_best is None:
            self.nn_best = self.workplace.nns[0]

        if nn is None:
            for nn in self.workplace.nns:
                if self.nn_best.fitness < nn.fitness:
                    self.nn_best = nn
        else:
            self.nn_best = nn

    @staticmethod
    def recreate_best_nn(filename):

        dir_path = os.path.join(os.getcwd(), "results")
        file_path = os.path.join(dir_path, filename)

        assert os.path.isfile(file_path), "file does not exist"

        connects = None
        node_indices = []
        results = []

        with open(file_path) as file:

            for line in file:
                print(line)
                if line.startswith("connect_genes"):
                    line = file.readline()
                    line = line.replace('[', '').replace(']', '').replace('\n', '')
                    nums = line.split()
                    gene = []
                    for num in nums:
                        gene.append(float(num))

                    gene.append(0)
                    gene.append(0)

                    connects = np.array(gene)

                    for line in file:
                        if line[0] == '[':
                            if ']' not in line:
                                line += '  ' + file.readline()
                            line = line.replace('[', '').replace(']', '').replace('\n', '')
                            nums = line.split()
                            gene = []
                            for num in nums:
                                gene.append(float(num))

                            gene.append(0)
                            gene.append(0)

                            connects = np.vstack((connects, gene))
                        elif line.startswith("node_indices"):
                            line = file.readline()
                            line = line.replace('[', '').replace(']', '').replace('\n', '').replace(' ', '')
                            nums = line.split(',')
                            for num in nums:
                                node_indices.append(float(num))

                        elif line.startswith("results"):
                            line = file.readline()
                            line = line.replace('[', '').replace(']', '').replace('\n', '').replace(' ', '')
                            nums = line.split(',')
                            for num in nums:
                                results.append(float(num))

        nn_best = NeuralNetwork()
        nn_best.connect_genes = connects
        nn_best.node_indices = node_indices
        nn_best.results = results

        return nn_best








