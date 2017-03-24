from players.activation import sigmoid
import collections


class Workplace:
    """
    innov_history is dictionary with key:(in, out), value:counter
    """

    def __init__(self, n_input, n_output, n_nn=1, bias=200,
                 activ_func=sigmoid, c1=400, c2=400, c3=200, cmp_thr=1, drop_rate=0.8, small_group=5,
                 weight_max=10, weight_min=-10,
                 pc=0.75, pc_interspecies=0.001,
                 pm_node_small=0.03, pm_connect_small=0.05,
                 pm_node_big=0.3, pm_connect_big=0.7,
                 pm_weight=0.8, pm_weight_random=0.1, weight_mutate_rate=0.1,
                 pm_disable=0.4, pm_enable=0.2):

        assert n_input > -1 and n_output > -1 and n_nn > 0, "number of inputs, outputs and neural network " \
                                                            "should be positive integer"

        self.n_input = n_input
        self.n_output = n_output
        self.innov_counter = -1
        self.innov_history = {}
        self.n_nn = n_nn
        self.nns = []
        self.activ_func = activ_func
        self.node_genes_global = []
        self.inputs = None
        if bias is not None:
            self.n_bias = 1
            self.bias = bias
        else:
            self.n_bias = 0
            self.bias = None
        self.is_initialized = False
        self.fitnesses_adjusted = None
        self.species = collections.OrderedDict()
        self.species_of_nns = []
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.cmp_thr = cmp_thr
        self.drop_rate = drop_rate
        self.small_group = small_group
        self.weight_mutate_rate = weight_mutate_rate
        self.n_champion_keeping = 10
        # crossover probability
        self.pc = pc
        self.pc_interspecies = pc_interspecies
        # mutation probability
        self.pm_node_small = pm_node_small
        self.pm_node_big = pm_node_big
        self.pm_connect_small = pm_connect_small
        self.pm_connect_big = pm_connect_big
        self.pm_weight = pm_weight
        self.pm_weight_random = pm_weight_random
        self.pm_disable = pm_disable
        self.pm_enable = pm_enable
        self.weight_max = weight_max
        self.weight_min = weight_min


