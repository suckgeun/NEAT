from players.activation import sigmoid
import collections


class Workplace:
    """
    innov_history is dictionary with key:(in, out), value:counter
    """

    def __init__(self, n_input, n_output, n_nn=1, bias=None,
                 activ_func=sigmoid, c1=1, c2=1, c3=1, cmp_thr=1, drop_rate=0.5,
                 pc=0.8, pm=0.1, pm_node=0.1, pm_connect=0.1, pm_weight=0.8):

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
        # crossover probability
        self.pc = pc
        # mutation probability
        self.pm = pm
        self.pm_node = pm_node
        self.pm_connect = pm_connect
        self.pm_weight = pm_weight


