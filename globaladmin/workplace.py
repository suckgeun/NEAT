from players.activation import sigmoid


class Workplace:
    """
    innov_history is dictionary with key:(in, out), value:counter
    """

    def __init__(self, n_input, n_output, n_nn=1, bias=None, activ_func=sigmoid):

        assert n_input > -1 and n_output > -1 and n_nn > 0, "number of inputs, outputs and neural network " \
                                                            "should be positive integer"

        self.n_input = n_input
        self.n_output = n_output
        self.innov_counter = -1
        self.innov_history = {}
        self.n_nn = n_nn
        self.nns = []
        self.activ_func = activ_func
        self.node_genes = []
        self.inputs = None
        if bias is not None:
            self.n_bias = 1
            self.bias = bias
        else:
            self.n_bias = 0
            self.bias = None
        self.is_initialized = False
        self.fitnesses = None



