from players.activation import sigmoid

class Workplace:
    """
    innov_history is dictionary with key:(in, out), value:counter
    """

    def __init__(self, n_input, n_output, n_nn=1, bias=True, activ_func=sigmoid):

        assert type(bias) is bool, "bias flag should be boolean value"
        assert n_input > -1 and n_output > -1 and n_nn > 0, "number of inputs, outputs and neural network " \
                                                            "should be positive integer"

        self.n_input = n_input
        self.n_output = n_output
        self.innov_counter = -1
        self.innov_history = {}
        self.n_nn = n_nn
        self.nns = []
        self.activ_func = activ_func
        if bias:
            self.n_bias = 1
        else:
            self.n_bias = 0



