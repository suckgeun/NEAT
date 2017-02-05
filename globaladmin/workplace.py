from players.activation import sigmoid

class Workplace:
    """
    innov_history is dictionary with key:(in, out), value:counter
    """

    def __init__(self, n_input, n_output, n_nn=1, bias=1, activ_func=sigmoid):
        self.n_input = n_input
        self.n_output = n_output
        self.innov_counter = -1
        self.innov_history = {}
        self.n_nn = n_nn
        self.nns = []
        self.bias = bias
        self.activ_func = activ_func



