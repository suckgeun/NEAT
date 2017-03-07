class NeuralNetwork:
    """
    Neural Network class
    """

    def __init__(self):
        self.node_indices = None
        self.connect_genes = None
        self.fitness = None
        self.results = None
        self._is_front = False
        self._result_col1 = 5
        self._result_col2 = 6

    @property
    def result_col_prev(self):
        if self._is_front:
            return self._result_col1
        else:
            return self._result_col2

    @property
    def result_col_cur(self):
        if self._is_front:
            return self._result_col2
        else:
            return self._result_col1

    def flip(self):
        self._is_front = not self._is_front











