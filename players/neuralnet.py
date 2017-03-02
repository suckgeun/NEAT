class NeuralNetwork:
    """
    Neural Network class
    """

    def __init__(self):
        self.node_index = None
        self.connect_genes = None
        self.fitness = None
        self._outputs_front = None
        self._outputs_back = None
        self._is_output_front = False

    @property
    def outputs_prev(self):
        if self._is_output_front:
            return self._outputs_front
        else:
            return self._outputs_back

    @outputs_prev.setter
    def outputs_prev(self, val):
        assert type(val) is list, "val must be list"

        if self._is_output_front:
            self._outputs_front = val
        else:
            self._outputs_back = val

    @property
    def outputs_cur(self):
        if self._is_output_front:
            return self._outputs_back
        else:
            return self._outputs_front

    def flip_outputs_list(self):
        self._is_output_front = not self._is_output_front

    @outputs_cur.setter
    def outputs_cur(self, val):
        assert type(val) is list, "val must be list"

        if self._is_output_front:
            self._outputs_back = val
        else:
            self._outputs_front = val











