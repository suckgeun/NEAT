class NeuralNetwork:
    """
    Neural Network class
    """

    def __init__(self):
        self.node_genes = None
        self.connect_genes = None
        self.fitness = None
        self._outputs1 = None
        self._outputs2 = None
        self._front = False

    @property
    def outputs_prev(self):
        if self._front:
            return self._outputs1
        else:
            return self._outputs2

    @outputs_prev.setter
    def outputs_prev(self, val):
        assert type(val) is list, "val must be list"

        if self._front:
            self._outputs1 = val
        else:
            self._outputs2 = val

    @property
    def outputs_cur(self):
        if self._front:
            return self._outputs2
        else:
            return self._outputs1

    def toggle(self):
        self._front = not self._front

    @outputs_cur.setter
    def outputs_cur(self, val):
        assert type(val) is list, "val must be list"

        if self._front:
            self._outputs2 = val
        else:
            self._outputs1 = val











