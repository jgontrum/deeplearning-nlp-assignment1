from blocks.extensions import SimpleExtension
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from copy import deepcopy
import matplotlib.pyplot as plt
import os


class PlotWeights(SimpleExtension):
    DEFAULT_PARAMETERS = {
        "computation_graph": None,
        "x_label": "Output Layer",
        "y_label": "Input Layer",
        "folder": ".",
        "folder_per_layer": False,
        "fileprefix": "",
        "dpi": 50
    }

    def __init__(self, **kwargs):
        super_kwargs = {}
        self.parameters = deepcopy(self.DEFAULT_PARAMETERS)
        self.iteration = 0
        for key, value in kwargs.items():
            if key in self.DEFAULT_PARAMETERS:
                self.parameters[key] = value
            else:
                super_kwargs[key] = value
        if not self.parameters["computation_graph"]:
            raise ValueError("Please specify at least the computation_graph.")
        super(PlotWeights, self).__init__(**super_kwargs)

    def do(self, which_callback, *args):
        plt.clf()
        cg = self.parameters['computation_graph']
        self.iteration += 1

        if not os.path.exists(self.parameters['folder'] + "/"):
            os.makedirs(self.parameters['folder'] + "/")

        for i, w in enumerate(VariableFilter(roles=[WEIGHT])(cg.variables)):
            fig = plt.figure()
            base_filename = self.parameters['folder'] + "/"

            # Create folder for layer if it does not exist
            if self.parameters['folder_per_layer']:
                base_filename = base_filename + str(i) + "/"
                if not os.path.exists(base_filename):
                    os.makedirs(base_filename)

            title = "Weights from Layer {0} to {1}. Iteration: {2}".format(
                    i + 1, i + 2, self.iteration)

            filename = "{0}{1}{2}_{3}.png".format(base_filename,
                                                  self.parameters['fileprefix'],
                                                  i,
                                                  self.iteration)

            fig.add_subplot(111).imshow(w.get_value())
            fig.add_subplot(111).set_xlabel(self.parameters['x_label'])
            fig.add_subplot(111).set_ylabel(self.parameters['y_label'])
            fig.add_subplot(111).set_title(title)
            fig.savefig(filename, dpi=self.parameters['dpi'])
            plt.close(fig)
