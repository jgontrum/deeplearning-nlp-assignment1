from theano import tensor
from blocks.bricks import Linear, Logistic, Softmax, Tanh, Softplus, Rectifier
from blocks.initialization import IsotropicGaussian, Constant
from data_functions import get_mushroom_data
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
from blocks.algorithms import GradientDescent, Scale
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.main_loop import MainLoop
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks_extras.extensions.plot import Plot
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from PlotWeights import PlotWeights
import logging


def shroom_mlp(shrooms_train, shrooms_test, num_epochs, hidden_dims,
               activation_function):

    # These are theano variables
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')

    # Construct the graph
    input_to_hidden = Linear(name='input_to_hidden', input_dim=117,
                             output_dim=hidden_dims)
    h = activation_function.apply(input_to_hidden.apply(x))
    hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_dims,
                              output_dim=2)
    y_hat = Softmax().apply(hidden_to_output.apply(h))

    # And initialize with random varibales and set the bias vector to 0
    weights = IsotropicGaussian(0.01)
    input_to_hidden.weights_init = hidden_to_output.weights_init = weights
    input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
    input_to_hidden.initialize()
    hidden_to_output.initialize()

    # And now the cost function
    cost = CategoricalCrossEntropy().apply(y, y_hat)
    cg = ComputationGraph(cost)
    W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
    cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
    cost.name = 'cost_with_regularization'

    error_rate = MisclassificationRate().apply(y.argmax(axis=1), y_hat)

    # The data streams give us access to our corpus and allow us to perform a
    # mini-batch training.
    data_stream = Flatten(DataStream.default_stream(
        shrooms_train,
        iteration_scheme=SequentialScheme(shrooms_train.num_examples,
                                          batch_size=50)))

    test_data_stream = Flatten(DataStream.default_stream(
        shrooms_test,
        iteration_scheme=SequentialScheme(shrooms_test.num_examples,
                                          batch_size=1000)))

    extensions = [
        ProgressBar(),
        PlotWeights(after_epoch=True,
                    folder="results_logistic_40",
                    computation_graph=cg,
                    folder_per_layer=True,
                    dpi=150),
        FinishAfter(after_n_epochs=num_epochs),
        DataStreamMonitoring(variables=[cost, error_rate],
                             data_stream=test_data_stream, prefix="test"),
        Printing()
    ]

    # Now we tie up lose ends and construct the algorithm for the training
    # and define what happens in the main loop.
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                step_rule=Scale(learning_rate=0.1))

    main = MainLoop(data_stream=data_stream,
                    algorithm=algorithm,
                    extensions=extensions)

    main.run()

    # print(error_rate())

    return 0

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.CRITICAL)

    # Here are out corpora
    shrooms_train, shrooms_test = get_mushroom_data()

    shroom_mlp(shrooms_train,
               shrooms_test,
               10,
               40,
               Logistic())

    # epochs = range(1, 11)
    # layers = range(10, 100, 10)
    # activation_functions = [
    #     Logistic(),
    #     Tanh(),
    #     Linear(),
    #     Softplus(),
    #     Rectifier()
    # ]
    #
    # results = []
    #
    # for activation_function in activation_functions:
    #     for epoch in epochs:
    #         for layer in layers:
    #             results.append((shroom_mlp(shrooms_train,
    #                                        shrooms_test,
    #                                        epoch,
    #                                        layer,
    #                                        activation_function),
    #                            ({'epochs': epoch,
    #                              'layers': layer,
    #                              'function': activation_function
    #                              })))
    #
    # for result, configuration in sorted(results, key=lambda x: x[0]):
    #     print(result, configuration)
