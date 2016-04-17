from theano import tensor
from blocks.bricks import MLP, Linear, Logistic, Softmax, Tanh
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
from blocks.model import Model
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import FinishAfter, Printing

shrooms_train, shrooms_test = get_mushroom_data()

x = tensor.matrix('features')
input_to_hidden = Linear(name='input_to_hidden', input_dim=117, output_dim=50)
h = Logistic().apply(input_to_hidden.apply(x))
hidden_to_output = Linear(name='hidden_to_output', input_dim=50, output_dim=2)
y_hat = Softmax().apply(hidden_to_output.apply(h))

y = tensor.lmatrix('targets')
cost = CategoricalCrossEntropy().apply(y, y_hat)

cg = ComputationGraph(cost)
W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
cost.name = 'cost_with_regularization'

input_to_hidden.weights_init = hidden_to_output.weights_init = IsotropicGaussian(
    0.01)
input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_output.initialize()

data_stream = Flatten(DataStream.default_stream(
    shrooms_train,
    iteration_scheme=SequentialScheme(shrooms_train.num_examples, batch_size=200)))

algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                            step_rule=Scale(learning_rate=0.1))

main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                     extensions=[FinishAfter(after_n_epochs=10), ])

main_loop.run()
