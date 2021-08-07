import unittest
import re
import tensorflow as tf

from pg import build_mlp
from pg import PG

class TestBuildMlp(unittest.TestCase):
    """
    Unit test case for pg.build_mlp function
    """

    @classmethod
    def setUpClass(cls):
        # Setting up parameter grid to test architecture builds
        output_sizes = range(1, 5)
        scope = ['test']
        n_layers_s = range(0, 4)
        sizes = range(1, 10)
        output_activations = ['BiasAdd', 'Relu']
        cls.act_fs = {'BiasAdd': None, 'Relu': tf.nn.relu}
        cls.test_cases = [
                (o, sc, nl, s, act) 
                for o in output_sizes 
                for sc in scope for nl in n_layers_s 
                for s in sizes for act in output_activations
                ]
        return None

    def check_scope(self, scope, tensors):
        """
        Checks that the variable scope of all tensors in :param tensors: is
        :param scope:

        :param scope: str, name of correct variable scope
        :param tensors: list(tf.Tensor)

        ---> None
        """
        for t in tensors:
            self.assertEqual(scope, t.name.split('/')[0])
        return None

    def group_by_layer(self, tensors):
        """
        Groups :param tensors: by layer

        :param tensors: list(tf.Tensor)

        ---> dict(str --> list(tf.Tensor)), mapping between layer names and its
             tensors
        """
        layers = {}
        for t in tensors:
            layer_name = t.name.split('/')[1]
            layer = layers.get(layer_name, [])
            layer.append(t)
            layers[layer_name] = layer
        return layers

    def check_layer(self, size, layer_tensors, output=None):
        """
        Checks the size and activation function of a layer

        :param size: int, correct layer size
        :param layer_tensors: list(tf.Tensors), list of tensors in layer
        :param output: str or None, optional name of activation function or
                       None

        ---> None
        """
        act_tensor = layer_tensors[-1] # Last tensor should be activation func
        # Checking size of layer
        self.assertEqual(int(act_tensor.shape[-1]), size)
        # Checking activation function
        if output:
            self.assertIn(output, act_tensor.name)
        else:
            self.assertIn('Relu', act_tensor.name)
        return None

    def check_graph(
            self, n_layers, scope, size, 
            output_size, output_act, graph
            ):
        """
        Checks structure of graph and each layer.
        Tests:
        1. scope of NN is :param scope:
        2. number of hidden layers created is :param n_layers:
        3. each hidden layer generates the same amount of tensors (identical)
        4. each layer has the correct activation function and size

        :param n_layers: int, correct number of dense layers to be created
        :param scope: str, name of variable scope network needs to be in
        :param size: int, size of dense hidden layers
        :param output_size: int, size of dense output layer
        :param output_act: str, name of output activation function
        :param graph: tf.Graph, tensorflow graph object

        ---> None
        """
        # Generating list of tensors created by build_mlp
        all_tensors = []
        for op in graph.get_operations():
            if op.type != 'Placeholder':
                for t in op.values():
                    all_tensors.append(t)
        # Checking if all tensors are in scope
        self.check_scope(scope, all_tensors)
        # Grouping tensors by layers
        layers = self.group_by_layer(all_tensors)
        # Checking number of layers
        correct_layers = [
                'dense_{}'.format(i) if i != 0 else 'dense'
                for i in range(n_layers+1)
                ]
        self.assertEqual(list(layers.keys()), correct_layers)
        # Checking if hidden layers have same number of tensors
        if n_layers != 0:
            hlayer_nums = {
                    n: len(layers[n]) 
                    for n in layers if n != 'dense_{}'.format(n_layers)
                    }
            for n in hlayer_nums:
                self.assertEqual(
                        hlayer_nums[n], 
                        next(iter(hlayer_nums.values()))
                        )
        # Checking each layer
        for ln in layers:
            if (n_layers != 0) and (ln != 'dense_{}'.format(n_layers)):
                self.check_layer(size, layers[ln]) 
            else:
                self.check_layer(output_size, layers[ln], output=output_act)

        return None
    
    def test_build_mlp(self):
        """
        Test function for pg.build_mlp function.

        ---> None
        """
        for case in self.test_cases:
            with self.subTest(case=case):
                # subTest Setup
                output_size, scope, n_layers, size, output_act = case
                mlp_input = tf.placeholder(tf.float32, (None, 20)) 
                output = build_mlp(
                        mlp_input, output_size, scope, 
                        n_layers, size, self.act_fs[output_act]
                        )
                # Checking structure of the generated graph
                self.check_graph(
                        n_layers, scope, size, 
                        output_size, output_act, tf.get_default_graph()
                        )
            # subTest Teardown
            tf.reset_default_graph()
        return None

class TestPG(unittest.TestCase):
    """
    Unit test case for pg.PG object
    """
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

if __name__ == '__main__':
    unittest.main()
