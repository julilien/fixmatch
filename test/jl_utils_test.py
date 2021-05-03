import tensorflow as tf
import numpy as np

from jl_utils import calculate_ece


class ECETest(tf.test.TestCase):
    def setUp(self):
        super(ECETest, self).setUp()

    def test_metric(self):
        preds = np.array([[1., 0., 0.], [0.1, 0.8, 0.1], [0.4, 0.3, 0.3]])
        y_test = np.array([0, 1, 2])
        n_bins = 4

        result = calculate_ece(preds, y_test, n_bins)
        expected_result = (1 / preds.shape[0]) * ((2 * np.abs(1 - 0.9)) + (1 * np.abs(0 - 0.4)))

        self.assertEqual(expected_result, result)
