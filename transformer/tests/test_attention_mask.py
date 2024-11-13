import unittest
import numpy as np

from modelling.attention import Attention


class TestAttentionMask(unittest.TestCase):
    def test_attention_mask(self):
        # Define input tensors for the attention function
        q = np.array([[1.0, 0.0], [0.0, 1.0]])  # [2, 2] query matrix
        k = np.array([[1.0, 0.0], [0.0, 1.0]])  # [2, 2] key matrix
        v = np.array([[1.0, 2.0], [3.0, 4.0]])  # [2, 2] value matrix

        # Define a mask that blocks the second key
        mask = np.array([[0.0, -1e10], [0.0, 0.0]])  # [2, 2] mask with a large negative value

        # Perform attention
        output = Attention(mask)

        # Expected behavior: first row should ignore second key, second row should consider both keys
        expected_output = np.array([[1.0, 2.0], [2.0, 3.0]])

        # Check if the attention output matches the expected output
        np.testing.assert_almost_equal(output, expected_output, decimal=5)
