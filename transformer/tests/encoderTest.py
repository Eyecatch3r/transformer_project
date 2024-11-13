import torch
import pytest

from modelling.functional import BaseTransformerLayer


def test_encoder_attention_mask():
    # Define parameters for the encoder and test data
    input_dim = 4  # Dimension of input features
    num_heads = 2  # Number of attention heads
    feature_dim = 8  # Dimension of feed-forward network
    seq_len = 3  # Length of the sequence
    batch_size = 2  # Batch size

    # Create an instance of the encoder layer with dropout disabled for testing
    encoder_layer = BaseTransformerLayer(input_dim=input_dim, num_heads=num_heads, feature_dim=feature_dim, dropout=0.0)

    # Define a test input tensor and an attention mask
    x = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
        [[-1.0, -2.0, -3.0, -4.0],
         [-5.0, -6.0, -7.0, -8.0],
         [0.0, 0.0, 0.0, 0.0]]
    ], dtype=torch.float32)  # Shape: [batch_size, seq_len, input_dim]

    # Mask to ignore the last position in the second batch
    attention_mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0]
    ])  # Shape: [batch_size, seq_len]

    # Pass the input through the encoder layer
    output = encoder_layer(x, attention_mask)

    # Apply the mask to the output to zero out masked positions
    masked_output = output * attention_mask.unsqueeze(-1)

    # Print the raw output before masking for debugging
    print("Raw Output from Encoder:\n", output)

    # Check if masked positions are zero
    for batch in range(batch_size):
        for pos in range(seq_len):
            if attention_mask[batch, pos] == 0:
                assert torch.allclose(output[batch, pos], torch.zeros_like(output[batch, pos])), \
                    f"Masked position in batch {batch}, position {pos} is not zeroed out."

    # For additional debugging, print masked output comparison
    print("Output after applying mask:\n", masked_output)


# Run the test function
test_encoder_attention_mask()
