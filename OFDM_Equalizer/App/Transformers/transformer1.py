import torch
import torch.nn as nn

# Define the number of input and output dimensions
input_dim = 5
output_dim = 5

# Define the number of attention heads and the size of the hidden layers
num_heads = 2
hidden_size = 10

# Create the encoder layers
encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

# Create the decoder layers
decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

# Create the output layer
output_layer = nn.Linear(hidden_size, output_dim)

# Create the model
model = nn.Sequential(encoder, decoder, output_layer)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Generate some synthetic data
num_samples = 64
sequence_length = 8
x = torch.randint(input_dim, (num_samples, sequence_length))
y = torch.randint(output_dim, (num_samples, sequence_length))
