# Your models
import torch
import torch.nn as nn
import torch.nn.init as init


class MultiheadAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.MultiheadAttention(input_dim, num_heads, batch_first=True) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.apply(self.init_weights)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        # Create a mask to ignore the padding
        if mask is None:
            mask = torch.all(x == 0, dim=-1)

        x = self.norm(x)
        for idx, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            x = x + layer(x, x, x, key_padding_mask=mask)[0]
            x = norm(x)
            if idx < self.num_layers - 1:
                x = self.act(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
