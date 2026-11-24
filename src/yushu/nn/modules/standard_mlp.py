"""A plain residual MLP stack."""

import torch
import torch.nn as nn

__all__ = ["ResidualBlock", "StandardMLP"]


class ResidualBlock(nn.Module):
    def __init__(self, size,*, dropout):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(size, size)
        self.norm1=nn.LayerNorm(size)
        self.linear2 = nn.Linear(size, size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.activation(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        #out = self.activation(out)
        out = self.dropout(out)
        return out+residual

class StandardMLP(nn.Module):
    def __init__(self, *,input_size, hidden_size, output_size,num_layers, dropout):
        assert num_layers >= 2 and num_layers %2==0
        super().__init__()
        #self.activation = nn.GELU()
        layers=nn.ModuleList()
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range((num_layers-2)//2):
            layers.append(ResidualBlock(hidden_size, dropout=dropout))
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers=layers
    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return x



if __name__ == "__main__":
    model=StandardMLP(
        input_size=10,
        hidden_size=20,
        output_size=5,
        num_layers=4,
        dropout=0.1
    )
    with torch.no_grad():
        model.eval()
        x=torch.randn(2,10)
        output=model(x)
        print(output)