import torch

input_nodes = 1
hidden_nodes = 100
output_nodes = 1


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = torch.nn.Linear(hidden_nodes, output_nodes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
net = SimpleNet()