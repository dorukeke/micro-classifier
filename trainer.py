import torch
from torch import nn, optim


class CustomTrainer:
    def __init__(self, net: nn.Module, device):
        super().__init__()
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=0.001)
        self.loss_algorithm = nn.CrossEntropyLoss()
        self.device = device

    def train_start(self):
        self.net.to(self.device)
        self.net.train()

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        # Pass to GPU if available.
        x, y = x.to(self.device), y.to(self.device)

        # Zero out the gradients of the optimizer
        self.optimizer.zero_grad()

        # Get the outputs of your model and compute your loss
        outputs = self.net(x)
        loss = self.loss_algorithm(outputs, y)

        # Compute the loss gradient using the backward method and have the optimizer take a step
        loss.backward()
        self.optimizer.step()

    def train_end(self):
        self.net.eval()
        print("Finished training")

    def train(self, number_of_epochs, training_input_batches: [torch.Tensor], training_output_batches: [torch.Tensor]):
        self.train_start()
        for epoch in range(number_of_epochs):
            for idx, data in enumerate(training_input_batches):
                print(f"Epoch {epoch}")
                self.train_step(data, training_output_batches[idx])
        self.train_end()
