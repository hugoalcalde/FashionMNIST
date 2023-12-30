
import torch

from model import SimpleCNN
from torch import optim
import matplotlib.pyplot as plt 

from data import load

import click

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=0.03, help="learning rate to use for training")
def train(lr) : 

    # Instantiate the model
    model = SimpleCNN()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    folder = "dtu_mlops/data/corruptmnist/"
    train_loader, _ = load(folder)
    # Training loop
    num_epochs = 5  
    loss_list = []
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0

        for inputs, labels in train_loader :
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        loss_list.append(average_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')
    plt.plot(loss_list)
    plt.savefig("training.png")
    torch.save(model.state_dict(), 'checkpoint.pth')


@click.command()
@click.argument("model_checkpoint", default = "checkpoint.pth")
def evaluate(model_checkpoint) : 
    
    folder = "dtu_mlops/data/corruptmnist/"
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    _, test_loader = load(folder)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

