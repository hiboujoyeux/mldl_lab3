import torch
from models.custom_net import CustomNet
from data.dataloader import get_dataloaders
from data.preprocess import preprocess_tiny_imagenet
import wandb

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        wandb.log({"loss": loss.item()})

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    wandb.log({"val_loss": val_loss, "val_acc": val_accuracy})
    return val_accuracy

def main():
    wandb.init(project='gpt3')

    config = wandb.config
    config.learning_rate = 0.01
    
    preprocess_tiny_imagenet()
    train_loader, val_loader = get_dataloaders()
    model = CustomNet().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    best_acc = 0
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader, criterion)
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f'Model saved with accuracy: {best_acc:.2f}%')
    
    print(f'Best validation accuracy: {best_acc:.2f}%')
    wandb.log({"best_val_acc": best_acc})

    

if __name__ == '__main__':
    main()
