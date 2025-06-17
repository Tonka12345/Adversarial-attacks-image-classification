from cifar100_strategy import CIFAR100_startegy
import torch

def main():
    strategy = CIFAR100_startegy()
    model = strategy.load_model()
    loader = strategy.get_test_loader()
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images = strategy.prepare_input(images)
            outputs = model(images)
            probs, predicted = strategy.get_predictions(outputs)
            loss = strategy.calculate_loss(outputs, labels)
            total_loss += loss
            if predicted.item() == labels.item():
                correct += 1
            total += 1

    print(f"avg loss: {total_loss/total}")
    print(f"accuracy: {correct}/{total} = {correct/total}")

if __name__ == "__main__":
    main()