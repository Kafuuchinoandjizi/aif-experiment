import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

import src.model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = torchvision.transforms.Compose([
    transforms.Resize((150, 150)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = torchvision.transforms.Compose([
    transforms.Resize((150, 150)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set_path = r"D:\10439\文档\Pycharm projects\aif实验\dataset\train"
val_set_path = r"D:\10439\文档\Pycharm projects\aif实验\dataset\val"
train_set = ImageFolder(train_set_path, transform_train)
val_set = ImageFolder(val_set_path, transform_test)

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)


def main():
    extrain_epoch = 0
    model = src.model.Model()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    writer = SummaryWriter(log_dir='.././logs')

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU上

            # 零化梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 后向传播和优化
            loss.backward()
            optimizer.step()

            # 累积损失
            running_loss += loss.item()

            # 计算准确度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 300 == 299:  # 每100个小批量输出一次损失
                print(
                    f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}, accuracy: {100 * correct / total:.2f}%')
                writer.add_scalar('training loss', running_loss / 100, epoch * len(train_dataloader) + i)
                writer.add_scalar('training accuracy', 100 * correct / total, epoch * len(train_dataloader) + i)
                running_loss = 0.0
                correct = 0
                total = 0

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU上

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 记录验证损失和准确度
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1} validation loss: {avg_val_loss:.3f}, validation accuracy: {val_accuracy:.2f}%')
        writer.add_scalar('validation loss', avg_val_loss, epoch)
        writer.add_scalar('validation accuracy', val_accuracy, epoch)

    print('Finished Training')

    torch.save(model, f'.././models/model_{extrain_epoch + 50}.pth')

    writer.close()


if __name__ == '__main__':
    main()
