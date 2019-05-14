import torch
import torch.nn.functional as F


class Trainer:
    @staticmethod
    def train(args, model, train_loader, optimizer, epoch):
        model.train()
        # for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx, sample in enumerate(train_loader):
            optimizer.zero_grad()
            data = sample['image']
            print(data.shape)
            target = sample['label']
            output = model(data)
            loss = F.ctc_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                      format(epoch,
                             batch_idx * len(data), len(train_loader.dataset),
                             100. * batch_idx / len(train_loader),
                             loss.item()
                             )
                      )

    @staticmethod
    def test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for sample in test_loader:
                data = sample['image']
                target = sample['label']
                output = model(data)
                test_loss += F.ctc_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss,
                     correct,
                     len(test_loader.dataset),
                     100. * correct / len(test_loader.dataset)
                     )
              )
