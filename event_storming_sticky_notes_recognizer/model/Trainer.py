import numpy as np
import torch
from torch.autograd import Variable

from event_storming_sticky_notes_recognizer.Name import Name
from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder


class Trainer:
    @staticmethod
    def train(args, criterion, model, train_loader, optimizer, epoch) -> list:
        for p in model.parameters():
            p.requires_grad = True
        model.train()
        losses = []
        # for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx, sample in enumerate(train_loader):
            optimizer.zero_grad()
            data, targets, target_lens = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[
                Name.LABEL_LEN.value]
            log_probs = model(data)
            preds_size = Variable(torch.tensor([log_probs.size(0)] * log_probs.shape[1], dtype=torch.int32))
            targets = concat_targets(targets=targets, target_lengths=target_lens)
            loss = criterion(log_probs=log_probs,
                             targets=targets,
                             input_lengths=preds_size,
                             target_lengths=target_lens)
            losses.append(loss.item())
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
        return losses

    @staticmethod
    def test(criterion, model, test_loader) -> (float, float):
        model.eval()
        test_loss = 0
        correct = 0
        encoder_decoder = LabelEncoderDecoder(alphabet='russian')
        with torch.no_grad():
            for sample in test_loader:
                data, targets, target_lens = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[
                    Name.LABEL_LEN.value]
                log_probs = model(data)
                preds_size = Variable(torch.tensor([log_probs.size(0)] * log_probs.shape[1], dtype=torch.int32))
                targets = concat_targets(targets=targets, target_lengths=target_lens)
                test_loss += criterion(log_probs=log_probs,
                                       targets=targets,
                                       input_lengths=preds_size,
                                       target_lengths=target_lens).item()  # sum up batch loss

                _, probs = log_probs.max(2)

                probs = probs.transpose(1, 0)
                preds = []
                for prob in probs:
                    preds.append(encoder_decoder.from_raw_to_label(prob.numpy()))
                preds = np.asarray(preds)

                for pred, target in zip(preds, targets.numpy()):
                    if np.array_equal(pred, target):
                        correct += 1

        test_loss /= (len(test_loader.dataset) / test_loader.batch_size)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss,
                     correct,
                     len(test_loader.dataset),
                     100. * correct / len(test_loader.dataset)
                     )
              )
        return test_loss, 100. * correct / len(test_loader.dataset)


def concat_targets(targets, target_lengths):
    result = []
    for target, target_len in zip(targets, target_lengths):
        nonzero_taget = target[:target_len]
        result.append(nonzero_taget)
    result = np.hstack(result)
    return torch.tensor(result, dtype=torch.int32)
