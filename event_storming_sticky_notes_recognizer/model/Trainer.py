import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from event_storming_sticky_notes_recognizer.Name import Name
from event_storming_sticky_notes_recognizer.dataset.LabelEncoderDecoder import LabelEncoderDecoder


class Trainer:
    @staticmethod
    def train(args, model, train_loader, optimizer, epoch) -> list:
        model.train()
        losses = []
        # for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx, sample in enumerate(train_loader):
            optimizer.zero_grad()
            data, target, target_lens = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[Name.LABEL_LEN.value]
            log_probs = model(data)
            preds_size = Variable(torch.IntTensor([log_probs.size(0)] * log_probs.shape[1]))
            loss = F.ctc_loss(log_probs=log_probs,
                              targets=target,
                              input_lengths=preds_size,
                              target_lengths=target_lens,
                              reduction='mean',
                              zero_infinity=True)
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
    def test(model, test_loader) -> (float, float):
        model.eval()
        test_loss = 0
        correct = 0
        encoder_decoder = LabelEncoderDecoder()
        with torch.no_grad():
            for sample in test_loader:
                data, targets, target_lens = sample[Name.IMAGE.value], sample[Name.LABEL.value], sample[
                    Name.LABEL_LEN.value]
                log_probs = model(data)
                preds_size = Variable(torch.IntTensor([log_probs.size(0)] * log_probs.shape[1]))
                test_loss += F.ctc_loss(log_probs=log_probs,
                                        targets=targets,
                                        input_lengths=preds_size,
                                        target_lengths=target_lens,
                                        reduction='mean',
                                        zero_infinity=True).item()  # sum up batch loss

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
