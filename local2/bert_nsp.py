import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForNextSentencePrediction, BertTokenizer, AdamW
import multiprocessing
import numpy as np
import os
import string


class CustomDataset(Dataset):
    def __init__(self, path):
        self.data = None
        with open(path, 'r') as f:
            self.data = f.readlines()
            self.data = [x.translate(str.maketrans('', '', string.punctuation)) for x in self.data]
            self.lengths = [len(y.split()) for y in self.data]
            self.substr_len = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.lengths[index]
        try:
            neg_x = self.data[index + 1]
            neg_y = self.lengths[index + 1]
        except:
            neg_x = self.data[index - 1]
            neg_y = self.lengths[index - 1]

        rand_idx_pos = np.random.randint(0, y - self.substr_len - 1)
        rand_idx_neg = np.random.randint(0, neg_y - self.substr_len - 1)

        return x, \
               " ".join(x.split()[rand_idx_pos:rand_idx_pos + self.substr_len]), \
               " ".join(neg_x.split()[rand_idx_neg:rand_idx_neg + self.substr_len])

        # return x, y, neg_x, neg_y


# def custom_collate(batch):
#     search_audio_lengths = [len(x[0]) for x in batch]
#     max_audio_len = np.max(search_audio_lengths) + 1
#     query_length = 100  # keeping fixed for now
#     query_start_idx = np.random.randint(max_audio_len - query_length)
#     # is it good to pad and then extract query? if the difference between lengths is large, padding will be really bad,
#     # also should we try edge padding instead of constant padding
#     search = np.array([_pad_2d(x[0], max_audio_len) for x in batch], dtype=np.float)
#     search_batch = torch.FloatTensor(search)
#     pos_query = search_batch[:, query_start_idx: query_start_idx + query_length]
#
#     neg_audio_lengths = [len(x[1]) for x in batch]
#     max_neg_audio_len = np.max(neg_audio_lengths) + 1
#     neg_query_start_idx = np.random.randint(max_neg_audio_len - query_length)
#     neg_query = np.array([_pad_2d(x[1], max_neg_audio_len) for x in batch], dtype=np.float)
#     neg_query_batch = torch.FloatTensor(neg_query)
#     neg_query = neg_query_batch[:, neg_query_start_idx: neg_query_start_idx+query_length]
#
#     return search_batch, pos_query, neg_query

path = "wikitext-2"
train_dataset = CustomDataset(os.path.join("train_.txt"))
valid_dataset = CustomDataset(os.path.join("valid_.txt"))
test_dataset = CustomDataset(os.path.join("test_.txt"))

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# load pretrained model and a pretrained tokenizer
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.to(device)

batch_size = 8
num_workers = 8 if cuda else multiprocessing.cpu_count()
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

print("len(trainloader): ", len(train_dataloader), "batch_size: ", batch_size,
      "len(train_dataloader)//batch_size - 1: ", len(train_dataloader) // batch_size - 1)


# true_batch_size = 16

def train(model, train_loader, valid_loader, test_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch {}:".format(epoch))
        model.train()
        num_correct = 0
        running_loss = 0.
        num_total = 0
        for batch_num, d in enumerate(train_loader):
            pos, pos_sub, neg_sub = d[0], d[1], d[2]
            labels = torch.zeros(pos.shape[0] * 2)
            labels[pos.shape[0]:] = 1
            labels.to(device)
            optimizer.zero_grad()
            pairofstrings = list(zip(pos, pos_sub))
            pairofstrings.extend(list(zip(pos, neg_sub)))
            encoded_batch = tokenizer.batch_encode_plus(pairofstrings, add_special_tokens=True, return_tensors='pt',
                                                        return_special_tokens_masks=True, max_length=512,
                                                        pad_to_max_length=True)
            attention_mask = (encoded_batch['attention_mask'] - encoded_batch['special_tokens_mask']).to(device)
            input_ids, token_type_ids = encoded_batch['input_ids'].to(device), encoded_batch['token_type_ids'].to(
                device)
            loss, logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 next_sentence_label=labels)
            # print(logits)
            predicted = torch.max(logits, 1)[1]
            num_total += labels.size(0)
            # print("predicted:", predicted)
            # print("labels:", labels)
            num_correct += (predicted == labels).sum().item()

            # loss = criterion(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            # todo check if the batch_num == len(train_dataloader) - 1 constraint works
            # if batch_num % true_batch_size == 0 or batch_num == len(train_dataloader) // true_batch_size - 1:
            #     optimizer.step()
            optimizer.step()
            if batch_num % 100 == 0 or batch_num == len(train_dataloader) // batch_size - 1:
                print("acc : ", (num_correct) / num_total, "batch_num:", batch_num)
                torch.save(model.state_dict(), 'model.npy')
                torch.save(optimizer.state_dict(), 'optimizer.npy')

        print('Train Accuracy: {}'.format(num_correct / num_total),
              'Average Train Loss: {}'.format(running_loss / len(train_loader)))

        if epoch % 1 == 0:
            ep_num = epoch + 1
            torch.save(model.state_dict(), 'model' + str(ep_num) + '.npy')
            torch.save(optimizer.state_dict(), 'optimizer' + str(ep_num) + '.npy')

        model.eval()
        num_correct = 0
        running_loss = 0.
        num_total = 0
        with torch.no_grad():
            for batch_num, d in enumerate(valid_loader):
                pos, pos_sub, neg_sub = d[0], d[1], d[2]
                labels = torch.zeros(pos.shape[0] * 2)
                labels[pos.shape[0]:] = 1
                labels.to(device)
                pairofstrings = list(zip(pos, pos_sub))
                pairofstrings.extend(list(zip(pos, neg_sub)))
                encoded_batch = tokenizer.batch_encode_plus(pairofstrings, add_special_tokens=True, return_tensors='pt',
                                                            max_length=512, return_special_tokens_masks=True,
                                                            pad_to_max_length=True)
                attention_mask = (encoded_batch['attention_mask'] - encoded_batch['special_tokens_mask']).to(device)
                input_ids, token_type_ids = encoded_batch['input_ids'].to(device), encoded_batch['token_type_ids'].to(
                    device)
                loss, logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                     next_sentence_label=labels)
                predicted = torch.max(logits, 1)[1]
                # print("labels:")
                num_total += labels.size(0)
                num_correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                print("predicted, labels:", predicted.cpu().detach().numpy(), labels.cpu().detach().numpy())

        print('Validation Accuracy: {}'.format(num_correct / num_total),
              'Average Validation Loss: {}'.format(running_loss / len(valid_loader)))

        num_correct = 0
        running_loss = 0.
        num_total = 0
        with torch.no_grad():
            for batch_num, d in enumerate(test_loader):
                pos, pos_sub, neg_sub = d[0], d[1], d[2]
                labels = torch.zeros(pos.shape[0] * 2)
                labels[pos.shape[0]:] = 1
                labels.to(device)
                pairofstrings = list(zip(pos, pos_sub))
                pairofstrings.extend(list(zip(pos, neg_sub)))
                encoded_batch = tokenizer.batch_encode_plus(pairofstrings, add_special_tokens=True, return_tensors='pt',
                                                            max_length=512, return_special_tokens_masks=True,
                                                            pad_to_max_length=True)
                attention_mask = (encoded_batch['attention_mask'] - encoded_batch['special_tokens_mask']).to(device)
                input_ids, token_type_ids = encoded_batch['input_ids'].to(device), encoded_batch['token_type_ids'].to(
                    device)
                loss, logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                     next_sentence_label=labels)
                predicted = torch.max(logits, 1)[1]
                # print("labels:")
                num_total += labels.size(0)
                num_correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                print("predicted, labels:", predicted.cpu().detach().numpy(), labels.cpu().detach().numpy())

        print('Test Accuracy: {}'.format(num_correct / num_total),
              'Average Test Loss: {}'.format(running_loss / len(valid_loader)))


lr = 3e-5
# optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = AdamW(model.parameters(), lr=lr)
num_epochs = 25

train(model, train_dataloader, valid_dataloader, test_dataloader, optimizer, num_epochs)

torch.save(model.state_dict(), 'model_final.npy')
torch.save(optimizer.state_dict(), 'optimizer_final.npy')
