import argparse
from utils import *
from torch.utils.data.dataset import *
from torch.utils.data.sampler import *
from torch.nn.utils.rnn import *
import bisect
from model import *
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


parser = argparse.ArgumentParser(description="Joint Extraction of Entities and Relations")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.5)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit (default: 30)')
parser.add_argument('--char_kernel_size', type=int, default=3,
                    help='character-level kernel size (default: 3)')
parser.add_argument('--word_kernel_size', type=int, default=3,
                    help='word-level kernel size (default: 3)')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of character embeddings (default: 50)')
parser.add_argument('--char_layers', type=int, default=3,
                    help='# of character-level convolution layers (default: 3)')
parser.add_argument('--word_layers', type=int, default=3,
                    help='# of word-level convolution layers (default: 3)')
parser.add_argument('--char_nhid', type=int, default=50,
                    help='number of hidden units per character-level convolution layer (default: 50)')
parser.add_argument('--word_nhid', type=int, default=300,
                    help='number of hidden units per word-level convolution layer (default: 300)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--weight', type=float, default=10.0,
                    help='manual rescaling weight given to each tag except "O"')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
device = torch.device("cuda" if args.cuda else "cpu")


charset = Charset()
vocab = Vocabulary()
vocab.load("data/NYT_CoType/vocab.txt")
tag_set = Index()
tag_set.load("data/NYT_CoType/tag2id.txt")
relation_labels = Index()
relation_labels.load('data/NYT_CoType/relation_labels.txt')

train_data = load('data/NYT_CoType/train.pk')
test_data = load('data/NYT_CoType/test.pk')
val_size = int(0.01 * len(train_data))
train_data, val_data = random_split(train_data, [len(train_data)-val_size, val_size])


#torch.Size([47463, 300])

def group(data, breakpoints):
    groups = [[] for _ in range(len(breakpoints)+1)]
    for idx, item in enumerate(data):
        i = bisect.bisect_left(breakpoints, len(item[0]))
        groups[i].append(idx)
    data_groups = [Subset(data, g) for g in groups]
    return data_groups


train_data_groups = group(train_data, [10, 20, 30, 40, 50, 60])
val_data_groups = group(val_data, [10, 20, 30, 40, 50, 60])
test_data_groups = group(test_data, [10, 20, 30, 40, 50, 60])


class GroupBatchRandomSampler(object):
    def __init__(self, data_groups, batch_size, drop_last):
        self.batch_indices = []
        for data_group in data_groups:
            self.batch_indices.extend(list(BatchSampler(SubsetRandomSampler(data_group.indices),
                                                        batch_size, drop_last=drop_last)))

    def __iter__(self):
        return (self.batch_indices[i] for i in torch.randperm(len(self.batch_indices)))

    def __len__(self):
        return len(self.batch_indices)


def get_batch(batch_indices, data):
    batch = [data[idx] for idx in batch_indices]
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    sentences, tokens, tags = zip(*sorted_batch)

    padded_sentences, lengths = pad_packed_sequence(pack_sequence([torch.LongTensor(_) for _ in sentences]),
                                                    batch_first=True, padding_value=vocab["<pad>"])
    padded_tokens, _ = pad_packed_sequence(pack_sequence([torch.LongTensor(_) for _ in tokens]),
                                           batch_first=True, padding_value=charset["<pad>"])
    padded_tags, _ = pad_packed_sequence(pack_sequence([torch.LongTensor(_) for _ in tags]),
                                         batch_first=True, padding_value=tag_set["O"])

    return padded_sentences.to(device), padded_tokens.to(device), padded_tags.to(device), lengths.to(device)


word_embeddings = torch.tensor(np.load("data/NYT_CoType/word2vec.vectors.npy"))
word_embedding_size = word_embeddings.size(1)
pad_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
unk_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
word_embeddings = torch.cat([pad_embedding, unk_embedding, word_embeddings])


char_channels = [args.emsize] + [args.char_nhid] * args.char_layers
word_channels = [word_embedding_size + args.char_nhid] + [args.word_nhid] * args.word_layers

if os.path.exists("model.pt"):
    model=torch.load('model.pt')
else:
    model = Model(charset_size=len(charset), char_embedding_size=args.emsize, char_channels=char_channels,
                  char_padding_idx=charset["<pad>"], char_kernel_size=args.char_kernel_size, weight=word_embeddings,
                  word_embedding_size=word_embedding_size, word_channels=word_channels,
                  word_kernel_size=args.word_kernel_size, num_tag=len(tag_set), dropout=args.dropout,
                  emb_dropout=args.emb_dropout).to(device)


weight = [args.weight] * len(tag_set)
weight[tag_set["O"]] = 1
weight = torch.tensor(weight).to(device)
criterion = nn.NLLLoss(weight, size_average=False)
optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)


def train():
    model.train()
    total_loss = 0
    count = 0
    sampler = GroupBatchRandomSampler(train_data_groups, args.batch_size, drop_last=False)
    for idx, batch_indices in enumerate(sampler):
        sentences, tokens, targets, lengths = get_batch(batch_indices, train_data)
        optimizer.zero_grad()
        output = model(sentences, tokens)
        output = pack_padded_sequence(output, lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, lengths, batch_first=True).data
        loss = criterion(output, targets)
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        count += len(targets)
        if (idx+1) % args.log_interval == 0:
            cur_loss = total_loss / count
            elapsed = time.time() - start_time
            percent = ((epoch-1)*len(sampler)+(idx+1))/(args.epochs*len(sampler))
            remaining = elapsed / percent - elapsed
            print("| Epoch {:2d}/{:2d} | Batch {:5d}/{:5d} | Elapsed Time {:s} | Remaining Time {:s} | "
                  "lr {:4.2e} | Loss {:5.3f} |".format(epoch, args.epochs, idx+1, len(sampler), time_display(elapsed),
                                                       time_display(remaining), lr, cur_loss))
            total_loss = 0
            count = 0


def evaluate(data_groups):
    model.eval()
    total_loss = 0
    count = 0
    TP = 0
    TP_FP = 0
    TP_FN = 0
    with torch.no_grad():
        for batch_indices in GroupBatchRandomSampler(data_groups, args.batch_size, drop_last=False):
            sentences, tokens, targets, lengths = get_batch(batch_indices, train_data)
            output = model(sentences, tokens)
            tp, tp_fp, tp_fn = measure(output, targets, lengths)
            TP += tp
            TP_FP += tp_fp
            TP_FN += tp_fn
            output = pack_padded_sequence(output, lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, lengths, batch_first=True).data
            loss = criterion(output, targets)
            total_loss += loss.item()
            count += len(targets)
    return total_loss / count, TP/TP_FP, TP/TP_FN, 2*TP/(TP_FP+TP_FN)


def measure(output, targets, lengths):
    assert output.size(0) == targets.size(0) and targets.size(0) == lengths.size(0)
    tp = 0
    tp_fp = 0
    tp_fn = 0
    batch_size = output.size(0)
    output = torch.argmax(output, dim=-1)
    for i in range(batch_size):
        length = lengths[i]
        out = output[i][:length].tolist()
        target = targets[i][:length].tolist()
        out_triplets = get_triplets(out)
        tp_fp += len(out_triplets)
        target_triplets = get_triplets(target)
        tp_fn += len(target_triplets)
        for target_triplet in target_triplets:
            for out_triplet in out_triplets:
                if out_triplet == target_triplet:
                    tp += 1
    return tp, tp_fp, tp_fn


def get_triplets(tags):
    temp = {}
    triplets = []
    for idx, tag in enumerate(tags):
        if tag == tag_set["O"]:
            continue
        pos, relation_label, role = tag_set[tag].split("-")
        if pos == "B" or pos == "S":
            if relation_label not in temp:
                temp[relation_label] = [[], []]
            temp[relation_label][int(role) - 1].append(idx)
    for relation_label in temp:
        role1, role2 = temp[relation_label]
        if role1 and role2:
            len1, len2 = len(role1), len(role2)
            if len1 > len2:
                for e2 in role2:
                    idx = np.argmin([abs(e2 - e1) for e1 in role1])
                    e1 = role1[idx]
                    triplets.append((e1, relation_label, e2))
                    del role1[idx]
            else:
                for e1 in role1:
                    idx = np.argmin([abs(e2 - e1) for e2 in role2])
                    e2 = role2[idx]
                    triplets.append((e1, relation_label, e2))
                    del role2[idx]
    return triplets


if __name__ == "__main__":
    best_val_loss = None
    lr = args.lr
    all_val_loss = []
    all_precision = []
    all_recall = []
    all_f1 = []

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        start_time = time.time()
        print("-" * 118)
        for epoch in range(1, args.epochs+1):
            train()
            val_loss, precision, recall, f1 = evaluate(val_data_groups)

            elapsed = time.time() - start_time
            print("-" * 118)
            print("| End of Epoch {:2d} | Elapsed Time {:s} | Validation Loss {:5.3f} | Precision {:5.3f} "
                  "| Recall {:5.3f} | F1 {:5.3f} |".format(epoch, time_display(elapsed),
                                                           val_loss, precision, recall, f1))
            print("-" * 118)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr = lr / 4.0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_val_loss.append(val_loss)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

    except KeyboardInterrupt:
        print('-' * 118)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data
    test_loss, precision, recall, f1 = evaluate(test_data_groups)
    print("=" * 118)
    print("| End of Training | Test Loss {:5.3f} | Precision {:5.3f} "
          "| Recall {:5.3f} | F1 {:5.3f} |".format(test_loss, precision, recall, f1))
    print("=" * 118)

    with open("record.tsv", "wt", encoding="utf-8") as f:
        for idx in range(len(all_val_loss)):
            f.write("{:d}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n"
                    .format(idx+1, all_val_loss[idx], all_precision[idx], all_recall[idx], all_f1[idx]))
        f.write("\n{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".format(test_loss, precision, recall, f1))

'''
charset_size: 96
char_embedding_size: 50
char_channels: [50, 50, 50, 50]
char_padding_idx: 94
char_kernel_size: 3
weight: tensor([[-0.4441,  0.4617,  0.1288,  ...,  0.4624,  0.2199, -0.1290],
        [-0.4024,  0.2595,  0.4271,  ...,  0.3322, -0.1303,  0.0604],
        [ 0.2019, -0.1130, -0.1495,  ..., -0.2575,  0.0146, -0.1554],
        ...,
        [ 0.1529,  0.1099, -0.1428,  ..., -0.1544,  0.1506, -0.0258],
        [ 0.1294,  0.0606, -0.0225,  ..., -0.2985,  0.0162, -0.1880],
        [ 0.0975, -0.0968, -0.2453,  ..., -0.2132,  0.1037, -0.2321]])
word_embedding_size: 300
word_channels: [350, 300, 300, 300]
word_kernel_size: 3
num_tag: 193
dropout: (0.5,)
emb_dropout: 0.25

'''