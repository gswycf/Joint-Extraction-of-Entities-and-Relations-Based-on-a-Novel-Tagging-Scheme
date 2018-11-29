import os
import pickle
import json


def triplet_stats(fin):
    valid = 0
    total = 0
    for line in fin:
        line = line.strip()
        if not line:
            continue
        sentence = json.loads(line)
        total += len(sentence["relationMentions"])
        for relationMention in sentence["relationMentions"]:
            if relationMention["label"] != "None":
                valid += 1
    return valid, total


def sentence_length_stats():
    length_dict = {}
    with open('data/NYT_CoType/corpus.txt', 'rt', encoding='utf-8') as fin:
        for line in fin:
            sentence = line.strip().split()
            length = len(sentence)
            if length not in length_dict:
                length_dict[length] = 1
            else:
                length_dict[length] = length_dict[length] + 1
    with open('data/NYT_CoType/sentence_length_stats.pk', 'wb') as f:
        pickle.dump(length_dict, f)


def token_length_stats():
    length_dict = {}
    with open('data/NYT_CoType/corpus.txt', 'rt', encoding='utf-8') as fin:
        for line in fin:
            sentence = line.strip().split()
            for word in sentence:
                length = len(word)
                if length not in length_dict:
                    length_dict[length] = 1
                else:
                    length_dict[length] = length_dict[length] + 1
    with open('data/NYT_CoType/token_length_stats.pk', 'wb') as f:
        pickle.dump(length_dict, f)


def show_length(length_dict, groups):
    max_length = max(length_dict.keys())
    total = sum(length_dict.values())
    length_list = list(length_dict.items())
    length_list.sort(key=lambda tup: tup[0])
    length_groups = [0] * len(groups)
    for length_tuple in length_list:
        for index, group in enumerate(groups):
            if length_tuple[0] <= group:
                length_groups[index] = length_groups[index] + length_tuple[1]
                break

    print("-" * 36)
    print("| (  0, {:3d}] | {:8d} | {:7.3f}% |".format(groups[0], length_groups[0], length_groups[0]/total*100))
    for i in range(1, len(length_groups)):
        print("| ({:3d}, {:3d}] | {:8d} | {:7.3f}% |"
              .format(groups[i-1], groups[i], length_groups[i], length_groups[i]/total*100))
    print("-" * 36)
    print("|    Total   | {:8d} | {:7.3f}% |"
          .format(sum(length_groups), sum(length_groups) / total * 100))
    print("-" * 36)
    print("Max Length: {:d}".format(max_length))


if __name__ == "__main__":
    if not os.path.exists('data/NYT_CoType/sentence_length_stats.pk'):
        sentence_length_stats()
    with open('data/NYT_CoType/sentence_length_stats.pk', 'rb') as f:
        length_dict = pickle.load(f)
    groups = list(range(10, 110, 10))
    show_length(length_dict, groups)

    if not os.path.exists('data/NYT_CoType/token_length_stats.pk'):
        token_length_stats()
    with open('data/NYT_CoType/token_length_stats.pk', 'rb') as f:
        length_dict = pickle.load(f)
    groups = list(range(5, 25, 3))
    show_length(length_dict, groups)

    print()
    with open('data/NYT_CoType/train.json', 'rt', encoding='utf-8') as fin:
        valid, total = triplet_stats(fin)
        print("Train\n\tValid Triplets: {}\n\tTotal Triplets: {}".format(valid, total))
    with open('data/NYT_CoType/test.json', 'rt', encoding='utf-8') as fin:
        valid, total = triplet_stats(fin)
        print("Test\n\tValid Triplets: {}\n\tTotal Triplets: {}".format(valid, total))

# ------------------------------------
# | (  0,  10] |     2251 |   0.952% |
# | ( 10,  20] |    22997 |   9.729% |
# | ( 20,  30] |    55208 |  23.356% |
# | ( 30,  40] |    65138 |  27.557% |
# | ( 40,  50] |    48907 |  20.690% |
# | ( 50,  60] |    24943 |  10.552% |
# | ( 60,  70] |    10123 |   4.283% |
# | ( 70,  80] |     3848 |   1.628% |
# | ( 80,  90] |     1619 |   0.685% |
# | ( 90, 100] |      664 |   0.281% |
# ------------------------------------
# |    Total   |   235698 |  99.713% |
# ------------------------------------
# Max Length: 9621
# ------------------------------------
# | (  0,   5] |  6119176 |  68.385% |
# | (  5,   8] |  2008222 |  22.443% |
# | (  8,  11] |   682932 |   7.632% |
# | ( 11,  14] |   120471 |   1.346% |
# | ( 14,  17] |    12470 |   0.139% |
# | ( 17,  20] |     3492 |   0.039% |
# | ( 20,  23] |      807 |   0.009% |
# ------------------------------------
# |    Total   |  8947570 |  99.994% |
# ------------------------------------
# Max Length: 107
#
# Train
# 	Valid Triplets: 111610
# 	Total Triplets: 372853
# Test
# 	Valid Triplets: 410
# 	Total Triplets: 3880
