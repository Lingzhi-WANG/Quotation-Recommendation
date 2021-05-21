# encoding=utf-8
import random
import numpy as np
import torch
import torch.utils.data as data
from itertools import chain
import codecs
import json
import collections
import jieba


class MyDataset(data.Dataset):
    def __init__(self, corp, config, mode='TRAIN'):
        self.data_convs = []
        self.data_labels = []
        self.data_quotes = []
        self.mode = mode
        self.print_attention = config.print_attention
        if config.print_attention:
            self.history = corp.history
            self.history_num = config.history_num
        if mode == 'TRAIN':
            convs = corp.convs
            labels = corp.labels

        elif mode == 'TEST':
            convs = corp.test_convs
            labels = corp.test_labels

        else:
            convs = corp.valid_convs
            labels = corp.valid_labels
        for cid in convs:
            self.data_labels.append(labels[cid])
            self.data_convs.append([turn for turn in convs[cid]])

    def __getitem__(self, idx):
        return self.data_convs[idx], self.data_labels[idx]

    def __len__(self):
        return len(self.data_labels)

    def pad_vector(self, texts, text_size, sent_len):  # Pad with 0s to fixed size
        text_vec = []
        text_len = []
        turn_len = []
        for one_text in texts:
            t = []
            tl = []
            for sent in one_text:
                pad_len = max(0, sent_len - len(sent))
                t.append(sent + [0] * pad_len)
                tl.append(len(sent))
            pad_size = max(0, text_size - len(t))
            text_len.append(len(t))
            t.extend([[0] * sent_len] * pad_size)
            tl.extend([0] * pad_size)
            text_vec.append(t)
            turn_len.append(tl)
        padded_vec = torch.LongTensor(text_vec)
        return padded_vec, text_len, turn_len


    def my_collate(self, batch):
        conv_vecs = [item[0] for item in batch]
        my_labels = [item[1] for item in batch]
        #quote_vecs = [item[2] for item in batch]
        conv_turn_size = max([len(c) for c in conv_vecs])
        conv_turn_len = max([len(sent) for sent in chain.from_iterable([c for c in conv_vecs])])
        conv_vecs, conv_lens, conv_turn_lens = self.pad_vector(conv_vecs, conv_turn_size, conv_turn_len)
        if self.print_attention:
            #print(self.history)
            # hist_vecs = [self.history[l][:self.history_num] for l in my_labels]  # if self.history[l] else []
            hist_vecs = [self.history[l] for l in my_labels]
            hist_size = max([len(h) for h in hist_vecs])
            hist_turn_len = max([len(sent) for sent in chain.from_iterable([h for h in hist_vecs])])
            hist_vecs, hist_lens, hist_turn_lens = self.pad_vector(hist_vecs, hist_size, hist_turn_len)

        my_labels = torch.Tensor(my_labels)
        #padded_quotes = torch.Tensor(padded_quotes)
        if self.print_attention:
            return conv_vecs, conv_lens, conv_turn_lens, my_labels, hist_vecs, hist_lens, hist_turn_lens
        return conv_vecs, conv_lens, conv_turn_lens, my_labels

class Corpus:
    def __init__(self, config):
        self.turnNum = 0            # Number of messages
        self.convNum = 0            # Number of conversations
        self.userNum = 0            # Number of users
        self.userIDs = {}           # Dictionary that maps users to integer IDs
        self.r_userIDs = {}         # Inverse of last dictionary
        self.wordNum = 3            # Number of words
        self.wordIDs = {'<Pad>': 0, '<UNK>': 1, '<CLS>': 2}           # Dictionary that maps words to integers
        self.r_wordIDs = {0: '<Pad>', 1: '<UNK>', 2: '<CLS>'}         # Inverse of last dictionary
        self.wordCount = collections.Counter()
        self.quote_wordNum = 3
        self.quote_wordIDs = {'<Pad>': 0, '<UNK>': 1, '<CLS>': 2}
        self.quote_r_wordIDs = {0: '<Pad>', 1: '<UNK>', 2: '<CLS>'}
        self.quote_wordCount = collections.Counter()

        self.use_transformer = (config.turn_encoder == 'transformer')

        # Each conv is a list of turns, each turn is [userID, [w1, w2, w3, ...]]
        self.convs = collections.defaultdict(list)
        self.explanation = dict()
        self.example = {}
        self.labels = {}
        self.quotes = []
        self.quotes_exp = []
        # Each history is a list of query turns, each turn is [w1, w2, w3, ...]
        self.history = collections.defaultdict(list)
        self.test_convs = collections.defaultdict(list)
        self.test_labels = {}
        self.test_quotes = {}
        self.valid_convs = collections.defaultdict(list)
        self.valid_labels = {}
        self.valid_quotes = {}
        self.turn_length_max = config.turn_length_max
        self.length_max = config.length_max
        f1 = codecs.open(config.filename + '_quote_3dicts.json', 'r', 'utf-8')
        self.quote_dic, self.exp_dic, self.exa_dic = json.load(f1)
        self.r_quote_dic = {}
        for k in self.quote_dic:
            self.r_quote_dic[self.quote_dic[k]] = k

        config.quote_len = len(self.quote_dic)

        with codecs.open(config.train_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            #line = [[turn1, turn2, ...], quote, [quote_expanation, quote_example]]
            for line in lines:
                msgs = json.loads(line)
                #current_turn_num = 0
                # msgs[0] = [turn1, turn2, ...]
                for turn in msgs[0][-config.turn_length_max:]:
                    if config.turn_encoder == 'transformer':
                        words = [2]
                    else:
                        words = []
                    for word in turn.split(' '):
                        # self.wordIDs:  {word: id}
                        if word not in self.wordIDs:
                            self.wordIDs[word] = self.wordNum
                            self.r_wordIDs[self.wordNum] = word
                            self.wordNum += 1
                        self.wordCount[self.wordIDs[word]] += 1
                        words.append(self.wordIDs[word])
                        words = words[:config.length_max]

                    self.convs[self.convNum].append(words)

                    #self.users[self.userIDs[user_id]].append([self.convNum, words])
                    self.turnNum += 1
                if 10 <= len(self.convs[self.convNum][-1]) <= 20:
                    self.history[self.quote_dic[msgs[1]]].append(self.convs[self.convNum][-1])

                self.labels[self.convNum] = self.quote_dic[msgs[1]]
                self.convNum += 1

            for quote in self.quote_dic:
                if config.turn_encoder == 'transformer':
                    words = [2]
                else:
                    words = []
                if config.filename == 'Weibo':
                    list1 = list(quote)
                    #print(list1)
                elif config.filename == 'Reddit' or config.filename == 'test':
                    list1 = quote.split(' ')
                    #print(list1)
                else:
                    print('Wrong')
                    exit()
                list1_exp = jieba.lcut(self.exp_dic[str(self.quote_dic[quote])])
                #list2 = list1.exten(list1_exp)
                for word in list1:
                    if not config.same_vocab:
                        #print('same_vocab')
                        if word not in self.quote_wordIDs:
                            self.quote_wordIDs[word] = self.quote_wordNum
                            self.quote_r_wordIDs[self.quote_wordNum] = word
                            self.quote_wordNum += 1
                        self.quote_wordCount[self.quote_wordIDs[word]] += 1
                        words.append(self.quote_wordIDs[word])
                    else:
                        if word not in self.wordIDs:
                            self.wordIDs[word] = self.wordNum
                            self.r_wordIDs[self.wordNum] = word
                            self.wordNum += 1
                        self.wordCount[self.wordIDs[word]] += 1
                        words.append(self.wordIDs[word])
                self.quotes.append(words)

                if config.turn_encoder == 'transformer':
                    words = [2]
                else:
                    words = []
                for word in list1_exp:
                    if not config.same_vocab:
                        #print('same_vocab')
                        if word not in self.quote_wordIDs:
                            self.quote_wordIDs[word] = self.quote_wordNum
                            self.quote_r_wordIDs[self.quote_wordNum] = word
                            self.quote_wordNum += 1
                        self.quote_wordCount[self.quote_wordIDs[word]] += 1
                        words.append(self.quote_wordIDs[word])
                    else:
                        if word not in self.wordIDs:
                            self.wordIDs[word] = self.wordNum
                            self.r_wordIDs[self.wordNum] = word
                            self.wordNum += 1
                        self.wordCount[self.wordIDs[word]] += 1
                        words.append(self.wordIDs[word])
                self.quotes_exp.append(words)

            max_quote_len = max([len(q) for q in self.quotes])
            max_exp_len = max([len(q) for q in self.quotes_exp])
            self.padded_quotes = []
            self.quote_lens = []
            self.padded_exp = []
            self.exp_lens = []

            for sent in self.quotes:
                pad_len = max(0, max_quote_len - len(sent))
                self.padded_quotes.append(sent+[0]*pad_len)
                self.quote_lens.append(len(sent))
            self.padded_quotes = torch.LongTensor(self.padded_quotes)

            for sent in self.quotes_exp:
                pad_len = max(0, max_exp_len - len(sent))
                self.padded_exp.append(sent + [0] * pad_len)
                self.exp_lens.append(len(sent))
            self.padded_exp = torch.LongTensor(self.padded_exp)

        self.train_data = MyDataset(self, config, 'TRAIN')
        print("Corpus initialization over! QuoteNum: %d ConvNum: %d TurnNum: %d" % (len(self.quote_lens), self.convNum, self.turnNum))

    def test_corpus(self, test_file, mode='TEST'):  # mode == 'TEST' or mode == 'VALID'
        with codecs.open(test_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                msgs = json.loads(line)
                for turn in msgs[0][-self.turn_length_max:]:
                    if self.use_transformer:
                        words = [2]
                    else:
                        words = []
                    for word in turn.split(' '):
                        try:
                            words.append(self.wordIDs[word])
                        except KeyError:  # for the words that is out of vocabulary
                            words.append(self.wordIDs['<UNK>'])
                            # if word not in self.oovIDs:
                            #     self.oovIDs[word] = len(self.oovIDs)
                            #     self.r_oovIDs[self.oovIDs[word]] = word
                    if len(words) == 0:  # in case some turns are null turn without words
                        words.append(self.wordIDs['<UNK>'])
                    words = words[:self.length_max]
                    if mode == 'TEST':
                        self.test_convs[self.convNum].append(words)
                    else:
                        self.valid_convs[self.convNum].append(words)
                # words = []
                # for word in msgs[1].split():
                #     try:
                #         words.append(self.wordIDs[word])
                #     except KeyError:
                #         words.append(self.wordIDs['<UNK>'])
                # if mode == 'TEST':
                #     self.test_quotes[self.convNum] = words
                # else:
                #     self.valid_quotes[self.convNum] = words
                #     # current_turn_num += 1
                # self.quotes[self.convNum] = words
                if mode == 'TEST':
                    self.test_labels[self.convNum] = self.quote_dic[msgs[1]]
                    if 10 <= len(self.test_convs[self.convNum][-1]) <= 20:
                        self.history[self.quote_dic[msgs[1]]].append(self.test_convs[self.convNum][-1])
                else:
                    self.valid_labels[self.convNum] = self.quote_dic[msgs[1]]
                    if 10 <= len(self.valid_convs[self.convNum][-1]) <= 20:
                        self.history[self.quote_dic[msgs[1]]].append(self.valid_convs[self.convNum][-1])
                self.convNum += 1
        print("%s Corpus process over!" % mode)


def create_embedding_matrix(dataname, word_idx, word_num, embedding_dim=200):
    pretrain_file = 'Tencent_AILab_ChineseEmbedding.txt' if dataname[0] == 'W' else 'glove.6B.200d.txt'
    pretrain_words = {}
    with open(pretrain_file, 'r') as f:
        for line in f:
            infos = line.split()
            wd = infos[0]
            vec = np.array(infos[1:]).astype(np.float)
            pretrain_words[wd] = vec
    weights_matrix = np.zeros((word_num, embedding_dim))
    for idx in word_idx.keys():
        if idx == 0:
            continue
        try:
            weights_matrix[idx] = pretrain_words[word_idx[idx]]
        except KeyError:
            weights_matrix[idx] = np.random.normal(size=(embedding_dim,))
    if torch.cuda.is_available():  # run in GPU
        return torch.Tensor(weights_matrix).cuda()
    else:
        return torch.Tensor(weights_matrix)


def pretrain_corpus_construction(batch, config):
    convs, conv_lens, conv_turn_lens = batch[0], batch[1], batch[2]
    users, user_lens, user_turn_lens = batch[3], batch[4], batch[5]
    labels = batch[-1]
    if config.pretrain_type == "RR":
        need_replace = torch.rand(len(labels))
        need_replace = need_replace.le(config.Prob)
        for i in range(len(labels)):
            if need_replace[i]:
                if user_lens[i] == 0:
                    turn_len = conv_turn_lens[i][conv_lens[i]-1]
                    convs[i, conv_lens[i]-1, :turn_len] = convs[i, conv_lens[i]-1, torch.randperm(turn_len)]
                else:
                    replace_idx = random.randint(0, user_lens[i]-1)
                    replace_turn_len = min(max(convs.size(-1), conv_turn_lens[i][conv_lens[i]-1]), user_turn_lens[i][replace_idx])
                    convs[i, conv_lens[i]-1, :replace_turn_len] = users[i, replace_idx, :replace_turn_len]
                    conv_turn_lens[i][conv_lens[i] - 1] = replace_turn_len
                labels[i] = 1
            else:
                labels[i] = 0
    elif config.pretrain_type == "REPLACE":
        conv_num = len(convs)
        turn_num = max(conv_lens)
        labels = torch.zeros((conv_num, turn_num))
        for c in range(conv_num):
            for t in range(turn_num):
                if t >= conv_lens[c]:
                    break
                if random.random() <= config.Prob:
                    labels[c, t] = 1
                    rc = random.choice([i for i in range(conv_num) if i != c])
                    rt = random.choice([i for i in range(conv_lens[rc])])
                    convs[c, t, :] = convs[rc, rt, :]
                    conv_turn_lens[c][t] = conv_turn_lens[rc][rt]
    elif config.pretrain_type == "SWITCH":
        conv_num = len(convs)
        turn_num = max(conv_lens)
        labels = torch.zeros((conv_num, turn_num))
        for c in range(conv_num):
            need_switch = [random.random() <= config.Prob for i in range(conv_lens[c])]
            if sum(need_switch) <= 1:
                switch_idx = random.sample(list(range(conv_lens[c])), 2)
            else:
                switch_idx = [i for i in range(conv_lens[c]) if need_switch[i]]
            original_idx = list(switch_idx)
            random.shuffle(switch_idx)
            for i, idx in enumerate(switch_idx):
                if idx == original_idx[i]:
                    switch_idx[i], switch_idx[(i+1) % len(switch_idx)] = switch_idx[(i+1) % len(switch_idx)], switch_idx[i]
            convs[c, original_idx, :] = convs[c, switch_idx, :]
            new_turn_len = [l for l in conv_turn_lens[c]]
            for i in range(len(original_idx)):
                new_turn_len[original_idx[i]] = conv_turn_lens[c][switch_idx[i]]
            conv_turn_lens[c] = new_turn_len
            labels[c, original_idx] = 1
    else:
        print('Wrong Pretrain Type!')
        exit(0)
    return convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens, labels





