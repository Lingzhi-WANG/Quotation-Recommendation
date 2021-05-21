import torch
import math
import numpy as np
from torch import nn
from nltk.translate.bleu_score import sentence_bleu

def find(listt1, ele):
    for idx, i in enumerate(listt1):
        if i == ele:
            return idx
    return -1

def MAP(true_idiomid_list, rec_idiomid_2dlist):
    num_all = len(true_idiomid_list)
    s = 0.0
    for idx, listt in enumerate(rec_idiomid_2dlist):
        rank = find(listt, true_idiomid_list[idx]) + 1
        if rank != 0:
            s += 1.0 / rank
    return s / num_all

def nDCG_k(k, true_idiomid_list, rec_idiomid_2dlist):
    num_all = len(true_idiomid_list)
    s = 0.0
    for idx, listt in enumerate(rec_idiomid_2dlist):
        rank = find(listt, true_idiomid_list[idx]) + 1
        if rank != 0:
            if rank <= k:
                s += 1.0 / (math.log(rank + 1, 2))
    return s/num_all


def P_k(k, true_idiomid_list, rec_idiomid_2dlist):
    num_all = len(true_idiomid_list)
    p = 0.0
    for idx, listt in enumerate(rec_idiomid_2dlist):
        rank = find(listt, true_idiomid_list[idx]) + 1
        if rank != 0:
            if rank <= k:
                p += 1
    return p / num_all

def BLEU(true_idiomid_list, rec_idiomid_2dlist):
    true_idiomid_list = [' '.join([j for j in i]) for i in true_idiomid_list]
    listt = []
    for i in rec_idiomid_2dlist:
        listt.append([' '.join(k for k in j) for j in i])
    score = []
    for t, r in zip(true_idiomid_list, listt):
        tt = t.split(' ')
        rr = r[0].split(' ')
        ss = sentence_bleu([tt], rr, [0.25, 0.25, 0.25, 0.25])
        score.append(ss)
    ss = 0
    for i in score:
        ss += i
    return ss / len(score)

def ROUGE(true_idiomid_list, rec_idiomid_2dlist):
    true_idiomid_list = [' '.join(j for j in i) for i in true_idiomid_list]
    listt = []
    for i in rec_idiomid_2dlist:
        listt.append([' '.join(k for k in j) for j in i])
    score = []
    for t, r in zip(true_idiomid_list, listt):
        rouge = Rouge()
        rouge_score = rouge.get_scores(r[0], t)
        score.append([rouge_score[0]["rouge-1"], rouge_score[0]["rouge-l"]])
    s1 = 0
    s2 = 0
    for i in score:
        s1 += i[0]['f']
        s2 += i[1]['f']
    return s1 / len(score), s2 / len(score)

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(torch.clamp(output, min=1e-10, max=1))) + \
            weights[0] * ((1 - target) * torch.log(torch.clamp(1 - output, min=1e-10, max=1)))
    else:
        loss = target * torch.log(torch.clamp(output, min=1e-10, max=1)) + \
               (1 - target) * torch.log(torch.clamp(1 - output, min=1e-10, max=1))

    return torch.neg(torch.mean(loss))


def valid_evaluate(model, valid_loader, config, padded_quotes, quote_lens, padded_exp, exp_lens):  # validation, report the valid auc, loss and threshold
    model.eval()
    true_labels = []
    pred_labels = [[0 for i in range(config.quote_len)] for j in range(config.batch_size)]
    avg_loss = 0.0
    for batch_idx, batch in enumerate(valid_loader):
        convs, conv_lens, conv_turn_lens, labels = batch[0], batch[1], batch[2], batch[3]
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            convs = convs.cuda()
        predictions = model(convs, conv_lens, conv_turn_lens, padded_quotes, quote_lens, padded_exp, exp_lens)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = nn.Softmax()(predictions)
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            predictions = predictions.cpu()
        avg_loss += nn.CrossEntropyLoss()(predictions, labels.long()).item()

        true_labels = np.concatenate([true_labels, labels.data.numpy()])
        pred_labels = np.concatenate((pred_labels, predictions.data.numpy()), axis=0)

    pred_labels = pred_labels[config.batch_size:]
    pred_labels = [np.argsort(-i)for i in pred_labels]

    avg_loss /= len(valid_loader)
    if len(true_labels) == 0:
        print("wrong in valid evaluation")
        map = 0
    else:
        map = MAP(true_labels, pred_labels)
    return map, avg_loss


def test_evaluate(model, test_loader, config, padded_quotes, quote_lens, padded_exp, exp_lens):  # evaluation, report the auc, f1, pre, rec, acc
    model.eval()
    true_labels = []
    pred_labels = [[0 for i in range(config.quote_len)] for j in range(config.batch_size)]
    for batch_idx, batch in enumerate(test_loader):
        #print(batch)
        convs, conv_lens, conv_turn_lens, labels = batch[0], batch[1], batch[2], batch[3]
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            convs = convs.cuda()
        predictions = model(convs, conv_lens, conv_turn_lens, padded_quotes, quote_lens, padded_exp, exp_lens)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = nn.Softmax()(predictions)
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            predictions = predictions.cpu()

        true_labels = np.concatenate([true_labels, labels.data.numpy()])
        pred_labels = np.concatenate((pred_labels, predictions.data.numpy()), axis=0)
    pred_labels = pred_labels[config.batch_size:]
    pred_labels = [np.argsort(-i) for i in pred_labels]

    if len(true_labels) == 0:
        print("wrong in test evaluation")
        return 0,0,0
    else:
        map = MAP(true_labels, pred_labels)
        p1 = P_k(1, true_labels, pred_labels)
        p3 = P_k(3, true_labels, pred_labels)
        ndcg5 = nDCG_k(5, true_labels, pred_labels)
        ndcg10 = nDCG_k(10, true_labels, pred_labels)
    return map, p1, p3, ndcg5, ndcg10


