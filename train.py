import torch
import time
import numpy as np
from torch import nn, optim
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
from data_process import MyDataset
from evaluate import valid_evaluate, test_evaluate


def train_epoch(model, train_data, loss_weights, optimizer, epoch, config, padded_quotes, quote_lens, padded_exp, exp_lens):
    start = time.time()
    model.train()
    print('Train Epoch: %d start!' % epoch)
    avg_loss = 0.0
    train_loader = data.DataLoader(train_data, collate_fn=train_data.my_collate, batch_size=config.batch_size, num_workers=0, shuffle=True)
    for batch_idx, batch in enumerate(train_loader):
        convs, conv_lens, conv_turn_lens, labels = batch[0], batch[1], batch[2], batch[3]
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            convs = convs.cuda()
            labels = labels.cuda()
            padded_quotes = padded_quotes.cuda()
        predictions = model(convs, conv_lens, conv_turn_lens, padded_quotes, quote_lens, padded_exp, exp_lens, labels=labels)
        if config.quote_query_loss_weight > 0 or config.hist_query_loss_weight > 0:
            preds, query_reps, quote_reps, hist_reps = predictions[0], predictions[1], predictions[2], predictions[3]
            loss = nn.CrossEntropyLoss()(preds, labels.long())
            if batch_idx == 0:
                loss_print_1 = loss.data
                loss_print_2 = 0
                loss_print_3 = 0
            if config.quote_query_loss_weight > 0:
                loss += config.quote_query_loss_weight * nn.MSELoss(reduction='sum')(query_reps, quote_reps) / config.batch_size
                if batch_idx == 0:
                    loss_print_2 = (nn.MSELoss(reduction='sum')(query_reps, quote_reps) / config.batch_size).data
            if batch_idx == 0:
                print('loss, quote_query, hist_query', loss_print_1, loss_print_2, loss_print_3)
        else:
            loss = nn.CrossEntropyLoss()(predictions, labels.long())
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    end = time.time()
    print('Train Epoch: %d done! Train avg_loss: %g! Using time: %.2f minutes!' % (epoch, avg_loss, (end - start) / 60))
    return avg_loss


def define_att_weight(model, train_data, config):
    model.eval()
    print('Begin to find objective attention weights!')
    train_loader = data.DataLoader(train_data, collate_fn=train_data.my_collate, batch_size=config.batch_size, num_workers=0)
    for batch_idx, batch in enumerate(train_loader):
        convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens, labels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            convs = convs.cuda()
            users = users.cuda()
            labels = labels.cuda()
        important_turn = torch.zeros_like(convs[:, :, 0]).float()
        predictions, _ = model(convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens)
        for t in range(max(conv_lens)-1):
            have_masked = []
            for c in range(len(convs)):
                if conv_lens[c] - 1 > t:
                    convs[c, t] = torch.ones_like(convs[c, t])
                    have_masked.append(c)
            new_predictions, _ = model(convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens)
            current_importance = torch.ge(torch.abs(new_predictions-labels), torch.abs(predictions-labels)).long()
            for c in have_masked:
                important_turn[c, t] = current_importance[c]
        for turn_weight in important_turn:
            if torch.sum(turn_weight) == 0:
                turn_weight = 1 - turn_weight
            turn_weight /= torch.sum(turn_weight)
            train_data.att_labels.append(turn_weight)
    train_data.att = True
    train_data.att_labels = rnn_utils.pad_sequence(train_data.att_labels, batch_first=True)
    print('Finish finding objective attention weights!')
    return train_data


def define_att_weight_idf(corp, config):
    print('Begin to find objective attention weights based on IDF!')
    global_idf = np.zeros(corp.wordNum)
    for w in range(corp.wordNum):
        global_idf[w] = float(corp.convNum) / (len(corp.global_word_record[w]) + 1)
    global_idf = np.log10(global_idf)
    for c in range(len(corp.convs)):
        local_idf = np.ones(corp.wordNum)
        current_turn_num = len(corp.convs[c])
        for w in corp.local_word_record[c]:
            local_idf[w] = float(current_turn_num) / (len(corp.local_word_record[c][w]) + 1)
        local_idf = np.log10(local_idf)
        current_att_weights = []
        for turn in corp.convs[c]:
            word_idf = [global_idf[w]+config.idf_tradeoff*local_idf[w] for w in turn[1]]
            current_att_weights.append(sum(word_idf) / np.log(len(turn[1])+1))
        current_att_weights = torch.Tensor(current_att_weights)
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            current_att_weights = current_att_weights.cuda()
        current_att_weights /= torch.sum(current_att_weights)
        corp.train_data.att_labels.append(current_att_weights)
    corp.train_data.att = True
    corp.train_data.att_labels = rnn_utils.pad_sequence(corp.train_data.att_labels, batch_first=True)
    print('Finish finding objective attention weights!')
    return corp.train_data


def train(corp, model, config):
    train_optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2_weight)
    # train_data: conv_vecs, conv_lens, conv_turn_lens, my_labels
    train_data = corp.train_data
    padded_quotes = corp.padded_quotes
    quote_lens = corp.quote_lens
    padded_exp = corp.padded_exp
    exp_lens = corp.exp_lens
    corp.test_corpus(config.valid_file, mode='VALID')
    valid_data = MyDataset(corp, config, 'VALID')
    valid_loader = data.DataLoader(valid_data, collate_fn=valid_data.my_collate, batch_size=config.batch_size,
                                   num_workers=0)
    best_state = None
    best_valid_thr = 0.0
    best_valid_map = -1.0
    best_valid_loss = 999999.99
    no_improve = 0

    for epoch in range(config.max_epoch):
        train_epoch(model, train_data, config.loss_weights, train_optimizer, epoch, config, padded_quotes, quote_lens, padded_exp, exp_lens)
        valid_map, valid_loss = valid_evaluate(model, valid_loader, config, padded_quotes, quote_lens, padded_exp, exp_lens)
        if best_valid_map < valid_map or best_valid_loss > valid_loss:
            no_improve = 0
            best_state = model.state_dict()
            if best_valid_map < valid_map:
                best_valid_map = valid_map
                print('New Best MAP Valid Result!!! Valid MAP: %g, Valid Loss: %g' % (valid_map, valid_loss))
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                print('New Best Loss Valid Result!!! Valid MAP: %g, Valid Loss: %g' % (valid_map, valid_loss))
        else:
            no_improve += 1
            print(
                        'No improve! Current Valid MAP: %g, Best Valid AUC: %g;  Current Valid Loss: %g, Best Valid Loss: %g' % (
                valid_map, best_valid_map, valid_loss, best_valid_loss))
        if no_improve == 5:
            break
    model.load_state_dict(best_state)


    # Final step: Evaluate the model
    corp.test_corpus(config.test_file, mode='TEST')
    test_data = MyDataset(corp, config, 'TEST')
    test_loader = data.DataLoader(test_data, collate_fn=test_data.my_collate, batch_size=config.batch_size, num_workers=0)
    res = test_evaluate(model, test_loader, config, padded_quotes, quote_lens, padded_exp, exp_lens)
    print('Result in test set: MAP %g, P@1 %g, P@3 %g, nDCG@5 %g,nDCG@10 %g' % (res[0], res[1], res[2], res[3], res[4]))
    torch.save(model.state_dict(), config.path + 'map%.4f_p@1%.4f_best_seed%d.model' % (res[0], res[1], config.random_seed))
    with open(config.path + 'map%.4f_p@1%.4f_best_seed%d.res' % (res[0], res[1], config.random_seed), 'w') as f:
        f.write('MAP %g \t P@1 %g \t P@3 %g \t nDCG@5 %g\t nDCG@10 %g\n'% (res[0], res[1], res[2], res[3], res[4]))
        f.write('\n\nParameters:\n')
        for key in config.__dict__:
            f.write('%s : %s\n' % (key, config.__dict__[key]))










