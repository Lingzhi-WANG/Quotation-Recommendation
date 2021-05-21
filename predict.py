import torch
import collections
import torch.utils.data as data
from data_process import MyDataset


def print_attention(corp, model, data_loader, config):
    model.eval()
    for batch_idx, batch in enumerate(data_loader):
        convs, conv_lens, conv_turn_lens, labels, hists, hist_lens, hist_turn_lens = batch[0], batch[1], batch[2], \
                                                                                     batch[3], batch[4], batch[5], \
                                                                                     batch[6]
        get_label = False
        for idx, label in enumerate(labels):
            if label == config.print_label:
                get_label = True
                break
        if not get_label:
            continue
        padded_quotes = corp.padded_quotes
        quote_lens = corp.quote_lens
        padded_exp = corp.padded_exp
        exp_lens = corp.exp_lens
        predictions, hist_word_atts, hist_query_atts = model(convs, conv_lens, conv_turn_lens, padded_quotes, quote_lens, padded_exp, exp_lens, hists, hist_lens, hist_turn_lens, labels)
        word_att = hist_word_atts[idx]
        hist_att = hist_query_atts[idx]
        with open('attention_res_'+config.filename+str(config.print_label)+'.txt', 'w') as f:
            print('Quotation:', corp.r_quote_dic[config.print_label])
            f.write('Quotation: ' + corp.r_quote_dic[config.print_label] + '\n')
            hist_weight_dict = collections.Counter()
            word_weight_dict = collections.Counter()
            hist_word_weight_dict = collections.defaultdict(list)
            for h in range(hist_lens[idx]):
                hist = hists[idx][h][:hist_turn_lens[idx][h]]
                words = [corp.r_wordIDs[int(i)] for i in hist[1:]]
                hist_weight_dict[' '.join(words)] = hist_att[h]
                hist_word_weight_dict[' '.join(words)] = word_att[h][1:hist_turn_lens[idx][h]]
                for i, word in enumerate(words):
                    word_weight_dict[word] += word_att[h][i+1]
                    # word_weight_dict[word] += word_att[h][i+1] * hist_att[h]
            print('Attention weights for each history:')
            f.write('Attention weights for each history:\n')
            print([float(hist_weight_dict[h]) for h in sorted(hist_weight_dict.keys(), key=lambda x: hist_weight_dict[x], reverse=True)])
            f.write(str([float(hist_weight_dict[h]) for h in sorted(hist_weight_dict.keys(), key=lambda x: hist_weight_dict[x], reverse=True)])+'\n')
            for i, h in enumerate(sorted(hist_weight_dict.keys(), key=lambda x: hist_weight_dict[x], reverse=True)):
                print(h+':', hist_word_weight_dict[h].tolist())
                f.write(h+': '+str(hist_word_weight_dict[h].tolist())+'\n')
            print('Attention weights for each word:')
            f.write('Attention weights for each word:\n')
            for w in sorted(word_weight_dict.keys(), key=lambda x: word_weight_dict[x], reverse=True)[:100]:
                print(w+':', float(word_weight_dict[w]))
                f.write(w+': '+str(float(word_weight_dict[w]))+'\n')

        break


def predict(corp, model, config):
    # print attention weights
    assert config.print_attention
    if config.model_path is not None:
        print('Loading model from:', config.model_path)
        model.load_state_dict(torch.load(config.model_path, map_location='cpu'))
    print('Begin prediction!')
    corp.test_corpus(config.valid_file, mode='VALID')
    corp.test_corpus(config.test_file, mode='TEST')
    train_data = MyDataset(corp, config, 'TRAIN')
    train_loader = data.DataLoader(train_data, collate_fn=corp.train_data.my_collate, batch_size=config.batch_size, num_workers=0)
    print_attention(corp, model, train_loader, config)


