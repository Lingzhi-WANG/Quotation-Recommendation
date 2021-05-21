import os
import sys
import random
import torch
import argparse
import numpy as np
from data_process import Corpus, create_embedding_matrix
from train import train
from predict import predict
from mymodel import SSNP

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, choices=["test", "Weibo", "Reddit"])
    parser.add_argument("--cuda_dev", type=str, default="0")
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--history_size", type=int, default=50)
    parser.add_argument("--pre_lr", type=float, default=0.0001)
    parser.add_argument("--att_lr", type=float, default=0.00001)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no_pretrained_embedding", action="store_true")
    parser.add_argument("--train_weight", type=float, default=1)
    parser.add_argument("--pretrain_weight", type=float, default=None)
    parser.add_argument("--no_use_gpu", dest='use_gpu', action='store_false')
    parser.add_argument("--different_embed", dest='same_embed', action='store_false')
    parser.add_argument("--different_vocab", dest='same_vocab', action='store_false')
    parser.add_argument("--for_test", dest='for_train', action='store_false')
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--l2_weight", type=float, default=0.0003)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"])
    parser.add_argument("--turn_encoder", type=str, default='gru', choices=['gru', 'transformer'])
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=2)
    parser.add_argument("--length_max", type=int, default=200)
    parser.add_argument("--turn_length_max", type=int, default=20)
    parser.add_argument("--history_num", type=int, default=20)
    parser.add_argument("--quote_query_loss_weight", type=float, default=0.0)
    parser.add_argument("--hist_query_loss_weight", type=float, default=0.0)
    parser.add_argument("--conv_not_transform", action='store_true')
    parser.add_argument("--no_transform", action='store_true')
    parser.add_argument("--print_attention", action='store_true')
    parser.add_argument("--print_label", type=int, default=1)
    parser.add_argument("--quote_query_loss_func", type=str, default='mse', choices=['kldiv', 'mse'])
    parser.add_argument("--model_name", type=str, default="LSTM-LSTM", choices=["LSTM-LSTM", "ncf", "ncf_query", "ncf_query_exp"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="0")
    parser.add_argument("--Prob", type=float, default=0.25)
    parser.add_argument("--scheduler", type=str, default=None, choices=['OnValidAUC'])

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    config = parse_config()
    setup_seed(config.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_dev
    if config.filename in ["test", "Weibo", "Reddit"]:
        config.train_file, config.test_file, config.valid_file = 'data/' + config.filename + "_train.json", \
     config.filename + "_test.json", config.filename + "_valid.json"
    else:
        print('Data name not correct!')
        sys.exit()

    corp = Corpus(config)
    config.vocab_num = corp.wordNum
    config.quote_vocab_num = corp.quote_wordNum
    if config.mode != 'train' or not config.for_train or config.no_pretrained_embedding: # if local test, to same time no embedding
        config.embedding_matrix = None
        print('No pretain embedding! \n')
    else:
        config.embedding_matrix = create_embedding_matrix(config.filename, corp.r_wordIDs, corp.wordNum, config.embedding_dim)
        if config.same_embed == False:
            config.quote_embedding_matrix = create_embedding_matrix(config.filename, corp.quote_r_wordIDs, corp.quote_wordNum, config.embedding_dim)
    config.path = "Results/" + config.filename + "/"
    if not os.path.isdir(config.path):
        os.makedirs(config.path)
    model = SSNP(config)
    config.loss_weights = torch.Tensor([1, config.train_weight])
    if torch.cuda.is_available() and config.use_gpu and config.mode == 'train':  # run in GPU
        model = model.cuda()
        config.loss_weights = config.loss_weights.cuda()
    if config.mode == "train":
        train(corp, model, config)
    else:
        predict(corp, model, config)



