import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from printable_transformer import TransformerEncoder, TransformerEncoderLayer


class TurnEncoder(nn.Module):
    def __init__(self, config, input_dim, hidden_dim):
        super(TurnEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gpu = config.use_gpu
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, dropout=config.dropout, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, sentences, sentence_lengths, initial_vectors=None):
        """sentences = [batch_size, sentence_length, embedding_dim], sentence_lengths = [batch_size]
           is_target = True / False, initial_vectors[0] = [batch_size, embedding_dim]"""
        sorted_sentence_lengths, indices = torch.sort(sentence_lengths, descending=True)
        sorted_sentences = sentences[indices]
        _, desorted_indices = torch.sort(indices, descending=False)
        packed_sentences = rnn_utils.pack_padded_sequence(sorted_sentences, sorted_sentence_lengths, batch_first=True)
        if initial_vectors is not None:
            initial_vectors[0] = initial_vectors[0][indices]
            initial_vectors[1] = initial_vectors[1][indices]
            _, output = self.gru(packed_sentences, initial_vectors)
        else:
            _, output = self.gru(packed_sentences)
        output = torch.cat([output[-1], output[-2]], dim=-1)[desorted_indices]
        output = self.dropout(output)
        return output


class SSNP(nn.Module):
    def __init__(self, config):
        super(SSNP, self).__init__()
        self.word_embedding = nn.Embedding(config.vocab_num, config.embedding_dim, padding_idx=0)
        self.quote_embedding = nn.Embedding(config.quote_vocab_num, config.embedding_dim, padding_idx=0)
        if config.embedding_matrix is not None:
            self.word_embedding.load_state_dict({'weight': config.embedding_matrix})
            if config.same_embed == False:
                self.quote_embedding.load_state_dict({'weight': config.quote_embedding_matrix})
        self.same_embed = config.same_embed
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.batch_size = config.batch_size
        self.model_name = config.model_name
        self.use_gpu = config.use_gpu
        self.dropout = nn.Dropout(config.dropout)
        self.quote_len = config.quote_len
        self.conv_not_transform = config.conv_not_transform
        self.no_transform = config.no_transform
        self.print_attention = config.print_attention
        if config.quote_query_loss_weight > 0:
            self.use_quote_query_loss = True
        else:
            self.use_quote_query_loss = False
        if config.hist_query_loss_weight > 0:
            self.use_hist_query_loss = True
        else:
            self.use_hist_query_loss = False
        if config.turn_encoder == 'transformer':
            self.quote_query_transform = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.encode_layer = TransformerEncoderLayer(config.embedding_dim, config.transformer_heads)
            self.turn_encoder = TransformerEncoder(self.encode_layer, config.transformer_layers)
            if config.model_name in ['ncf', 'ncf_query', 'ncf_query_exp']:
                self.quote_encode_layer = TransformerEncoderLayer(config.embedding_dim, config.transformer_heads)
                self.quote_encoder = TransformerEncoder(self.quote_encode_layer, config.transformer_layers)

        else:
            self.quote_query_transform = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
            self.turn_encoder = TurnEncoder(config, config.embedding_dim, config.hidden_dim)
            if config.model_name in ['ncf', 'ncf_query', 'ncf_query_exp']:
                self.quote_encoder = nn.GRU(config.hidden_dim, self.hidden_dim, dropout=config.dropout, bidirectional=True)
        if config.model_name == 'ncf_query_exp':
            if config.turn_encoder == 'transformer':
                self.exp_encode_layer = TransformerEncoderLayer(config.embedding_dim, config.transformer_heads)
                self.exp_encoder = TransformerEncoder(self.exp_encode_layer, config.transformer_layers)
            else:
                self.exp_encoder = nn.GRU(config.hidden_dim, self.hidden_dim, dropout=config.dropout, bidirectional=True)

        if config.model_name in ['ncf', 'ncf_query', 'ncf_query_exp']:
            if config.turn_encoder == 'transformer':
                self.hidden2label = nn.Linear(self.hidden_dim * 2 + config.quote_len, config.quote_len)
            else:
                self.hidden2label = nn.Linear(self.hidden_dim * 4 + config.quote_len, config.quote_len)
        elif config.model_name == 'LSTM-LSTM':
            self.hidden2label = nn.Linear(self.hidden_dim * 4, config.quote_len)
        if config.turn_encoder == 'transformer':
            self.conv_encoder = nn.GRU(self.hidden_dim, self.hidden_dim // 2, dropout=config.dropout, bidirectional=True)
        else:
            self.conv_encoder = nn.GRU(self.hidden_dim*2, self.hidden_dim, dropout=config.dropout, bidirectional=True)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def init_hidden(self, batch_size, hidden_dim, zero_init=False):
        if torch.cuda.is_available() and self.use_gpu and not self.print_attention:  # run in GPU
            if zero_init:
                return torch.zeros(2, batch_size, hidden_dim).cuda()
            else:
                return torch.randn(2, batch_size, hidden_dim).cuda()
        else:
            if zero_init:
                return torch.zeros(2, batch_size, hidden_dim)
            else:
                return torch.randn(2, batch_size, hidden_dim)

    def length_mask(self, turn_lens, max_len):
        masks = torch.arange(max_len).unsqueeze(0).repeat(turn_lens.size(0), 1)  # [turn_num, max_len]
        masks = torch.where(masks < turn_lens.unsqueeze(-1), torch.zeros_like(masks), torch.ones_like(masks))
        if torch.cuda.is_available() and self.use_gpu and not self.print_attention:  # run in GPU
            masks = masks.cuda()
        return masks

    def quote_query(self, padded_quotes):
        pass

    def forward(self, convs, conv_lens, conv_turn_lens, padded_quotes, quote_lens, padded_exp, exp_lens, hist_vecs=None, hist_lens=None, hist_turn_lens=None, labels=None):
        """convs: conversation input tokens [batch_size, conv_len, token_num]
           conv_lens: lengths of each conversation [batch_size]
           conv_turn_lens: lengths of each turn in conversation [batch_size, conv_len]
           users: user history input tokens [batch_size, history_len, token_num]
           user_lens: lengths of each user history [batch_size]
           user_turn_lens: lengths of each turn in user history [batch_size, history_len]"""

        """Conversation processing"""
        sorted_conv_lens, sorted_conv_indices = torch.sort(torch.LongTensor(conv_lens), descending=True)
        _, desorted_conv_indices = torch.sort(sorted_conv_indices, descending=False)
        conv_reps = []
        query_reps = []
        for idx in sorted_conv_indices:
            current_conv = convs[idx]
            current_len = conv_lens[idx]
            # Encode each turn in conversation, target turn initialized with user reps
            if not isinstance(self.turn_encoder, TransformerEncoder):
                turn_init = self.init_hidden(current_len, self.embedding_dim, zero_init=True)
                context_reps = self.turn_encoder(self.word_embedding(current_conv[:current_len]), torch.LongTensor(conv_turn_lens[idx][:current_len]), turn_init)
            else:
                masks = self.length_mask(torch.LongTensor(conv_turn_lens[idx][:current_len]), current_conv.size(1))  # [turn_num, max_len]
                context_reps = self.turn_encoder(self.word_embedding(current_conv[:current_len]).transpose(0,1), src_key_padding_mask=masks.bool())[0]
                context_reps = context_reps.transpose(0, 1)[:, 0]
            query_reps.append(context_reps[current_len-1])  # batch_size hidden*2
            conv_reps.append(context_reps) # batch_size turn_length hidden*2
        padded_convs = rnn_utils.pad_sequence(conv_reps, batch_first=True)
        packed_padded_convs = rnn_utils.pack_padded_sequence(padded_convs, sorted_conv_lens, batch_first=True)
        # Encode the whole conversation
        conv_initial_hidden = self.init_hidden(len(padded_convs), self.hidden_dim if isinstance(self.turn_encoder, TurnEncoder) else self.hidden_dim // 2)
        conv_output, conv_hidden = self.conv_encoder(packed_padded_convs, conv_initial_hidden)
        #conv_output: (32, 5, 400), if transformer -> 200
        conv_output = rnn_utils.pad_packed_sequence(conv_output, batch_first=True)[0][desorted_conv_indices]
        target_turn = torch.cat([conv_hidden[-1], conv_hidden[-2]], dim=1)[desorted_conv_indices]
        query_reps = torch.stack(query_reps)[desorted_conv_indices]
        if not self.no_transform:
            if not self.conv_not_transform:
                target_turn = self.quote_query_transform(target_turn)
            query_reps = self.quote_query_transform(query_reps)
        """Quotation processing"""
        if self.model_name in['ncf', 'ncf_query']:
            if not isinstance(self.quote_encoder, TransformerEncoder):
                sorted_quote_lens, sorted_quote_indices = torch.sort(torch.LongTensor(quote_lens), descending=True)
                _, desorted_quote_indices = torch.sort(sorted_quote_indices, descending=False)
                if torch.cuda.is_available() and self.use_gpu and not self.print_attention:  # run in GPU
                    if self.same_embed:
                        packed_padded_quotes = rnn_utils.pack_padded_sequence(self.word_embedding(padded_quotes.cuda())[sorted_quote_indices], sorted_quote_lens, batch_first=True)
                    else:
                        packed_padded_quotes = rnn_utils.pack_padded_sequence(self.quote_embedding(padded_quotes.cuda())[sorted_quote_indices], sorted_quote_lens, batch_first=True)
                else:
                    if self.same_embed:
                        packed_padded_quotes = rnn_utils.pack_padded_sequence(self.word_embedding(padded_quotes)[sorted_quote_indices], sorted_quote_lens, batch_first=True)
                    else:
                        packed_padded_quotes = rnn_utils.pack_padded_sequence(self.quote_embedding(padded_quotes)[sorted_quote_indices], sorted_quote_lens, batch_first=True)
                turn_init = self.init_hidden(len(padded_quotes), self.hidden_dim)
                quote_output, quote_hidden = self.quote_encoder(packed_padded_quotes, turn_init)
                quote_turn = torch.cat([quote_hidden[-1], quote_hidden[-2]], dim=1)[desorted_quote_indices]
            else:
                masks = self.length_mask(torch.LongTensor(quote_lens), max(quote_lens))  # [quote_num, quote_max_len]
                if torch.cuda.is_available() and self.use_gpu and not self.print_attention:  # run in GPU
                    quote_turn = self.quote_encoder(self.word_embedding(padded_quotes.cuda()).transpose(0, 1),
                                                    src_key_padding_mask=masks.bool())[0]
                else:
                    quote_turn = self.quote_encoder(self.word_embedding(padded_quotes).transpose(0, 1),
                                                    src_key_padding_mask=masks.bool())[0]
                quote_turn = quote_turn.transpose(0, 1)[:, 0]
            if self.model_name == 'ncf':
                mul = torch.mm(target_turn, quote_turn.t())
            elif self.model_name == 'ncf_query':
                mul = torch.mm(query_reps, quote_turn.t())
            target_turn = torch.cat((target_turn, mul), dim = 1)

        if self.model_name == 'ncf_query_exp':
            padded_quotes = self.word_embedding(padded_quotes.cuda())
            padded_exp = self.word_embedding(padded_exp.cuda())

            att = torch.stack([torch.mm(padded_quotes[i], padded_exp[i].transpose(0,1)) for i in range(self.quote_len)])
            quote_i = F.softmax(att, dim=1) # (1053, 4, 43)
            exp_i = F.softmax(att, dim=2)
            quote_att = torch.stack([torch.mm(quote_i[i], padded_exp[i]) for i in range(self.quote_len)])
            exp_att = torch.stack([torch.mm(exp_i[i].transpose(0,1), padded_quotes[i]) for i in range(self.quote_len)])
            sorted_quote_lens, sorted_quote_indices = torch.sort(torch.LongTensor(quote_lens), descending=True)
            _, desorted_quote_indices = torch.sort(sorted_quote_indices, descending=False)
            turn_init = self.init_hidden(len(padded_quotes), self.hidden_dim)

            sorted_exp_lens, sorted_exp_indices = torch.sort(torch.LongTensor(exp_lens), descending=True)
            _, desorted_exp_indices = torch.sort(sorted_exp_indices, descending=False)
            exp_init = self.init_hidden(len(padded_exp), self.hidden_dim)

            packed_padded_quotes = rnn_utils.pack_padded_sequence(padded_quotes.cuda(),
                                                            sorted_quote_lens, batch_first=True)
            packed_padded_exp = rnn_utils.pack_padded_sequence(padded_exp.cuda(),
                                                                  sorted_exp_lens, batch_first=True)

            quote_output, quote_hidden = self.quote_encoder(packed_padded_quotes, turn_init)
            quote_output = rnn_utils.pad_packed_sequence(quote_output, batch_first=True)[0][desorted_quote_indices]
            quote_turn = torch.cat([quote_hidden[-1], quote_hidden[-2]], dim=1)[desorted_quote_indices]

            exp_output, exp_hidden = self.exp_encoder(packed_padded_exp, exp_init)
            exp_output = rnn_utils.pad_packed_sequence(exp_output, batch_first=True)[0][desorted_exp_indices]
            exp_turn = torch.cat([exp_hidden[-1], exp_hidden[-2]], dim=1)[desorted_exp_indices]

            quote_turn = torch.cat((quote_turn, exp_turn), dim=1)
            query_reps = torch.cat((query_reps, query_reps), dim =1)
            query_reps = [i for i in query_reps]
            mul = torch.mm(query_reps, quote_turn.t())
            target_turn = torch.cat((target_turn, mul), dim=1)

        if self.print_attention or self.training and (self.use_quote_query_loss or self.use_hist_query_loss):
            assert labels is not None
            quote_reps = torch.index_select(quote_turn, 0, labels.long())  # batch_size hidden*2
        else:
            quote_reps = None

        """History processing for interpretation"""
        if self.print_attention:
            assert hist_vecs is not None and hist_lens is not None and hist_turn_lens is not None
            assert isinstance(self.turn_encoder, TransformerEncoder)
            hist_reps = []
            hist_word_atts = []
            for hist_vec, hist_len, hist_turn_len in zip(hist_vecs, hist_lens, hist_turn_lens):
                masks = self.length_mask(torch.LongTensor(hist_turn_len[:hist_len]), hist_vec.size(1))  # [turn_num, max_len]
                hist_rep, hist_word_att = self.turn_encoder(self.word_embedding(hist_vec[:hist_len]).transpose(0, 1),
                                                            src_key_padding_mask=masks.bool())
                hist_rep = hist_rep.transpose(0, 1)[:, 0]
                hist_word_att = hist_word_att[:, 0]
                hist_reps.append(hist_rep)
                hist_word_atts.append(hist_word_att)
            hist_reps = rnn_utils.pad_sequence(hist_reps, batch_first=True)  # batch_size hidden
            hist_word_atts = rnn_utils.pad_sequence(hist_word_atts, batch_first=True)  # batch_size hist_num
            # quote aware attention
            att_weights = torch.bmm(hist_reps, quote_reps.unsqueeze(-1)).squeeze(-1)  # batch_size turn_len
            att_masks = self.length_mask(torch.LongTensor(hist_lens), max(hist_lens))
            att_masks = torch.where(att_masks == 1, torch.zeros_like(att_masks).fill_(-1e8), torch.zeros_like(att_masks))
            att_weights += att_masks
            hist_query_atts = F.softmax(att_weights, dim=-1)

        else:
            hist_reps = None

        """Prediction Layer"""
        target_turn = torch.cat([target_turn, query_reps], dim=1)
        conv_labels = self.hidden2label(target_turn)
        if self.training and (self.use_quote_query_loss or self.use_hist_query_loss):
            return conv_labels, query_reps, quote_reps, hist_reps
        elif self.print_attention:
            return conv_labels, hist_word_atts, hist_query_atts
        return conv_labels

















