#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import os, errno, sys, codecs, argparse, time, random, logging, json, collections
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from layers import cnn, embed, softmax, elmobilm

# python3 elmo.py --train_path data/train.txt --config_path data/config.json --model data/out


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def break_sent(sentence, max_sent_len):
  ret = []
  cur = 0
  l = len(sentence)
  while cur < l:
    if cur + max_sent_len + 5 >= l:
      ret.append(sentence[cur: l])
      break
    ret.append(sentence[cur: min(l, cur + max_sent_len)])
    cur += max_sent_len
  return ret


def read_corpus(path, max_chars=None, max_sent_len=20):
  
  data = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for line in fin:
      data.append('<bos>')
      for token in line.strip().split():
        if max_chars is not None and len(token) + 2 > max_chars:
          token = token[:max_chars - 2]
        data.append(token)
      data.append('<eos>')
  dataset = break_sent(data, max_sent_len)
  return dataset


def create_one_batch(x, word2id, char2id, config, oov='<oov>', pad='<pad>', sort=True):
 
  batch_size = len(x)
  lst = list(range(batch_size))
  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]
  lens = [len(x[i]) for i in lst]
  max_len = max(lens)

  if word2id is not None:
    oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
    assert oov_id is not None and pad_id is not None
    batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_w[i][j] = word2id.get(x_ij, oov_id)
  else:
    batch_w = None

  if char2id is not None:
    bow_id, eow_id, oov_id, pad_id = char2id.get('<eow>', None), char2id.get('<bow>', None), char2id.get(oov, None), char2id.get(pad, None)

    assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None

    if config['token_embedder']['name'].lower() == 'cnn':
      max_chars = config['token_embedder']['max_characters_per_token']
      assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars
    elif config['token_embedder']['name'].lower() == 'lstm':
      max_chars = max([len(w) for i in lst for w in x[i]]) + 2  # counting the <bow> and <eow>

    batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)

    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_c[i][j][0] = bow_id
        if x_ij == '<bos>' or x_ij == '<eos>':
          batch_c[i][j][1] = char2id.get(x_ij)
          batch_c[i][j][2] = eow_id
        else:
          for k, c in enumerate(x_ij):
            batch_c[i][j][k + 1] = char2id.get(c, oov_id)
          batch_c[i][j][len(x_ij) + 1] = eow_id
  else:
    batch_c = None

  masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

  for i, x_i in enumerate(x):
    for j in range(len(x_i)):
      masks[0][i][j] = 1
      if j + 1 < len(x_i):
        masks[1].append(i * max_len + j)
      if j > 0: 
        masks[2].append(i * max_len + j)

  assert len(masks[1]) <= batch_size * max_len
  assert len(masks[2]) <= batch_size * max_len

  masks[1] = torch.LongTensor(masks[1])
  masks[2] = torch.LongTensor(masks[2])

  return batch_w, batch_c, lens, masks

def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=True, sort=True, use_cuda=False):
 
  lst = perm or list(range(len(x)))
  if shuffle:
    random.shuffle(lst)

  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]

  sum_len = 0.0
  batches_w, batches_c, batches_lens, batches_masks = [], [], [], []
  size = batch_size
  nbatch = (len(x) - 1) // size + 1
  for i in range(nbatch):
    start_id, end_id = i * size, (i + 1) * size
    bw, bc, blens, bmasks = create_one_batch(x[start_id: end_id], word2id, char2id, config, sort=sort)
    sum_len += sum(blens)
    batches_w.append(bw)
    batches_c.append(bc)
    batches_lens.append(blens)
    batches_masks.append(bmasks)

  if sort:
    perm = list(range(nbatch))
    random.shuffle(perm)
    batches_w = [batches_w[i] for i in perm]
    batches_c = [batches_c[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    batches_masks = [batches_masks[i] for i in perm]

  #logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
  return batches_w, batches_c, batches_lens, batches_masks


class Model(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, n_class, use_cuda=False):
    super(Model, self).__init__() 
    self.use_cuda = use_cuda
    self.config = config

    if config['token_embedder']['name'].lower() == 'cnn':
      self.token_embedder = cnn(config, word_emb_layer, char_emb_layer, use_cuda)

    # if config['encoder']['name'].lower() == 'lstm':
    #   self.encoder = lstmbi(config, use_cuda)
    if config['encoder']['name'].lower() == 'elmo':
      self.encoder = elmobilm(config, use_cuda)

    self.output_dim = config['encoder']['projection_dim']
    if config['classifier']['name'].lower() == 'softmax':
      self.classify_layer = softmax(self.output_dim, n_class)

  def forward(self, word_inp, chars_inp, mask_package):
    
    classifier_name = self.config['classifier']['name'].lower()

    token_embedding = self.token_embedder(word_inp, chars_inp, (mask_package[0].size(0), mask_package[0].size(1)))
    token_embedding = F.dropout(token_embedding, self.config['dropout'], self.training)

    encoder_name = self.config['encoder']['name'].lower()
    if encoder_name == 'elmo':
      mask = Variable(mask_package[0].cuda()).cuda() if self.use_cuda else Variable(mask_package[0])
      encoder_output = self.encoder(token_embedding, mask)
      encoder_output = encoder_output[1]
      # [batch_size, len, hidden_size]
    # elif encoder_name == 'lstm':
    #   encoder_output = self.encoder(token_embedding)
    else:
      raise ValueError('')

    encoder_output = F.dropout(encoder_output, self.config['dropout'], self.training)
    forward, backward = encoder_output.split(self.output_dim, 2)

    word_inp = Variable(word_inp)
    if self.use_cuda:
      word_inp = word_inp.cuda()

    mask1 = Variable(mask_package[1].cuda()).cuda() if self.use_cuda else Variable(mask_package[1])
    mask2 = Variable(mask_package[2].cuda()).cuda() if self.use_cuda else Variable(mask_package[2])

    forward_x = forward.contiguous().view(-1, self.output_dim).index_select(0, mask1)
    forward_y = word_inp.contiguous().view(-1).index_select(0, mask2)

    backward_x = backward.contiguous().view(-1, self.output_dim).index_select(0, mask2)
    backward_y = word_inp.contiguous().view(-1).index_select(0, mask1)

    return self.classify_layer(forward_x, forward_y), self.classify_layer(backward_x, backward_y)

  def save_model(self, path, save_classify_layer):
    torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pkl'))    
    torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pkl'))
    if save_classify_layer:
      torch.save(self.classify_layer.state_dict(), os.path.join(path, 'classifier.pkl'))

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl')))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl')))
    self.classify_layer.load_state_dict(torch.load(os.path.join(path, 'classifier.pkl')))


# def eval_model(model, valid):
#   model.eval()
#   if model.config['classifier']['name'].lower() == 'cnn_softmax' or \
#       model.config['classifier']['name'].lower() == 'sampled_softmax':
#     model.classify_layer.update_embedding_matrix()
#     #print('emb mat size: ', model.classify_layer.embedding_matrix.size())
#   total_loss, total_tag = 0.0, 0
#   valid_w, valid_c, valid_lens, valid_masks = valid
#   for w, c, lens, masks in zip(valid_w, valid_c, valid_lens, valid_masks):
#     loss_forward, loss_backward = model.forward(w, c, masks)
#     total_loss += loss_forward.data[0]
#     n_tags = sum(lens)
#     total_tag += n_tags
#   model.train()
#   return np.exp(total_loss / total_tag)


def train_model(epoch, opt, model, optimizer,
                train, best_train):
  
  model.train()

  total_loss, total_tag = 0.0, 0
  cnt = 0
  start_time = time.time()

  train_w, train_c, train_lens, train_masks = train

  lst = list(range(len(train_w)))
  random.shuffle(lst)
  
  train_w = [train_w[l] for l in lst]
  train_c = [train_c[l] for l in lst]
  train_lens = [train_lens[l] for l in lst]
  train_masks = [train_masks[l] for l in lst]

  for w, c, lens, masks in zip(train_w, train_c, train_lens, train_masks):
    cnt += 1
    model.zero_grad()
    loss_forward, loss_backward = model.forward(w, c, masks)

    loss = (loss_forward + loss_backward) / 2.0
    total_loss += loss_forward.data[0]
    n_tags = sum(lens)
    total_tag += n_tags
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
    optimizer.step()
    if cnt * opt.batch_size % 1024 == 0:
      # logging.info("Epoch={} iter={} lr={:.6f} train_ppl={:.6f} time={:.2f}s".format(
      #   epoch, cnt, optimizer.param_groups[0]['lr'],
      #   np.exp(total_loss / total_tag), time.time() - start_time
      # ))
      logging.info("Epoch={} lr={:.6f} train_ppl={:.6f} time={:.2f}s".format(
        epoch, optimizer.param_groups[0]['lr'],
        np.exp(total_loss / total_tag), time.time() - start_time
      ))
      start_time = time.time()

    if cnt % opt.eval_steps == 0 or cnt % len(train_w) == 0:
      #if valid is None:
      train_ppl = np.exp(total_loss / total_tag)
      logging.info("Epoch={} iter={} lr={:.6f} train_ppl={:.6f}".format(
        epoch, cnt, optimizer.param_groups[0]['lr'], train_ppl))
      if train_ppl < best_train:
        best_train = train_ppl
        #logging.info("New record achieved on training dataset!")
        model.save_model(opt.model_path, opt.save_classify_layer)      
      
  return best_train


def get_truncated_vocab(dataset, min_count):
  
  word_count = Counter()
  for sentence in dataset:
    word_count.update(sentence)

  word_count = list(word_count.items())
  word_count.sort(key=lambda x: x[1], reverse=True)

  for i, (word, count) in enumerate(word_count):
    if count < min_count:
      break

  #logging.info('Truncated word count: {0}.'.format(sum([count for word, count in word_count[i:]])))
  logging.info('Original vocabulary size: {0}.'.format(len(word_count)))
  return word_count[:i]


def train():
  cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
  cmd.add_argument('--seed', default=1, type=int, help='The random seed.')
  cmd.add_argument('--gpu', default=1, type=int, help='Use id of gpu, -1 if cpu.')

  cmd.add_argument('--train_path', required=True, help='The path to the training file.')
  #cmd.add_argument('--valid_path', help='The path to the development file.')
  #cmd.add_argument('--test_path', help='The path to the testing file.')

  cmd.add_argument('--config_path', required=True, help='the path to the config file.')

  cmd.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adagrad'],
                   help='the type of optimizer: valid options=[sgd, adam, adagrad]')
  cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
  cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')

  cmd.add_argument("--model_path", required=True, help="path to save model")
  
  cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
  cmd.add_argument("--max_epoch", type=int, default=10, help='the maximum number of iteration.')
  
  cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')

  cmd.add_argument('--max_sent_len', type=int, default=20, help='maximum sentence length.')

  cmd.add_argument('--min_count', type=int, default=5, help='minimum word count.')

  cmd.add_argument('--max_vocab_size', type=int, default=150000, help='maximum vocabulary size.')

  cmd.add_argument('--save_classify_layer', default=False, action='store_true',
                   help="whether to save the classify layer")

  cmd.add_argument('--valid_size', type=int, default=0, help="size of validation dataset when there's no valid.")
  cmd.add_argument('--eval_steps', required=False, type=int, help='report every xx batches.')

  opt = cmd.parse_args(sys.argv[1:])

  with open(opt.config_path, 'r') as fin:
    config = json.load(fin)

  # Dump configurations
  # print(opt)
  # print(config)

  # set seed.
  torch.manual_seed(opt.seed)
  random.seed(opt.seed)
  if opt.gpu >= 0:
    torch.cuda.set_device(opt.gpu)
    if opt.seed > 0:
      torch.cuda.manual_seed(opt.seed)

  use_cuda = opt.gpu >= 0 and torch.cuda.is_available()

  token_embedder_name = config['token_embedder']['name'].lower()
  token_embedder_max_chars = config['token_embedder'].get('max_characters_per_token', None)
  if token_embedder_name == 'cnn':
    train_data = read_corpus(opt.train_path, token_embedder_max_chars, opt.max_sent_len)
  else:
    raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))

  logging.info('training instance: {}, training tokens: {}.'.format(len(train_data),
                                                                    sum([len(s) - 1 for s in train_data])))

  valid_data = None
  test_data = None

  word_lexicon = {}

  # Maintain the vocabulary. vocabulary is used in either WordEmbeddingInput or softmax classification
  vocab = get_truncated_vocab(train_data, opt.min_count)

  # Ensure index of '<oov>' is 0
  for special_word in ['<oov>', '<bos>', '<eos>',  '<pad>']:
    if special_word not in word_lexicon:
      word_lexicon[special_word] = len(word_lexicon)

  for word, _ in vocab:
    if word not in word_lexicon:
      word_lexicon[word] = len(word_lexicon)

  # Word Embedding
  if config['token_embedder']['word_dim'] > 0:
    word_emb_layer = embed(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False)
    logging.info('Word embedding size: {0}'.format(len(word_emb_layer.word2id)))
  else:
    word_emb_layer = None
    logging.info('Vocabulary size: {0}'.format(len(word_lexicon)))

  # Character Lexicon
  if config['token_embedder']['char_dim'] > 0:
    char_lexicon = {}
    for sentence in train_data:
      for word in sentence:
        for ch in word:
          if ch not in char_lexicon:
            char_lexicon[ch] = len(char_lexicon)

    for special_char in ['<bos>', '<eos>', '<oov>', '<pad>', '<bow>', '<eow>']:
      if special_char not in char_lexicon:
        char_lexicon[special_char] = len(char_lexicon)

    char_emb_layer = embed(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False)
    logging.info('Char embedding size: {0}'.format(len(char_emb_layer.word2id)))
  else:
    char_lexicon = None
    char_emb_layer = None

  train = create_batches(
    train_data, opt.batch_size, word_lexicon, char_lexicon, config, use_cuda=use_cuda)

  if opt.eval_steps is None:
    opt.eval_steps = len(train[0])
  #logging.info('Evaluate every {0} batches.'.format(opt.eval_steps))

  valid = None
  test = None

  # if valid_data is not None:
  #   valid = create_batches(
  #     valid_data, opt.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, use_cuda=use_cuda)
  # else:
  #   valid = None

  # if test_data is not None:
  #   test = create_batches(
  #     test_data, opt.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, use_cuda=use_cuda)
  # else:
  #   test = None

  label_to_ix = word_lexicon
  logging.info('vocab size: {0}'.format(len(label_to_ix)))
  
  nclasses = len(label_to_ix)

  #for s in train_w:
  #  for i in range(s.view(-1).size(0)):
  #    if s.view(-1)[i] >= nclasses:
  #      print(s.view(-1)[i])

  model = Model(config, word_emb_layer, char_emb_layer, nclasses, use_cuda)
  logging.info(str(model))
  if use_cuda:
    model = model.cuda()

  need_grad = lambda x: x.requires_grad
  optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=opt.lr)

  # if opt.optimizer.lower() == 'adam':
  #   optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=opt.lr)
  # elif opt.optimizer.lower() == 'sgd':
  #   optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=opt.lr)
  # elif opt.optimizer.lower() == 'adagrad':
  #   optimizer = optim.Adagrad(filter(need_grad, model.parameters()), lr=opt.lr)
  # else:
  #   raise ValueError('Unknown optimizer {}'.format(opt.optimizer.lower()))

  try:
    os.makedirs(opt.model_path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

  if config['token_embedder']['char_dim'] > 0:
    with codecs.open(os.path.join(opt.model_path, 'char.dic'), 'w', encoding='utf-8') as fpo:
      for ch, i in char_emb_layer.word2id.items():
        print('{0}\t{1}'.format(ch, i), file=fpo)

  with codecs.open(os.path.join(opt.model_path, 'word.dic'), 'w', encoding='utf-8') as fpo:
    for w, i in word_lexicon.items():
      print('{0}\t{1}'.format(w, i), file=fpo)

  json.dump(vars(opt), codecs.open(os.path.join(opt.model_path, 'config.json'), 'w', encoding='utf-8'))

  best_train = 1e+8

  for epoch in range(opt.max_epoch):
    best_train = train_model(epoch, opt, model, optimizer, train, best_train)
    if opt.lr_decay > 0:
      optimizer.param_groups[0]['lr'] *= opt.lr_decay

  logging.info("best train ppl: {:.6f}.".format(best_train))

if __name__ == "__main__":
  #if len(sys.argv) > 1 and sys.argv[1] == 'train':
  train()
