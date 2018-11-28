#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os, sys, codecs, argparse, logging, json, collections, random
import torch
import numpy as np
import h5py
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from layers import lstmbi, cnn, embed, softmax, elmobilm

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

#python3 representation.py --input input/input.txt --out_ave output/avg --out_emb output/emb --out_lstm output/lstm --out_lstm2 output/lstm2 --model data/mymodel

class embed(nn.Module):
  def __init__(self, n_d, word2id, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True):
    super(embed, self).__init__()
    if embs is not None:
      embwords, embvecs = embs
      # for word in embwords:
      #  assert word not in word2id, "Duplicate words in pre-trained embeddings"
      #  word2id[word] = len(word2id)

      logging.info("{} pre-trained word embeddings loaded.".format(len(word2id)))
      if n_d != len(embvecs[0]):
        logging.warning("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.".format(
          n_d, len(embvecs[0]), len(embvecs[0])))
        n_d = len(embvecs[0])

    self.word2id = word2id
    self.id2word = {i: word for word, i in word2id.items()}
    self.n_V, self.n_d = len(word2id), n_d
    self.oovid = word2id[oov]
    self.padid = word2id[pad]
    self.embedding = nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
    self.embedding.weight.data.uniform_(-0.25, 0.25)

    if embs is not None:
      weight = self.embedding.weight
      weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
      logging.info("embedding shape: {}".format(weight.size()))

    if normalize:
      weight = self.embedding.weight
      norms = weight.data.norm(2, 1)
      if norms.dim() == 1:
        norms = norms.unsqueeze(1)
      weight.data.div_(norms.expand_as(weight.data))

    if fix_emb:
      self.embedding.weight.requires_grad = False

  def forward(self, input_):
    return self.embedding(input_)

def create_one_batch(x, word2id, char2id, config, oov='<oov>', pad='<pad>', sort=True):

  batch_size = len(x)
  # lst represents the order of sentences
  lst = list(range(batch_size))
  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  # shuffle the sentences by
  x = [x[i] for i in lst]
  lens = [len(x[i]) for i in lst]
  max_len = max(lens)

  # get a batch of word id whose size is (batch x max_len)
  if word2id is not None:
    oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
    assert oov_id is not None and pad_id is not None
    batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_w[i][j] = word2id.get(x_ij, oov_id)
  else:
    batch_w = None

  # get a batch of character id whose size is (batch x max_len x max_chars)
  if char2id is not None:
    bow_id, eow_id, oov_id, pad_id = [char2id.get(key, None) for key in ('<eow>', '<bow>', oov, pad)]

    assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None

    if config['token_embedder']['name'].lower() == 'cnn':
      max_chars = config['token_embedder']['max_characters_per_token']
      assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars
    elif config['token_embedder']['name'].lower() == 'lstm':
      # counting the <bow> and <eow>
      max_chars = max([len(w) for i in lst for w in x[i]]) + 2
    else:
      raise ValueError('Unknown token_embedder: {0}'.format(config['token_embedder']['name']))

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

  # mask[0] is the matrix (batch x max_len) indicating whether
  # there is an id is valid (not a padding) in this batch.
  # mask[1] stores the flattened ids indicating whether there is a valid
  # previous token
  # mask[2] stores the flattened ids indicating whether there is a valid
  # next token
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


# shuffle training examples and create mini-batches
def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=True, sort=True, text=None):

  lst = perm or list(range(len(x)))
  if shuffle:
    random.shuffle(lst)

  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]
  if text is not None:
    text = [text[i] for i in lst]

  sum_len = 0.0
  batches_w, batches_c, batches_lens, batches_masks, batches_text = [], [], [], [], []
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
    if text is not None:
      batches_text.append(text[start_id: end_id])

  if sort:
    perm = list(range(nbatch))
    random.shuffle(perm)
    batches_w = [batches_w[i] for i in perm]
    batches_c = [batches_c[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    batches_masks = [batches_masks[i] for i in perm]
    if text is not None:
      batches_text = [batches_text[i] for i in perm]

  logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
  if text is not None:
    return batches_w, batches_c, batches_lens, batches_masks, batches_text
  return batches_w, batches_c, batches_lens, batches_masks


class Model(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, n_class, use_cuda=False):
    super(Model, self).__init__() 
    self.use_cuda = use_cuda
    self.config = config

    if config['token_embedder']['name'].lower() == 'cnn':
      self.token_embedder = cnn(config, word_emb_layer, char_emb_layer, use_cuda)

    if config['encoder']['name'].lower() == 'lstm':
      self.encoder = lstmbi(config, use_cuda)
    if config['encoder']['name'].lower() == 'elmo':
      self.encoder = elmobilm(config, use_cuda)

    self.output_dim = config['encoder']['projection_dim']

  def forward(self, word_inp, chars_inp, mask_package):
    token_embedding = self.token_embedder(word_inp, chars_inp, (mask_package[0].size(0), mask_package[0].size(1)))
    if self.config['encoder']['name'] == 'elmo':
      mask = Variable(mask_package[0]).cuda() if self.use_cuda else Variable(mask_package[0])
      encoder_output = self.encoder(token_embedding, mask)
      sz = encoder_output.size()
      token_embedding = torch.cat([token_embedding, token_embedding], dim=2).view(1, sz[1], sz[2], sz[3])
      encoder_output = torch.cat([token_embedding, encoder_output], dim=0)
    elif self.config['encoder']['name'] == 'lstm':
      encoder_output = self.encoder(token_embedding)
    return encoder_output

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl')))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl')))
    #self.classify_layer.load_state_dict(torch.load(os.path.join(path, 'classifier.pkl')))

def dict2namedtuple(dic):
  return collections.namedtuple('Namespace', dic.keys())(**dic)

def read_corpus(path, max_chars = None):
  """
  read raw text file
  :param path:
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for line in fin.read().strip().split('\n'):
      data = ['<bos>']
      text = []
      for token in line.split('\t'):
        text.append(token)
        if max_chars is not None and len(token) + 2 > max_chars:
          token = token[:max_chars - 2]
        data.append(token)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)

  #print (dataset, textset)
  #sys.exit()
  return dataset, textset

def test_main():
  # Configurations
  cmd = argparse.ArgumentParser('The testing components of')
  cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
  cmd.add_argument("--input", help="the path to the raw text file.")
  cmd.add_argument('--out_ave', help='the path to the average embedding file.')
  cmd.add_argument('--out_emb', help='the path to the word embedding file.')
  cmd.add_argument('--out_lstm', help='the path to the 1st lstm-output embedding file.')
  cmd.add_argument('--out_lstm2', help='the path to the 2st lstm-output embedding file.')
  cmd.add_argument("--model", required=True, help="path to the model")
  cmd.add_argument("--batch_size", "--batch", type=int, default=1, help='the batch size.')
  args = cmd.parse_args(sys.argv[1:])

  if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
  use_cuda = args.gpu >= 0 and torch.cuda.is_available()
  # load the model configurations
  args2 = dict2namedtuple(json.load(codecs.open(os.path.join(args.model, 'config.json'), 'r', encoding='utf-8')))
  #print (args2.config_path)

  with open(os.path.join(args2.config_path), 'r') as fin:
    config = json.load(fin)

  # For the model trained with character-based word encoder.
  if config['token_embedder']['char_dim'] > 0:
    char_lexicon = {}
    with codecs.open(os.path.join(args.model, 'char.dic'), 'r', encoding='utf-8') as fpi:
      for line in fpi:
        tokens = line.strip().split('\t')
        if len(tokens) == 1:
          tokens.insert(0, '\u3000')
        token, i = tokens
        char_lexicon[token] = int(i)
    char_emb_layer = embed(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False, embs=None)
    logging.info('char embedding size: ' + str(len(char_emb_layer.word2id)))
  else:
    char_lexicon = None
    char_emb_layer = None

  # For the model trained with word form word encoder.
  if config['token_embedder']['word_dim'] > 0:
    word_lexicon = {}
    with codecs.open(os.path.join(args.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
      for line in fpi:
        tokens = line.strip().split('\t')
        if len(tokens) == 1:
          tokens.insert(0, '\u3000')
        token, i = tokens
        word_lexicon[token] = int(i)
    word_emb_layer = embed(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
    logging.info('word embedding size: ' + str(len(word_emb_layer.word2id)))
  else:
    word_lexicon = None
    word_emb_layer = None

  # instantiate the model
  model = Model(config, word_emb_layer, char_emb_layer, use_cuda)

  if use_cuda:
    model.cuda()

  logging.info(str(model))
  model.load_model(args.model)

  if config['token_embedder']['name'].lower() == 'cnn':
    test, text = read_corpus(args.input, config['token_embedder']['max_characters_per_token'])
  else:
    test, text = read_corpus(args.input)

  # create test batches from the input data.
  test_w, test_c, test_lens, test_masks, test_text = create_batches(
    test, args.batch_size, word_lexicon, char_lexicon, config, text=text)

  # configure the model to evaluation mode.
  model.eval()

  sent_set = set()

  cnt = 0

  fout_ave = h5py.File(args.out_ave, 'w') if args.out_ave is not None else None
  fout_emb = h5py.File(args.out_emb, 'w') if args.out_emb is not None else None
  fout_lstm = h5py.File(args.out_lstm, 'w') if args.out_lstm is not None else None
  fout_lstm2 = h5py.File(args.out_lstm2, 'w') if args.out_lstm2 is not None else None

  for w, c, lens, masks, texts in zip(test_w, test_c, test_lens, test_masks, test_text):
    output = model.forward(w, c, masks)
    for i, text in enumerate(texts):
      sent = '\t'.join(text)
      sent = sent.replace('.', '$period$')
      sent = sent.replace('/', '$backslash$')
      if sent in sent_set:
        continue
      sent_set.add(sent)
      if config['encoder']['name'].lower() == 'lstm':
        data = output[i, 1:lens[i]-1, :].data
        if use_cuda:
          data = data.cpu()
        data = data.numpy()
      elif config['encoder']['name'].lower() == 'elmo':
        data = output[:, i, 1:lens[i]-1, :].data
        if use_cuda:
          data = data.cpu()
        data = data.numpy()
      
      print (data.shape)

      if fout_ave is not None:
        data_ave = np.average(data, axis=0)
        fout_ave.create_dataset(
          sent,
          data_ave.shape, dtype='float32',
          data=data_ave
        )
        print ('finish avg')
    
      if fout_emb is not None:
        data_emb = data[0]
        fout_emb.create_dataset(
            sent, 
            data_emb.shape, dtype='float32',
            data = data_emb
        )
        print ('finish emb')

      if fout_lstm is not None:
        data_lstm = data[1]
        fout_lstm.create_dataset(
          sent,
          data_lstm.shape, dtype='float32',
          data=data_lstm
        )
        print ('finish lstm 1')
      
      if fout_lstm2 is not None:
        data_lstm2 = data[2]
        fout_lstm2.create_dataset(
        sent,
        data_lstm2.shape, dtype='float32',
        data=data_lstm2
        )
        print ('finish lstm 2')

      cnt += 1
      if cnt % 1000 == 0:
        logging.info('Finished {0} sentences.'.format(cnt))
  if fout_ave is not None:
    fout_ave.close()
  if fout_lstm is not None:
    fout_lstm.close()
  if fout_emb is not None:
    fout_emb.close()
  if fout_lstm2 is not None:
    fout_lstm2.close()


if __name__ == "__main__":
  test_main()
  # if len(sys.argv) > 1 and sys.argv[1] == 'test':
  #   test_main()
  # else:
  #   print('Usage: {0} [test] [options]'.format(sys.argv[0]), file=sys.stderr)