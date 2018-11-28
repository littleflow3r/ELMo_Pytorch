from __future__ import absolute_import
from __future__ import unicode_literals
import logging, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
from typing import Optional, Tuple, List, Callable, Union
from overrides import overrides

class softmax(nn.Module):
  def __init__(self, output_dim, n_class):
    super(softmax, self).__init__()
    self.hidden2tag = nn.Linear(output_dim, n_class)
    #self.criterion = nn.CrossEntropyLoss(size_average=False)
    self.loss = nn.CrossEntropyLoss(size_average=False)

  def forward(self, x, y):
    tag_scores = self.hidden2tag(x)
    return self.loss(tag_scores, y)

class Highway(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                            for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
            gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input

class cnn(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda):
    super(cnn, self).__init__()
    self.config = config
    self.use_cuda = use_cuda

    self.word_emb_layer = word_emb_layer
    self.char_emb_layer = char_emb_layer

    self.output_dim = config['encoder']['projection_dim']
    self.emb_dim = 0
    if word_emb_layer is not None:
      self.emb_dim += word_emb_layer.n_d

    if char_emb_layer is not None:
      self.convolutions = []
      cnn_config = config['token_embedder']
      filters = cnn_config['filters']
      char_embed_dim = cnn_config['char_dim']

      for i, (width, num) in enumerate(filters):
        conv = torch.nn.Conv1d(
          in_channels=char_embed_dim,
          out_channels=num,
          kernel_size=width,
          bias=True
        )
        self.convolutions.append(conv)

      self.convolutions = nn.ModuleList(self.convolutions)
      
      self.n_filters = sum(f[1] for f in filters)
      self.n_highway = cnn_config['n_highway']

      self.highways = Highway(self.n_filters, self.n_highway, activation=torch.nn.functional.relu)
      self.emb_dim += self.n_filters

    self.projection = nn.Linear(self.emb_dim, self.output_dim, bias=True)
    
  def forward(self, word_inp, chars_inp, shape):
    embs = []
    batch_size, seq_len = shape
    if self.word_emb_layer is not None:
      batch_size, seq_len = word_inp.size(0), word_inp.size(1)
      word_emb = self.word_emb_layer(Variable(word_inp).cuda() if self.use_cuda else Variable(word_inp))
      embs.append(word_emb)

    if self.char_emb_layer is not None:
      chars_inp = chars_inp.view(batch_size * seq_len, -1)

      character_embedding = self.char_emb_layer(Variable(chars_inp).cuda() if self.use_cuda else Variable(chars_inp))

      character_embedding = torch.transpose(character_embedding, 1, 2)

      cnn_config = self.config['token_embedder']
      if cnn_config['activation'] == 'tanh':
        activation = torch.nn.functional.tanh
      elif cnn_config['activation'] == 'relu':
        activation = torch.nn.functional.relu
      else:
        raise Exception("Unknown activation")

      convs = []
      for i in range(len(self.convolutions)):
        convolved = self.convolutions[i](character_embedding)
        # (batch_size * sequence_length, n_filters for this width)
        convolved, _ = torch.max(convolved, dim=-1)
        convolved = activation(convolved)
        convs.append(convolved)
      char_emb = torch.cat(convs, dim=-1)
      char_emb = self.highways(char_emb)

      embs.append(char_emb.view(batch_size, -1, self.n_filters))
      
    token_embedding = torch.cat(embs, dim=2)

    return self.projection(token_embedding)

class embed(nn.Module):
  def __init__(self, n_d, word2id, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True):
    super(embed, self).__init__()
    self.word2id = word2id
    self.id2word = {i: word for word, i in word2id.items()}
    self.n_V, self.n_d = len(word2id), n_d
    self.oovid = word2id[oov]
    self.padid = word2id[pad]
    self.embedding = nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
    self.embedding.weight.data.uniform_(-0.25, 0.25)

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

from encoder_base import _EncoderBase
from lstm_cell_with_projection import LstmCellWithProjection

RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  # pylint: disable=invalid-name
RnnStateStorage = Tuple[torch.Tensor, ...]  # pylint: disable=invalid-name

class elmobilm(_EncoderBase):
  def __init__(self, config, use_cuda=False):
    super(elmobilm, self).__init__(stateful=True)
    self.config = config
    self.use_cuda = use_cuda
    input_size = config['encoder']['projection_dim']
    hidden_size = config['encoder']['projection_dim']
    cell_size = config['encoder']['dim']
    num_layers = config['encoder']['n_layers']
    memory_cell_clip_value = config['encoder']['cell_clip']
    state_projection_clip_value = config['encoder']['proj_clip']
    recurrent_dropout_probability = config['dropout']

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.cell_size = cell_size
    
    forward_layers = []
    backward_layers = []

    lstm_input_size = input_size
    go_forward = True
    for layer_index in range(num_layers):
      forward_layer = LstmCellWithProjection(lstm_input_size,
                                             hidden_size,
                                             cell_size,
                                             go_forward,
                                             recurrent_dropout_probability,
                                             memory_cell_clip_value,
                                             state_projection_clip_value)
      backward_layer = LstmCellWithProjection(lstm_input_size,
                                              hidden_size,
                                              cell_size,
                                              not go_forward,
                                              recurrent_dropout_probability,
                                              memory_cell_clip_value,
                                              state_projection_clip_value)
      lstm_input_size = hidden_size

      self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
      self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
      forward_layers.append(forward_layer)
      backward_layers.append(backward_layer)
    self.forward_layers = forward_layers
    self.backward_layers = backward_layers

  def forward(self, inputs, mask):
    batch_size, total_sequence_length = mask.size()
    stacked_sequence_output, final_states, restoration_indices = \
      self.sort_and_run_forward(self._lstm_forward, inputs, mask)

    num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
    # Add back invalid rows which were removed in the call to sort_and_run_forward.
    if num_valid < batch_size:
      zeros = stacked_sequence_output.data.new(num_layers,
                                               batch_size - num_valid,
                                               returned_timesteps,
                                               encoder_dim).fill_(0)
      zeros = Variable(zeros)
      stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

      # The states also need to have invalid rows added back.
      new_states = []
      for state in final_states:
        state_dim = state.size(-1)
        zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
        zeros = Variable(zeros)
        new_states.append(torch.cat([state, zeros], 1))
      final_states = new_states

    # It's possible to need to pass sequences which are padded to longer than the
    # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
    # the sequences mean that the returned tensor won't include these dimensions, because
    # the RNN did not need to process them. We add them back on in the form of zeros here.
    sequence_length_difference = total_sequence_length - returned_timesteps
    if sequence_length_difference > 0:
      zeros = stacked_sequence_output.data.new(num_layers,
                                               batch_size,
                                               sequence_length_difference,
                                               stacked_sequence_output[0].size(-1)).fill_(0)
      zeros = Variable(zeros)
      stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

    self._update_states(final_states, restoration_indices)

    # Restore the original indices and return the sequence.
    # Has shape (num_layers, batch_size, sequence_length, hidden_size)
    return stacked_sequence_output.index_select(1, restoration_indices)


  def _lstm_forward(self, 
                    inputs: PackedSequence,
                    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
      Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Parameters
    ----------
    inputs : ``PackedSequence``, required.
      A batch first ``PackedSequence`` to run the stacked LSTM over.
    initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
      A tuple (state, memory) representing the initial hidden state and memory
      of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
      (num_layers, batch_size, 2 * cell_size) respectively.
    Returns
    -------
    output_sequence : ``torch.FloatTensor``
      The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
    final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
      The per-layer final (state, memory) states of the LSTM, with shape
      (num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
      respectively. The last dimension is duplicated because it contains the state/memory
      for both the forward and backward layers.
    """
       
    if initial_state is None:
      hidden_states: List[Optional[Tuple[torch.Tensor,
                                   torch.Tensor]]] = [None] * len(self.forward_layers)
    elif initial_state[0].size()[0] != len(self.forward_layers):
      raise Exception("Initial states were passed to forward() but the number of "
                               "initial states does not match the number of layers.")
    else:
      hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

    inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
    forward_output_sequence = inputs
    backward_output_sequence = inputs

    final_states = []
    sequence_outputs = []
    for layer_index, state in enumerate(hidden_states):
      forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
      backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

      forward_cache = forward_output_sequence
      backward_cache = backward_output_sequence

      if state is not None:
        forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
        forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
        forward_state = (forward_hidden_state, forward_memory_state)
        backward_state = (backward_hidden_state, backward_memory_state)
      else:
        forward_state = None
        backward_state = None

      forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                             batch_lengths,
                                                             forward_state)
      backward_output_sequence, backward_state = backward_layer(backward_output_sequence,
                                                                batch_lengths,
                                                                backward_state)
      # Skip connections, just adding the input to the output.
      if layer_index != 0:
        forward_output_sequence += forward_cache
        backward_output_sequence += backward_cache

      sequence_outputs.append(torch.cat([forward_output_sequence,
                                         backward_output_sequence], -1))
      # Append the state tuples in a list, so that we can return
      # the final states for all the layers.
      final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                           torch.cat([forward_state[1], backward_state[1]], -1)))

    stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
    # Stack the hidden state and memory for each layer into 2 tensors of shape
    # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
    # respectively.
    final_hidden_states, final_memory_states = zip(*final_states)
    final_state_tuple: Tuple[torch.FloatTensor,
                             torch.FloatTensor] = (torch.cat(final_hidden_states, 0),
                                                   torch.cat(final_memory_states, 0))
    return stacked_sequence_outputs, final_state_tuple


# class lstmbi(nn.Module):
#   def __init__(self, config, use_cuda=False):
#     super(lstmbi, self).__init__()
#     self.config = config
#     self.use_cuda = use_cuda
    
#     self.encoder = nn.LSTM(self.config['encoder']['projection_dim'],
#                            self.config['encoder']['dim'],
#                            num_layers=self.config['encoder']['n_layers'], 
#                            bidirectional=True,
#                            batch_first=True, 
#                            dropout=self.config['dropout'])
#     self.projection = nn.Linear(self.config['encoder']['dim'], self.config['encoder']['projection_dim'], bias=True)

#   def forward(self, inputs):
#     forward, backward = self.encoder(inputs)[0].split(self.config['encoder']['dim'], 2)
#     return torch.cat([self.projection(forward), self.projection(backward)], dim=2)
