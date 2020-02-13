import numpy as np
import os
import tensorflow as tf
# from sklearn.metrics import f1_score,precision_score,recall_score
from .data_utils import minibatches, pad_sequences,pad_conversation
from .general_utils import Progbar
from .base_model import BaseModel
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers import core as layers_core
import collections
import json
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ntags=2


def get_hidden_state(cell_state):
    """ Get the hidden state needed in cell state which is
        possibly returned by LSTMCell, GRUCell, RNNCell or MultiRNNCell.

    Args:
      cell_state: a structure of cell state

    Returns:
      hidden_state: A Tensor
    """

    if type(cell_state) is tuple:
        cell_state = cell_state[-1]
    if hasattr(cell_state, "h"):
        hidden_state = cell_state.h
    else:
        hidden_state = cell_state
    return hidden_state

class AttentionA(object):


  def __init__(self,num_units,premise_mem,premise_mem_weights,name="AttentionA"):
    """ Init SeqMatchSeqAttention

    Args:
      num_units: The depth of the attention mechanism.
      premise_mem: encoded premise memory
      premise_mem_weights: premise memory weights
    """
    # Init layers
    self._name = name
    self._num_units = num_units
    # Shape: [batch_size,max_premise_len,rnn_size]
    self._premise_mem = premise_mem
    # Shape: [batch_size,max_premise_len]
    self._premise_mem_weights = premise_mem_weights

    with tf.name_scope(self._name):
      # self.query_layer = layers_core.Dense(num_units, name="query_layer", use_bias=False)
      self.hypothesis_mem_layer = layers_core.Dense(num_units, name="hypothesis_mem_layer", use_bias=False)
      self.premise_mem_layer = layers_core.Dense(num_units, name="premise_mem_layer", use_bias=False)
      # Preprocess premise Memory
      # Shape: [batch_size, max_premise_len, num_units]
      self._keys = self.premise_mem_layer(premise_mem)
      self.batch_size = self._keys.shape[0].value
      self.alignments_size = self._keys.shape[1].value

  def __call__(self, hypothesis_mem):
    """
    Perform attention

    Args:
      hypothesis_mem: hypothesis memory   [batch_size, max_time, rnn_size]

    Returns:
      attention: computed attention
    """
    with tf.name_scope(self._name):
      attention=[]
      v1 = tf.get_variable("attention_v13", [self._num_units], dtype=tf.float32)
      # v2 = tf.get_variable("attention_v23", [self._num_units], dtype=tf.float32)
      # v3 = tf.get_variable("attention_v33", [self._num_units], dtype=tf.float32)
      for i in range(hypothesis_mem.shape[1].value): 
        # q_hidden = hypothesis_mem[:,i,:]
        q_hidden = tf.squeeze(tf.slice(hypothesis_mem,[0,i,0],[-1,1,-1]),axis=1)
        q_hidden = tf.expand_dims(self.hypothesis_mem_layer(q_hidden),1)

        score = tf.reduce_sum(v1 * tf.tanh(self._keys + q_hidden ), [2])
        # score = tf.reduce_sum(v1 * tf.tanh(v2 * self._keys + v3 * q_hidden ), [2])
        # Mask score with -inf
        score_mask_values = float("-inf") * (1.-tf.cast(self._premise_mem_weights, tf.float32))
        # print(self._premise_mem_weights) 64*56
        masked_score = tf.where(tf.cast(self._premise_mem_weights, tf.bool), score, score_mask_values)
      # Calculate alignments
      # Shape: [batch_size, max_premise_len] 
        alignments = tf.nn.softmax(masked_score)
      # Calculate attention
      # Shape: [batch_size, rnn_size]
        attention.append(tf.reduce_sum(tf.expand_dims(alignments, 2) * self._premise_mem, axis=1))
      attention=tf.convert_to_tensor(attention)
      attention=tf.transpose(attention,[1,0,2])
      # print(attention)
      # exit(0)
      return attention
class AttentionQ(object):

  def __init__(self,num_units,premise_mem,premise_mem_weights,name="AttentionQ"):
    """ Init SeqMatchSeqAttention

    Args:
      num_units: The depth of the attention mechanism.
      premise_mem: encoded premise memory
      premise_mem_weights: premise memory weights
    """
    # Init layers
    self._name = name
    self._num_units = num_units
    # Shape: [batch_size,max_premise_len,rnn_size]
    self._premise_mem = premise_mem
    # Shape: [batch_size,max_premise_len]
    self._premise_mem_weights = premise_mem_weights

    with tf.name_scope(self._name):
      # self.query_layer = layers_core.Dense(num_units, name="query_layer", use_bias=False)
      self.hypothesis_mem_layer = layers_core.Dense(num_units, name="hypothesis_mem_layer", use_bias=False)
      self.premise_mem_layer = layers_core.Dense(num_units, name="premise_mem_layer", use_bias=False)
      # Preprocess premise Memory
      # Shape: [batch_size, max_premise_len, num_units]
      self._keys = self.premise_mem_layer(premise_mem)
      self.batch_size = self._keys.shape[0].value
      self.alignments_size = self._keys.shape[1].value

  def __call__(self, hypothesis_mem):
    """
    Perform attention

    Args:
      hypothesis_mem: hypothesis memory   [batch_size, max_time, rnn_size]

    Returns:
      attention: computed attention
    """
    with tf.name_scope(self._name):
      attention=[]
      # alignment=[]
      # a=[]
      v1 = tf.get_variable("attention_v12", [self._num_units], dtype=tf.float32)
      # v2 = tf.get_variable("attention_v22", [self._num_units], dtype=tf.float32)
      # v3 = tf.get_variable("attention_v32", [self._num_units], dtype=tf.float32)

      for i in range(hypothesis_mem.shape[1].value):
        # q_hidden = hypothesis_mem[:,i,:]
        # if hypothesis_mem.shape[0].value==1:
        #   q_hidden = tf.squeeze(tf.slice(hypothesis_mem,[0,i,0],[-1,1,-1]))
        #   q_hidden = tf.expand_dims(self.hypothesis_mem_layer(q_hidden), 0)
        # else:
        q_hidden = tf.squeeze(tf.slice(hypothesis_mem,[0,i,0],[-1,1,-1]),axis=1)
        # print(q_hidden)
        q_hidden = tf.expand_dims(self.hypothesis_mem_layer(q_hidden),1)
        # score = tf.reduce_sum(v1 * tf.tanh(v2 * self._keys + v3 * q_hidden ), [2])
        score = tf.reduce_sum(v1 * tf.tanh(self._keys + q_hidden ), [2])
        # Mask score with -inf
        score_mask_values = float("-inf") * (1.-tf.cast(self._premise_mem_weights, tf.float32))
        # print(self._premise_mem_weights) 64*56
        masked_score = tf.where(tf.cast(self._premise_mem_weights, tf.bool), score, score_mask_values)
      # Calculate alignments
      # Shape: [batch_size, max_premise_len] 
        alignments = tf.nn.softmax(masked_score)
      # Calculate attention
      # Shape: [batch_size, rnn_size]
      #   alignment.append(masked_score)
      #   a.append(alignments)
        attention.append(tf.reduce_sum(tf.expand_dims(alignments, 2) * self._premise_mem, axis=1))
      attention=tf.convert_to_tensor(attention)
      # alignment=tf.convert_to_tensor(alignment)
      # a=tf.convert_to_tensor(a)
      attention=tf.transpose(attention,[1,0,2])
      # alignment=tf.transpose(alignment,[1,0,2])
      # print(attention)
      # exit(0)
      return attention


class SeqMatchSeqAttentionState(
    collections.namedtuple("SeqMatchSeqAttentionState", ("cell_state", "attention"))):
    pass


class SeqMatchSeqAttention(object):
    """ Attention for SeqMatchSeq.
    """

    def __init__(self, num_units, premise_mem, premise_mem_weights, name="SeqMatchSeqAttention"):
        """ Init SeqMatchSeqAttention

        Args:
          num_units: The depth of the attention mechanism.
          premise_mem: encoded premise memory
          premise_mem_weights: premise memory weights
        """
        # Init layers
        self._name = name
        self._num_units = num_units
        # Shape: [batch_size,max_premise_len,rnn_size]
        self._premise_mem = premise_mem
        # Shape: [batch_size,max_premise_len]
        self._premise_mem_weights = premise_mem_weights

        with tf.name_scope(self._name):
            self.query_layer = layers_core.Dense(num_units, name="query_layer", use_bias=False)
            self.hypothesis_mem_layer = layers_core.Dense(num_units, name="hypothesis_mem_layer", use_bias=False)
            self.premise_mem_layer = layers_core.Dense(num_units, name="premise_mem_layer", use_bias=False)
            # Preprocess premise Memory
            # Shape: [batch_size, max_premise_len, num_units]
            self._keys = self.premise_mem_layer(premise_mem)
            self.batch_size = self._keys.shape[0].value
            self.alignments_size = self._keys.shape[1].value

    def __call__(self, hypothesis_mem, query):
        """
        Perform attention

        Args:
          hypothesis_mem: hypothesis memory   [batch_size, max_time, rnn_size][batch_size,rnn_size]
          query: hidden state from last time step

        Returns:
          attention: computed attention
        """
        with tf.name_scope(self._name):
            # Shape: [batch_size, 1, num_units]
            # [batch_size,rnn_size]->[batch_size,attention_size]->[batch_size,1,num_units]
            processed_hypothesis_mem = tf.expand_dims(self.hypothesis_mem_layer(hypothesis_mem), 1)
            # Shape: [batch_size, 1, num_units]
            processed_query = tf.expand_dims(self.query_layer(query), 1)

            # v1 = tf.get_variable("attention_v1", [self._num_units], dtype=tf.float32)
            # v2 = tf.get_variable("attention_v2", [self._num_units], dtype=tf.float32)
            # v3 = tf.get_variable("attention_v3", [self._num_units], dtype=tf.float32)
            # v4 = tf.get_variable("attention_v4", [self._num_units], dtype=tf.float32)
            # score = tf.reduce_sum(v1 * tf.tanh(v2 * self._keys + v3 * processed_hypothesis_mem + v4 * processed_query),[2])
            #
            v = tf.get_variable("attention_v", [self._num_units], dtype=tf.float32)
            score = tf.reduce_sum(v * tf.tanh(self._keys + processed_hypothesis_mem + processed_query), [2])
            # Mask score with -inf
            score_mask_values = float("-inf") * (1. - tf.cast(self._premise_mem_weights, tf.float32))
            # print(score)
            # print(score_mask_values)
            # print(self._premise_mem_weights)
            masked_score = tf.where(tf.cast(self._premise_mem_weights, tf.bool), score, score_mask_values)
            # Calculate alignments
            # Shape: [batch_size, max_premise_len] 
            alignments = tf.nn.softmax(masked_score)
            # Calculate attention
            # Shape: [batch_size, rnn_size]
            attention = tf.reduce_sum(tf.expand_dims(alignments, 2) * self._premise_mem, axis=1)
            return attention


class SeqMatchSeqWrapper(rnn_cell_impl.RNNCell):
    """ RNN Wrapper for SeqMatchSeq.
    """

    def __init__(self, cell, attention_mechanism,num_units, name='SeqMatchSeqWrapper'):
        super(SeqMatchSeqWrapper, self).__init__(name=name)
        self._cell = cell
        self._attention_mechanism = attention_mechanism
        self._num_units=num_units

    def call(self, inputs, state):
        """
        Args:
          inputs: inputs at some time step
          state: A (structure of) cell state
        """

        cell_inputs = tf.concat([state.attention, inputs], axis=-1)
        # v = tf.get_variable("attention_v", [self._num_units], dtype=tf.float32)
        # cell_inputs = tf.sigmoid(v * tf.concat([state.attention, inputs], axis=-1))

        cell_state = state.cell_state
        # Call cell function
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        # Get hidden state
        hidden_state = get_hidden_state(cell_state)
        # Calculate attention
        # hidden_state: from last time step
        # inputs :hypothesis memory
        attention = self._attention_mechanism(inputs, hidden_state)
        # Assemble next state
        next_state = SeqMatchSeqAttentionState(
            cell_state=next_cell_state,
            attention=attention)
        return cell_output, next_state

    @property
    def state_size(self):
        return SeqMatchSeqAttentionState(
            cell_state=self._cell.state_size,
            attention=self._attention_mechanism._premise_mem.get_shape()[-1].value
        )

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        cell_state = self._cell.zero_state(batch_size, dtype)
        attention = rnn_cell_impl._zero_state_tensors(self.state_size.attention, batch_size, tf.float32)
        return SeqMatchSeqAttentionState(
            cell_state=cell_state,
            attention=attention)


class QAModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(QAModel, self).__init__(config)
        self.idx_to_tag = {0:False,1:True}


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self._question=tf.placeholder(tf.int32,shape=[None,self.config.max_question_len],name='question_ids')
        self._answer=tf.placeholder(tf.int32,shape=[None,self.config.max_answer_len],name='answer_ids')

        # shape = (batch size)
        self._question_lengths = tf.placeholder(tf.int32, shape=[None],name="question_lengths")
        self._answer_lengths = tf.placeholder(tf.int32, shape=[None],name="answer_lengths")

        #distance
        self._distance = tf.placeholder(tf.int32, shape=[None,10],name="distance")

        #history batch_size*max_historyQ_len in batch*max_sentence_length in batch
        self._historyQ = tf.placeholder(tf.int32, shape=[None,None,None],name='history_question')
        self._historyA = tf.placeholder(tf.int32, shape=[None,None,None],name='history_answer')
        #history_lengths batch_size
        self._historyQ_lens=tf.placeholder(tf.int32,shape=[None],name='historyQ_lengths')
        self._historyA_lens=tf.placeholder(tf.int32,shape=[None],name='historyA_lengths')
        #history_sentence_lengths batch_size*max_historyQ_len in batch
        self._historyQ_senlens=tf.placeholder(tf.int32,shape=[None,None],name='historyQ_senlens')
        self._historyA_senlens=tf.placeholder(tf.int32,shape=[None,None],name='historyA_senlens')

        # shape = (batch size)
        self.labels = tf.placeholder(tf.int32, shape=[None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, questions, answers,distances,historyQ,historyA,
                      labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """

        questions, question_lengths = pad_sequences(questions, 0)
        answers, answer_lengths = pad_sequences(answers, 0)

        historyQ,historyQ_len = pad_conversation(historyQ)
        historyA,historyA_len = pad_conversation(historyA)
        
        historyQ_senlen=[]
        historyQ_padded=[]
        # print("historyQ:{}".format(historyQ))
        for i in range(len(historyQ)):
            # print(historyQ[i])
            historyQ_temp,historyQ_sen_temp=pad_sequences(historyQ[i],0)
            for j in range(historyQ_len[i],len(historyQ_sen_temp),1):
                historyQ_sen_temp[j]=0
            historyQ_padded.append(historyQ_temp)
            historyQ_senlen.append(historyQ_sen_temp)



        historyA_senlen=[]
        historyA_padded=[]
        for i in range(len(historyA)):
            historyA_temp,historyA_sen_temp=pad_sequences(historyA[i],0)
            for j in range(historyA_len[i],len(historyA_sen_temp),1):
                historyA_sen_temp[j]=0
            historyA_padded.append(historyA_temp)
            historyA_senlen.append(historyA_sen_temp)




        # print("historyQ:{}".format(historyQ))
        # print("historyA:{}".format(historyA))
        # print("historyQ_padded:{}".format(historyQ_padded))
        # print("historyA_padded:{}".format(historyA_padded))
        # print("historyQ_len:{}".format(historyQ_len))
        # print("historyA_len:{}".format(historyA_len))
        # print("historyQ_senlen:{}".format(historyQ_senlen))
        # print("historyA_senlen:{}".format(historyA_senlen))

        # build feed dictionary
        feed = {
            self._question: questions,
            self._answer  : answers,
            self._distance:distances,
            self._question_lengths : question_lengths,
            self._answer_lengths   : answer_lengths,
            self._historyQ:historyQ_padded,
            self._historyA:historyA_padded,
            self._historyQ_lens:historyQ_len,
            self._historyA_lens:historyA_len,
            self._historyQ_senlens:historyQ_senlen,
            self._historyA_senlens:historyA_senlen
        }


        if labels is not None:
        #     labels, _ = pad_sequences(labels, 0)

            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)
            #batch*length*embedding
            self.question_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self._question, name="word_embeddings")
            self.answer_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self._answer, name="word_embeddings")
            self.historyQ_embedding = tf.nn.embedding_lookup(_word_embeddings,self._historyQ,name='historyQ_embedding')
            self.historyA_embedding = tf.nn.embedding_lookup(_word_embeddings,self._historyA,name='historyA_embedding')


        # self.question_embeddings =  tf.nn.dropout(question_embeddings, self.dropout)
        # self.answer_embeddings =  tf.nn.dropout(answer_embeddings, self.dropout)

    def add_label_embeddings_op(self):
        self.label_onehot = tf.one_hot(self.labels, ntags, 1, 0,axis=-1)
        # print(self.labels)
        # print("****")
        # print(self.label_onehot)

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_label_embeddings_op()

        historyQ_encoder = []  # sentence embedding for histroy q  --corresponding length self._history_lens
        historyA_encoder = []
        with tf.variable_scope("sentence_encoding"):
            sentence_encoder = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm),
                                                             input_keep_prob=1 - self.config.dropout,
                                                             output_keep_prob=1 - self.config.dropout)
            for i in range(self.config.batch_size):
                _, state1 = tf.nn.dynamic_rnn(sentence_encoder, self.historyQ_embedding[i], self._historyQ_senlens[i],
                                              dtype=tf.float32)
                _, state2 = tf.nn.dynamic_rnn(sentence_encoder, self.historyA_embedding[i], self._historyA_senlens[i],
                                              dtype=tf.float32)
                historyQ_encoder.append(state1[1])
                historyA_encoder.append(state2[1])
        historyQ_encoder = tf.convert_to_tensor(historyQ_encoder)  # the list of sentence embeddings in a history
        historyA_encoder = tf.convert_to_tensor(historyA_encoder)


        # self.add_logits_op()
        with tf.variable_scope("answer_question_encoding"):
            lstm_cell=tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm)
            question_encoder = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                            input_keep_prob=1 - self.config.dropout,
                                                            output_keep_prob=1 - self.config.dropout)
            question_mem,_=tf.nn.dynamic_rnn(question_encoder,self.question_embeddings,sequence_length=self._question_lengths,dtype=tf.float32)
            # print(output)

        # with tf.variable_scope("answer_encoding"):
            # lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm, forget_bias=1, state_is_tuple=True)
            # answer_encoder = tf.contrib.rnn.DropoutWrapper(lstm_cell,
            #                                                 input_keep_prob=1 - self.config.dropout,
            #                                                 output_keep_prob=1 - self.config.dropout)
            answer_mem, _ = tf.nn.dynamic_rnn(question_encoder, self.answer_embeddings,sequence_length=self._answer_lengths ,dtype=tf.float32)

        with tf.name_scope("attentionHQ"):
            historyQ_weights=tf.cast(tf.sequence_mask(self._historyQ_lens,self.config.max_historyQ_len),tf.int32)
            # A attention history Q
            attention_mechinismQ = AttentionQ(self.config.hidden_size_lstm, historyQ_encoder, historyQ_weights)
            # print(answer_mem)
            # exit(0)
            attentionHQT = attention_mechinismQ(answer_mem)
            zeros = tf.zeros(shape=[self.config.batch_size, self.config.max_question_len, self.config.hidden_size_lstm])
            attentionHQ = tf.where(self._historyQ_lens > 0, attentionHQT, zeros)
            new_answer_mem = tf.concat([answer_mem, attentionHQ], 2)

        with tf.name_scope("attentionHA"):
            historyA_weights=tf.cast(tf.sequence_mask(self._historyA_lens,self.config.max_historyA_len),tf.int32)
            # Q attention history A
            attention_mechinismA = AttentionA(self.config.hidden_size_lstm, historyA_encoder, historyA_weights)
            attentionHAT = attention_mechinismA(question_mem)
            zeros = tf.zeros(shape=[self.config.batch_size, self.config.max_answer_len, self.config.hidden_size_lstm])
            attentionHA = tf.where(self._historyA_lens > 0, attentionHAT, zeros)
            new_question_mem = tf.concat([question_mem, attentionHA], 2)

        with tf.variable_scope("new_question_answer_encoding"):
            final_encoder = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm * 2),
                                                            input_keep_prob=1 - self.config.dropout,
                                                            output_keep_prob=1 - self.config.dropout)
            # Encode premise
            # Shape: [batch_size, max_time, rnn_size]
            final_question_mem, _ = tf.nn.dynamic_rnn(final_encoder, new_question_mem, self._question_lengths,
                                                     dtype=tf.float32)
            final_answer_mem, _ = tf.nn.dynamic_rnn(final_encoder, new_answer_mem, self._answer_lengths,
                                                        dtype=tf.float32)

        with tf.variable_scope("SMS_attention"):
            question_weights=tf.cast(tf.sequence_mask(self._question_lengths,self.config.max_question_len),tf.int32)
            attention_mechanism = SeqMatchSeqAttention(self.config.hidden_size_lstm*2, final_question_mem, question_weights)
            # match LSTM
            mLSTM = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm*2,forget_bias=1,state_is_tuple=True)
            # Wrap mLSTM
            mLSTM = SeqMatchSeqWrapper(mLSTM, attention_mechanism,self.config.hidden_size_lstm*4)
            _,state=tf.nn.dynamic_rnn(mLSTM,final_answer_mem,self._answer_lengths,dtype=tf.float32)
            hidden_state=get_hidden_state(state.cell_state)

        with tf.variable_scope("Fully_connected_layer"):
            fcn = layers_core.Dense(2,name='fcn')
            logits=fcn(tf.concat([hidden_state,tf.cast(self._distance,dtype=tf.float32)],axis=-1))

        with tf.name_scope("loss"):
            self._prob = tf.nn.softmax(logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
            self.loss = tf.reduce_mean(losses)
            # add summary
        loss_summary = tf.summary.scalar("loss", self.loss)

        with tf.name_scope("accuracy"):
            self.prediction = tf.cast(tf.argmax(self._prob, 1, name="prediction"),dtype=tf.int32)
            correct_prediction = tf.equal(self.prediction, self.labels)
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        # add summary
        accuracy_summary = tf.summary.scalar("accuracy_summary", self.accuracy)

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, questions,answers,distances,historyQ,historyA, Flag,labels=None):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        if Flag==True:#test
            fd = self.get_feed_dict(questions,answers,distances,historyQ,historyA,  dropout=1.0)

            labels_pred = self.sess.run(self._prob, feed_dict=fd)

            return labels_pred
        else:#dev
            fd = self.get_feed_dict(questions, answers,distances,historyQ,historyA, labels, dropout=1.0)

            labels_pred,loss = self.sess.run([self._prob,self.loss], feed_dict=fd)
            return labels_pred,loss


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        # previous_loss=[]
        # iterate over dataset
        # total_loss=0
        for i, (questions,answers, labels,_,_,_,_,_,distances,historyQ,historyA) in enumerate(minibatches(train, batch_size)):

            fd = self.get_feed_dict(questions,answers,distances,historyQ,historyA, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary,train_accuracy = self.sess.run(
                    [self.train_op, self.loss, self.merged,self.accuracy], feed_dict=fd)
            # total_loss+=train_loss
            prog.update(i + 1, [("train loss", train_loss),("train acc",train_accuracy)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)
        # total_loss=total_loss/nbatches
        metric = self.run_evaluate(dev,Flag=False)
        # msg = "{}".format(metric)
        msg = " - ".join(["{} {:04.6f}".format(k, v)
                          for k, v in metric.items()])
        self.logger.info(msg)

        return metric['dev accuracy'],metric['dev loss']
        # return metric['accuracy']


    def run_evaluate(self, test,Flag=False):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)
            Flag:True--evaluate\False--dev

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = 0
        total=0
        # truelist=[]
        output=[]
        loss_total=0
        for questions,answers, labels,originalQ,originalA,sentence1ID,sentence2ID,sessionID,distances,historyQ,historyA in minibatches(test, self.config.batch_size):
                #labels(batch_size)
            #labels_pred(batchsize*num_tags)
            # print(labels)
            if Flag==True:
                labels_pred = self.predict_batch(questions,answers,distances,historyQ,historyA,Flag)
            else:
                labels_pred,loss=self.predict_batch(questions,answers,distances,historyQ,historyA,Flag,labels)
                loss_total += loss
            # prob=np.softmax()
            pred=np.argmax(labels_pred,1)

            total += len(labels)
            if Flag==True:
                print(total)
            templabels=np.array(labels)



            accs+=np.sum(np.equal(pred,templabels))

            if Flag==True:
                for i in range(len(labels)):
                    temp={}
                    temp['sentence1']=originalQ[i]
                    temp['sentence2']=originalA[i]
                    temp['sentence1ID']=sentence1ID[i]
                    temp['sentence2ID']=sentence2ID[i]
                    # print(sessionID)
                    temp['sessionID']=sessionID[i]
                    temp['gold_label']=labels[i]
                    temp['predict_label']=int(pred[i])
                    temp['T_prob']=float(labels_pred[i][1])
                    output.append(temp)

        if Flag==True:
            
            print('begin to write into file:~~~~~~~~~~~~~~~~~')
            json_str = json.dumps(output)
            with open("./results/result.json", 'w') as json_file:
                json_file.write(json_str)

        # f1_all=f1_score(truelist, predlist, average=None)
        # f1_macro=f1_score(truelist, predlist, average='macro')
        # f1_micro=f1_score(truelist, predlist, average='micro')
        acc = float(accs/total)
        loss_total=float(loss_total/total*self.config.batch_size)
        # return {"acc":100*acc,"f1_macro":100*f1_macro,"f1_micro":100*f1_micro}
        if Flag == True:
            return {"test accuracy": acc}
        else:
            return {"dev accuracy": acc,"dev loss":loss_total}


    # def predict(self, words_raw):
    #     """Returns list of tags
    #
    #     Args:
    #         words_raw: list of words (string), just one sentence (no batch)
    #
    #     Returns:
    #         preds: list of tags (string), one for each word in the sentence
    #
    #     """
    #     words = [self.config.processing_word(w) for w in words_raw]
    #     if type(words[0]) == tuple:
    #         words = zip(*words)
    #     pred_ids= self.predict_batch([words])
    #     preds = [self.idx_to_tag[pred_ids]]
    #
    #     return preds
