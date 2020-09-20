import numpy as np
import os
import json
import codecs


# shared global variables to be imported from model also
UNK = '--NULL--'
NONE = "O"

# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class QADataset(object):
    """Class that iterates over Dataset

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        self.yield_ = """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename,'r') as f:
            data = json.load(f)
            # linenum=0
            # for line in f:
            for item in data:
                # words, tags = [], []
                questions,answers,targets=[],[],[]
                originalQ,originalA=[],[]
                sentence1ID,sentence2ID=[],[]
                sessionID=[]

                #history
                historyQ,historyA=[],[]

                historyq,historya=[],[]
                if len(item['historyQ'])==0:
                    historyq=[[0]]
                else:
                    for sen in item['historyQ'][-10:]:
                        senq=sen.strip().split(' ')[-100:]
                        historyq.append([self.processing_word[w] if w in self.processing_word else 0 for w in senq])
                historyQ+=historyq

                if len(item['historyA'])==0:
                    historya=[[0]]
                else:
                    for sen in item['historyA'][-10:]:
                        sena=sen.strip().split(' ')[-100:]
                        historya.append([self.processing_word[w] if w in self.processing_word else 0 for w in sena])
                historyA+=historya

                #distance
                distances=[]
                distance=[0]*10
                if item['distance']<10:
                    distance[item['distance']-1]=1
                else:
                    distance[9]=1

                question=item['sentence1'].strip().split(' ')[-100:]
                answer=item['sentence2'].strip().split(' ')[-100:]
                target=item['gold_label']

                if self.processing_word is not None:
                    question = [self.processing_word[w] if w in self.processing_word else 0 for w in question]
                    answer = [self.processing_word[w] if w in self.processing_word else 0 for w in answer]
                if self.processing_tag is not None:
                        target = self.processing_tag[target]
                questions+=question
                answers+=answer
                targets += [target]

                originalQ += [item['sentence1']]
                originalA += [item['sentence2']]
                sentence1ID += [item['sentence1ID']]
                sentence2ID += [item['sentence2ID']]
                sessionID  +=[item['sessionID']]
                distances.append(distance)

                yield questions,answers,targets,originalQ,originalA,sentence1ID,sentence2ID,sessionID,distances,historyQ,historyA



    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_processed_embeddings(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        embedding = []
        with codecs.open(filename, 'r', encoding='utf8') as file:
            for line in file:
                row = line.strip().split(',')
                embedding.append([float(i) for i in row])
        embedding=np.array(embedding)
        return embedding

    except IOError:
        raise MyIOError(filename)


def get_processing_word(filepath):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = (12345)
                 = ( word id)

    """
    with open(filepath,'r') as f:
        word2idx=json.load(f)

    return word2idx


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

def pad_conversation(sequence):
    sequence_length=[]
    max_length=10
    for j,item in enumerate(sequence):
        if item[0]==[0] and len(item)==1:
            sequence_length.append(0)
            sequence[j]+=[[0]]*(max_length-1)
        else:
            sequence_length.append(len(item))
            for i in range(max_length-len(item)):
                sequence[j]+=[[0]]

    return sequence,sequence_length



def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    sequence_padded, sequence_length = [], []
    if nlevels == 1:
        # max_length = max(map(lambda x : len(x), sequences))
        max_length = 100
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    q_batch,a_batch, y_batch,originalQ,originalA =[], [], [],[],[]
    sentence1ID , sentence2ID = [],[]
    sessionID=[]
    distances=[]
    historyQ,historyA=[],[]

    for (q, a, y,oq,oa,s1,s2,d,dis,hq,ha) in data:

        if len(q_batch) == minibatch_size:
            yield q_batch,a_batch, y_batch,originalQ,originalA,sentence1ID,sentence2ID,sessionID,distances,historyQ,historyA
            q_batch,a_batch, y_batch,originalQ,originalA,sentence1ID,sentence2ID,sessionID,distances,historyQ,historyA =[],[], [],[], [], [],[],[],[],[],[]

        if type(q[0]) == tuple:
            q = zip(*q)
        q_batch += [q]
        if type(a[0]) == tuple:
            a=zip(*a)
        a_batch += [a]
        y_batch+=y
        originalQ+=oq
        originalA+=oa
        sentence1ID+=s1
        sentence2ID+=s2
        sessionID+=d
        distances+=dis
        historyQ+=[hq]
        historyA+=[ha]


    if len(q_batch) != 0:
        num=minibatch_size-len(q_batch)
        count=0
        for (q, a, y, oq, oa, s1, s2, d,dis,hq,ha) in data:
            count+=1
            if(count>num):
                break
            if type(q[0]) == tuple:
                q = zip(*q)
            q_batch += [q]
            if type(a[0]) == tuple:
                a = zip(*a)
            a_batch += [a]
            y_batch += y
            originalQ += oq
            originalA += oa
            sentence1ID += s1
            sentence2ID += s2
            sessionID += d
            distances+=dis
            historyQ += [hq]
            historyA += [ha]

        yield q_batch,a_batch, y_batch,originalQ,originalQ,sentence1ID,sentence2ID,sessionID,distances,historyQ,historyA
