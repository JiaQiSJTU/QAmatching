import os


from .general_utils import get_logger
from .data_utils import get_processing_word,get_processed_embeddings


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        # self.vocab_words = load_vocab(self.filename_words)
        # self.vocab_tags  = load_vocab(self.filename_tags)
        # self.vocab_chars = load_vocab(self.filename_chars)

        # self.nwords     = len(self.vocab_words)
        # self.nchars     = len(self.vocab_chars)
        self.ntags      = 2

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.filename_word2idx)
        self.processing_tag  = {'False':0,"True":1}

        # 3. get pre-trained embeddings
        self.embeddings = get_processed_embeddings(self.filename_glove)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 100
    # dim_char = 100

    # embedding files
    filename_glove = "./datafile/word_emb_reduced.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    # filename_trimmed = "data/Ch_word2vec.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_dev = "./datafile/dev-full.json"
    filename_test = "./datafile/test-full.json"
    filename_train = "./datafile/train-full.json"

    # filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_word2idx = "./datafile/word2idx-new.json"


    # training
    train_embeddings = False

    nepochs          = 15
    dropout          = 0.3
    batch_size       = 1
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.95
    clip             = 5.0  # if negative, no clipping
    nepoch_no_imprv  = 3
    max_question_len  = 100
    max_answer_len  = 100
    max_historyQ_len  = 10
    max_historyA_len  = 10

    # model hyperparameters
    # hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 128 # lstm on word embeddings


