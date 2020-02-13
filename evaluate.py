from model.data_utils import QADataset
from model.full_model import QAModel
from model.config import Config

def main():
    # create instance of config
    config = Config()

    # build model
    model = QAModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = QADataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    # interactive_shell(model)


if __name__ == "__main__":
    main()
