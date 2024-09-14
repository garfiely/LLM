import pickle
import numpy as np
import configparser
class embedding:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('/Users/nancy/Desktop/LLM/config/config.ini')
        self.modul_dim = int(config['DEFAULT']['MODUL_DIM'])
        self.heads = int(config['DEFAULT']['HEADS'])
        self.q_weight_dim = int(config['DEFAULT']['Q_WEIGHTS_DIM'])
        self.k_weight_dim = int(config['DEFAULT']['K_WEIGHTS_DIM'])
        self.v_weight_dim = int(config['DEFAULT']['V_WEIGHTS_DIM'])
        self.ffn_hidden_dim = int(config['DEFAULT']['FFN_HIDDEN_DIM'])
        self.input_seq_length = int(config['DEFAULT']['INPUT_SEQUENCE_LENGTH'])
        self.output_seq_length = int(config['DEFAULT']['OUTPUT_SEQUENCE_LENGTH'])
        self.encoder_layers = int(config['DEFAULT']['ENCODER_LAYERS'])
        self.decoder_layers = int(config['DEFAULT']['DECODER_LAYERS'])
        self.vocab_size = int(config['DEFAULT']['VOCAB_SIZE'])
        with open(f'/Users/nancy/Desktop/LLM/parameter/vocab_embedding.pkl', 'rb') as file:
            self.embedding_dict = pickle.load(file)
        with open(f'/Users/nancy/Desktop/LLM/parameter/vocab_onehot.pkl', 'rb') as file:
            self.one_hot_dict = pickle.load(file)

    def __call__(self, sequence_length : int, input_list : list) -> np.ndarray:
        """
        Embedding layer.
        """
        embedding_matrix = np.zeros((sequence_length, self.modul_dim))
        for i in range(sequence_length):
            embedding_matrix[i] = self.embedding_dict[input_list[i]]
        return embedding_matrix

#x = embedding()
#print(x(3,['<SOS>','+','2']))