import numpy as np
import configparser
import pickle

class initialize_LLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
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

    def __call__(self):
        return self.kwargs
    
    def generate_position_encoding(self):
        """
        Generate the position encoding for the LLM model.
        """
        position_encoding = np.zeros((self.input_seq_length, self.modul_dim))
        for pos in range(self.input_seq_length):
            for i in range(0, self.modul_dim, 2):
                position_encoding[pos, i] = np.sin(pos / 10000 ** (i / self.modul_dim))
                position_encoding[pos, i + 1] = np.cos(pos / 10000 ** (i / self.modul_dim))
        with open('/Users/nancy/Desktop/LLM/parameter/position_encoding_matrix.pkl', 'wb') as file:
            pickle.dump(position_encoding, file)
        print("Position encoding matrix generated.", position_encoding.shape)

    def generate_QKV_weights(self):
        """
        Generate the weights for the LLM model.
        """
        for layer in range(self.encoder_layers):
            q_weights = np.random.uniform(-1,1,(self.heads, self.modul_dim, self.q_weight_dim))
            k_weights = np.random.uniform(-1,1,(self.heads, self.modul_dim, self.k_weight_dim))
            v_weights = np.random.uniform(-1,1,(self.heads, self.modul_dim, self.v_weight_dim))
            with open(f'/Users/nancy/Desktop/LLM/parameter/encoder_layer{layer}_q_weights.pkl', 'wb') as file:
                pickle.dump(q_weights, file)
            with open(f'/Users/nancy/Desktop/LLM/parameter/encoder_layer{layer}_k_weights.pkl', 'wb') as file:
                pickle.dump(k_weights, file)
            with open(f'/Users/nancy/Desktop/LLM/parameter/encoder_layer{layer}_v_weights.pkl', 'wb') as file:
                pickle.dump(v_weights, file)
            print("Encoder weights generated.", q_weights.shape, k_weights.shape, v_weights.shape)
        for layer in range(self.decoder_layers):
            q_weights = np.random.uniform(-1,1,(self.heads, self.modul_dim, self.q_weight_dim))
            k_weights = np.random.uniform(-1,1,(self.heads, self.modul_dim, self.k_weight_dim))
            v_weights = np.random.uniform(-1,1,(self.heads, self.modul_dim, self.v_weight_dim))
            with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{layer}_q_weights.pkl', 'wb') as file:
                pickle.dump(q_weights, file)
            with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{layer}_k_weights.pkl', 'wb') as file:
                pickle.dump(k_weights, file)
            with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{layer}_v_weights.pkl', 'wb') as file:
                pickle.dump(v_weights, file)
            print("Decoder weights generated.", q_weights.shape, k_weights.shape, v_weights.shape)
        for layer in range(self.decoder_layers):
            q_weights = np.random.uniform(-1,1,(self.heads, self.modul_dim, self.q_weight_dim))
            k_weights = np.random.uniform(-1,1,(self.heads, self.modul_dim, self.k_weight_dim))
            v_weights = np.random.uniform(-1,1,(self.heads, self.modul_dim, self.v_weight_dim))
            with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{layer}_cross_q_weights.pkl', 'wb') as file:
                pickle.dump(q_weights, file)
            with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{layer}_cross_k_weights.pkl', 'wb') as file:
                pickle.dump(k_weights, file)
            with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{layer}_cross_v_weights.pkl', 'wb') as file:
                pickle.dump(v_weights, file)
            print("Decoder weights generated.", q_weights.shape, k_weights.shape, v_weights.shape)
        print("Weights matrix generated.")

    def generate_ffn_weights(self):
        """
        Generate the weights for the feed forward network.
        """
        for layer in range(self.encoder_layers):
            ffn_up_dim_matrix = np.random.uniform(-1,1,(self.modul_dim, self.ffn_hidden_dim))
            with open(f'/Users/nancy/Desktop/LLM/parameter/encoder_layer{layer}_ffn_up_dim_matrix.pkl', 'wb') as file:
                pickle.dump(ffn_up_dim_matrix, file)
            ffn_down_dim_matrix = np.random.uniform(-1,1,(self.ffn_hidden_dim, self.modul_dim))
            with open(f'/Users/nancy/Desktop/LLM/parameter/encoder_layer{layer}_ffn_down_dim_matrix.pkl', 'wb') as file:
                pickle.dump(ffn_down_dim_matrix, file)
            print("Encoder FFN weights generated.", ffn_up_dim_matrix.shape, ffn_down_dim_matrix.shape)
        for layer in range(self.decoder_layers):
            ffn_up_dim_matrix = np.random.uniform(-1,1,(self.modul_dim, self.ffn_hidden_dim))
            with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{layer}_ffn_up_dim_matrix.pkl', 'wb') as file:
                pickle.dump(ffn_up_dim_matrix, file)
            ffn_down_dim_matrix = np.random.uniform(-1,1,(self.ffn_hidden_dim, self.modul_dim))
            with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{layer}_ffn_down_dim_matrix.pkl', 'wb') as file:
                pickle.dump(ffn_down_dim_matrix, file)
            print("Decoder FFN weights generated.", ffn_up_dim_matrix.shape, ffn_down_dim_matrix.shape)
        print("FFN weights matrix generated.")

    def generate_linear_matrix(self):
        """
        Generate the output weights for the LLM model.
        """
        linear_matrix = np.random.uniform(-1,1,(self.modul_dim, self.vocab_size))
        with open('/Users/nancy/Desktop/LLM/parameter/linear_matrix.pkl', 'wb') as file:
            pickle.dump(linear_matrix, file)
        """
        Generate the bias for the output layer.
        """
        bias = np.random.uniform(-1,1,(self.vocab_size))
        with open('/Users/nancy/Desktop/LLM/parameter/linear_bias.pkl', 'wb') as file:
            pickle.dump(bias, file)
        print("Output weights generated.", linear_matrix.shape, bias.shape)

    def generate_all_parameters(self):
        """
        Generate all the parameters for the LLM model.
        """
        self.generate_position_encoding()
        self.generate_QKV_weights()
        self.generate_ffn_weights()
        self.generate_linear_matrix()
        print("All parameters generated.")

#a = initialize_LLM()
#a.generate_QKV_weights()
#a.generate_all_parameters()