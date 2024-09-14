import numpy as np
from function_modul import softmax, layer_norm, ReLU
import pickle
import configparser
class decoder:
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
        self.layer = kwargs['layer']

        with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{self.layer}_q_weights.pkl', 'rb') as file:
            self.q_weight_matrix = pickle.load(file)
        with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{self.layer}_k_weights.pkl', 'rb') as file:
            self.k_weight_matrix = pickle.load(file)
        with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{self.layer}_v_weights.pkl', 'rb') as file:
            self.v_weight_matrix = pickle.load(file)
        with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{self.layer}_cross_q_weights.pkl', 'rb') as file:
            self.cross_q_weight_matrix = pickle.load(file)
        with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{self.layer}_cross_k_weights.pkl', 'rb') as file:
            self.cross_k_weight_matrix = pickle.load(file)
        with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{self.layer}_cross_v_weights.pkl', 'rb') as file:
            self.cross_v_weight_matrix = pickle.load(file)
        with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{self.layer}_ffn_up_dim_matrix.pkl', 'rb') as file:
            self.ffn_up_dim_matrix = pickle.load(file)
        with open(f'/Users/nancy/Desktop/LLM/parameter/decoder_layer{self.layer}_ffn_down_dim_matrix.pkl', 'rb') as file:
            self.ffn_down_dim_matrix = pickle.load(file)
        
    def __call__(self,encoder_out_matrix : np.ndarray,decoder_input_matrix : np.ndarray, mask_matrix : np.ndarray) -> np.ndarray:
        Q_matrix = np.zeros((self.heads, self.output_seq_length, self.q_weight_dim))
        K_matrix = np.zeros((self.heads, self.output_seq_length, self.k_weight_dim))
        V_matrix = np.zeros((self.heads, self.output_seq_length, self.v_weight_dim))
        attention_matrix_head = np.zeros((self.heads, self.output_seq_length, self.v_weight_dim))
        attention_matrix_out = np.zeros((self.output_seq_length, self.modul_dim))
        for head in range(self.heads):
            Q_matrix[head] = decoder_input_matrix @ self.q_weight_matrix[head]
            K_matrix[head] = decoder_input_matrix @ self.k_weight_matrix[head]
            V_matrix[head] = decoder_input_matrix @ self.v_weight_matrix[head]
            score_matrix_head = Q_matrix[head] @ K_matrix[head].T
            score_matrix_head = softmax(score_matrix_head + mask_matrix)
            attention_matrix_head[head] = score_matrix_head @ V_matrix[head]
            for i in range(self.output_seq_length):
                for j in range(self.v_weight_dim):
                    attention_matrix_out[i][head * self.v_weight_dim + j] = attention_matrix_head[head][i][j]
        add_norm_matrix = layer_norm(decoder_input_matrix + attention_matrix_out)

        cross_Q_matrix = np.zeros((self.heads, self.output_seq_length, self.q_weight_dim))
        cross_K_matrix = np.zeros((self.heads, self.input_seq_length, self.k_weight_dim))
        cross_V_matrix = np.zeros((self.heads, self.input_seq_length, self.v_weight_dim))
        cross_attention_matrix_head = np.zeros((self.heads, self.output_seq_length, self.v_weight_dim))
        cross_attention_matrix_out = np.zeros((self.output_seq_length, self.modul_dim))
        for head in range(self.heads):
            cross_Q_matrix[head] = decoder_input_matrix @ self.cross_q_weight_matrix[head]
            cross_K_matrix[head] = encoder_out_matrix @ self.cross_k_weight_matrix[head]
            cross_V_matrix[head] = encoder_out_matrix @ self.cross_v_weight_matrix[head]
            cross_score_matrix_head = cross_Q_matrix[head] @ cross_K_matrix[head].T
            cross_score_matrix_head = softmax(cross_score_matrix_head + mask_matrix)
            cross_attention_matrix_head[head] = cross_score_matrix_head @ cross_V_matrix[head]
            for i in range(self.output_seq_length):
                for j in range(self.v_weight_dim):
                    cross_attention_matrix_out[i][head * self.v_weight_dim + j] = cross_attention_matrix_head[head][i][j]

        add_norm_matrix = layer_norm(add_norm_matrix + cross_attention_matrix_out)
        ffn_up_matrix = add_norm_matrix @ self.ffn_up_dim_matrix
        ReLU_matrix = ReLU(ffn_up_matrix)
        ffn_down_matrix = ReLU_matrix @ self.ffn_down_dim_matrix
        decoder_output_matrix = layer_norm(add_norm_matrix + ffn_down_matrix)
        return decoder_output_matrix