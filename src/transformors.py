import numpy as np
from function_modul import softmax, padding_matrix, position_mask
from generate_training_data import generate_training_data
from cross_entropy_lost import cross_entropy_loss
from encoder import encoder
from decoder import decoder
import pickle
import configparser
from embedding import embedding
import re

training_mode = False

config = configparser.ConfigParser()
config.read('/Users/nancy/Desktop/LLM/config/config.ini')
modul_dim = int(config['DEFAULT']['MODUL_DIM'])
heads = int(config['DEFAULT']['HEADS'])
q_weight_dim = int(config['DEFAULT']['Q_WEIGHTS_DIM'])
k_weight_dim = int(config['DEFAULT']['K_WEIGHTS_DIM'])
v_weight_dim = int(config['DEFAULT']['V_WEIGHTS_DIM'])
ffn_hidden_dim = int(config['DEFAULT']['FFN_HIDDEN_DIM'])
input_seq_length = int(config['DEFAULT']['INPUT_SEQUENCE_LENGTH'])
output_seq_length = int(config['DEFAULT']['OUTPUT_SEQUENCE_LENGTH'])
encoder_layers = int(config['DEFAULT']['ENCODER_LAYERS'])
decoder_layers = int(config['DEFAULT']['DECODER_LAYERS'])
vocab_size = int(config['DEFAULT']['VOCAB_SIZE'])

with open('/Users/nancy/Desktop/LLM/parameter/position_encoding_matrix.pkl', 'rb') as file:
    position_encoding = pickle.load(file)
with open('/Users/nancy/Desktop/LLM/parameter/linear_matrix.pkl', 'rb') as file:
    linear_matrix = pickle.load(file)
with open('/Users/nancy/Desktop/LLM/parameter/linear_bias.pkl', 'rb') as file:
    bias = pickle.load(file)
with open(f'/Users/nancy/Desktop/LLM/parameter/vocab_index.pkl', 'rb') as file:
    characters_index_dict = pickle.load(file)
with open(f'/Users/nancy/Desktop/LLM/parameter/vocab_onehot.pkl', 'rb') as file:
    one_hot_dict = pickle.load(file)

def transformor(user_input,true_output = None):
    embedded_input = embedding()
    encoder_input_matrix = embedded_input(input_seq_length, list(user_input)) + position_encoding
    for layer in range(encoder_layers):
        encoder_layer = encoder(layer = layer)
        encoder_input_matrix = encoder_layer(encoder_input_matrix, mask_matrix = padding_matrix(input_seq_length,input_seq_length,len(user_input)))

    decoders = []
    for layer in range(decoder_layers):
        decoders.append(decoder(layer = layer))

    decoder_input_list = ['<SOS>','0','0']
    decoder_input_matrix = embedded_input(output_seq_length, decoder_input_list) + position_encoding

    for step in range(output_seq_length):
        for layer in range(decoder_layers):
            decoder_input_matrix = decoders[layer](encoder_input_matrix, decoder_input_matrix, mask_matrix = position_mask(output_seq_length, output_seq_length, step))
        if step == output_seq_length - 1:
            break
        next_char = characters_index_dict[softmax(decoder_input_matrix[step + 1] @ linear_matrix + bias).argmax()]
        decoder_input_list[step + 1] = next_char
        decoder_input_matrix = embedded_input(output_seq_length, decoder_input_list) + position_encoding
        if training_mode == True:
            pred_y = np.array(one_hot_dict[next_char])
            true_y = np.array(one_hot_dict[true_output[step + 1]] if true_output else None)
            print("Step", step + 1, "cross entropy loss: ", cross_entropy_loss(true_y, pred_y))
    print("The result is: ", user_input, '=', ''.join(decoder_input_list[1:]))

if training_mode:
    training_data = generate_training_data().items()
    for key, value in training_data:
        transformor(key,value)
else:    
    while True:
        user_input = input("输入一个一位数的加减乘除法, q为退出: ").replace(' ','')
        if user_input == 'q':
            exit()
        if re.match(r'^\d[\+\-\*/]\d$', user_input) == None:
            print("输入格式错误, 请重新输入.")
        else:
            break
    transformor(user_input)