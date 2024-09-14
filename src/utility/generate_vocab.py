import numpy as np
import configparser
import pickle

config = configparser.ConfigParser()
config.read('/Users/nancy/Desktop/LLM/config/config.ini')
modul_dim = int(config['DEFAULT']['MODUL_DIM'])
vocab_size = int(config['DEFAULT']['VOCAB_SIZE'])
characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-', '*', '/', '<SOS>']
embedding_dict = {char: np.random.uniform(-1,1,modul_dim) for char in characters}
one_hot = np.eye(vocab_size)
one_hot_dict = {}
for i in range(vocab_size):
    one_hot_dict[characters[i]] = one_hot[i]
characters_index_dict = {i : char for i, char in enumerate(characters)}
with open(f'/Users/nancy/Desktop/LLM/parameter/vocab_embedding.pkl', 'wb') as file:
    pickle.dump(embedding_dict, file)
with open(f'/Users/nancy/Desktop/LLM/parameter/vocab_onehot.pkl', 'wb') as file:
    pickle.dump(one_hot_dict, file)
with open(f'/Users/nancy/Desktop/LLM/parameter/vocab_index.pkl', 'wb') as file:
    pickle.dump(characters_index_dict, file)

print(embedding_dict) 
print(one_hot_dict)
print(characters_index_dict)

print("Vocab onehot and embedding generated.")