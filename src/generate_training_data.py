import json

def generate_training_data():
    training_data_dict = {}
    for i in range(10):
        for j in range(10):
            num_str = str(i+j)
            if len(num_str) == 1:
                num_str = '0' + num_str
            char_list = list(num_str)
            training_data_dict[f"{i}+{j}"] = ["<SOS>"] + char_list
            num_str = str(i*j)
            if len(num_str) == 1:
                num_str = '0' + num_str
            char_list = list(num_str)
            training_data_dict[f"{i}*{j}"] = ["<SOS>"] + char_list

    return training_data_dict
