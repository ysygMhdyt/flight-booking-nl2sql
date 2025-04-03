import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small', extra_ids=1, additional_special_tokens=['<extra_id_0>'])
        self.data = self.process_data(data_folder, split, self.tokenizer)
        self.split = split

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        if split != "test":
            nl_file_path = os.path.join(data_folder, f"{split}.nl")
            sql_file_path = os.path.join(data_folder, f"{split}.sql")

            data = []
            with open(nl_file_path, 'r', encoding='utf-8') as nl_file, open(sql_file_path, 'r', encoding='utf-8') as sql_file:
                for nl, sql in zip(nl_file, sql_file):
                    nl = nl.strip()
                    sql = sql.strip()
                    encoder_input = tokenizer(nl, return_tensors='pt')
                    decoder_input = tokenizer(f"<extra_id_0> {sql}", return_tensors='pt')
                    data.append((encoder_input, decoder_input))
            return data
        
        elif split == "test":
            data = []
            with open(f'{data_folder}/{split}.nl', 'r') as nl_file:
                for line in nl_file:
                    input_text = line.strip()
                    encoder_input = tokenizer(input_text, return_tensors='pt')
                    data.append((encoder_input, None))
            return data
    
    def __len__(self):
        # TODO
        return len(self.data)

    def __getitem__(self, idx):
        # TODO
        encoder_input, decoder_input = self.data[idx]
        if self.split == "test":
            return encoder_input['input_ids'].squeeze(), encoder_input['attention_mask'].squeeze()
        
        return (encoder_input['input_ids'].squeeze(), encoder_input['attention_mask'].squeeze(),
                decoder_input['input_ids'].squeeze(), decoder_input['attention_mask'].squeeze())

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids, encoder_mask, decoder_ids, decoder_mask = zip(*batch)
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=PAD_IDX)
    decoder_ids = pad_sequence(decoder_ids, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = decoder_ids[:, :-1]
    decoder_targets = decoder_ids[:, 1:].clone()

    bos_token_id = 32099
    initial_decoder_inputs = decoder_inputs.new_full((decoder_inputs.size(0), 1), bos_token_id)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids, encoder_mask = zip(*batch)
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=PAD_IDX)
    bos_token_id = 32099
    initial_decoder_inputs = encoder_ids.new_full((encoder_ids.size(0), 1), bos_token_id)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    print("Data loaded successfully!")
    return train_loader, dev_loader, test_loader

load_t5_data(2, 2)

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x