from torch.utils.data import Dataset

from transformers import BertTokenizer
import copy
import os
import json
import numpy as np
import emoji

from utils import add_tokens_to_tokenizer, get_token_rationale


class HateXplainDataset(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'val', 'test'], "the mode should be [train/val/test]"
        
        data_root = args.dir_hatexplain
        data_dir = os.path.join(data_root, 'hatexplain_thr_div.json')
        with open(data_dir, 'r') as f:
            dataset = json.load(f)

        self.label_list = ['hatespeech', 'normal', 'offensive']
        self.label_count = [0, 0, 0]
            
        if mode == 'train':
            self.dataset = dataset['train']
            for d in self.dataset:
                for i in range(len(self.label_list)):
                    if d['final_label'] == self.label_list[i]:
                        self.label_count[i] += 1         
        elif mode == 'val':
            self.dataset = dataset['val']
        else:  # 'test'
            self.dataset = dataset['test']

        if args.intermediate:  
            rm_idxs = []
            for idx, d in enumerate(self.dataset):
                if 1 not in d['final_rationale'] and d['final_label'] in ('offensive', 'hatespeech'):
                    rm_idxs.append(idx)
            rm_idxs.sort(reverse=True)
            for j in rm_idxs:
                del self.dataset[j]

        self.mode = mode
        self.intermediate = args.intermediate

        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        self.tokenizer = add_tokens_to_tokenizer(args, tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id = self.dataset[idx]['post_id']
        text = ' '.join(self.dataset[idx]['text'])
        text = emoji.demojize(text, use_aliases=True)
        label = self.dataset[idx]['final_label']
        cls_num = self.label_list.index(label)

        if self.intermediate:
            fin_rat = self.dataset[idx]['final_rationale']
            fin_rat_token = get_token_rationale(self.tokenizer, copy.deepcopy(text.split(' ')), copy.deepcopy(fin_rat), copy.deepcopy(id))

            tmp = []
            for f in fin_rat_token:
                tmp.append(str(f))
            fin_rat_str = ','.join(tmp)
            return (text, cls_num, fin_rat_str)
            
        elif self.intermediate == False:  # hate speech detection
            return (text, cls_num, id)

        else:  
            return ()

            
class HateXplainDatasetForBias(Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'val', 'test'], "mode should be [train/val/test]"
        
        data_root = args.dir_hatexplain
        data_dir = os.path.join(data_root, 'hatexplain_two_div.json')
        with open(data_dir, 'r') as f:
            dataset = json.load(f)

        self.label_list = ['non-toxic', 'toxic']
        self.label_count = [0, 0]

        if mode == 'train':
            self.dataset = dataset['train']
            for d in self.dataset:
                if d['final_label'] == self.label_list[0]:
                    self.label_count[0] += 1
                elif d['final_label'] == self.label_list[1]:
                    self.label_count[1] += 1
                else:
                    print("[!] exceptional label ", d['final_label'])
                    return
        elif mode == 'val':
            self.dataset = dataset['val']
        else:  # 'test'
            self.dataset = dataset['test']
        self.mode = mode
        self.intermediate = args.intermediate
        assert self.intermediate == False
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        post_id = self.dataset[idx]['post_id']
        text = ' '.join(self.dataset[idx]['text'])
        text = emoji.demojize(text)
        label = self.dataset[idx]['final_label']
        cls_num = self.label_list.index(label)
        
        return (text, cls_num, post_id)


    
    
    

    
    
    
    

        
