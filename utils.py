import torch
from torch.utils.data import WeightedRandomSampler
import os
import numpy as np
import json


def get_device():
    if torch.cuda.is_available():
        print("device = cuda")
        return torch.device('cuda')
    else:
        print("device = cpu")
        return torch.device('cpu')


def add_tokens_to_tokenizer(args, tokenizer):
    special_tokens_dict = {'additional_special_tokens': 
                            ['<user>', '<number>']}  # hatexplain
    n_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # print(tokenizer.all_special_tokens) 
    # print(tokenizer.all_special_ids)
    
    return tokenizer


class GetLossAverage(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()  # type -> int
        v = v.data.sum().item()  # type -> float
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def aver(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def get_weighted_sampler(args, dataset):
    cls_count = dataset.label_count
    clses = list(range(len(dataset.label_list)))
    total_count = len(dataset)

    class_weights = [total_count / cls_count[i] for i in range(len(cls_count))] 
    weights = [class_weights[clses[i]] for i in range(int(total_count))] 
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(total_count))

    return sampler 
    

def save_checkpoint(args, losses, model_state, trained_model):
    # checkpoint = {
    #     'args': args,
    #     'model_state': model_state,
    #     'optimizer_state': optimizer_state
    # }
    file_name = args.exp_name + '.ckpt'
    trained_model.save_pretrained(save_directory=os.path.join(args.dir_result, file_name))

    args.waiting += 1
    if losses[-1] <= min(losses):
        # print(losses)
        args.waiting = 0
        file_name = 'BEST_' + file_name
        trained_model.save_pretrained(save_directory=os.path.join(args.dir_result, file_name))
        
        if args.intermediate == 'mrp':
            # Save the embedding layer params
            emb_file_name = args.exp_name + '_emb.ckpt'
            torch.save(model_state.state_dict(), os.path.join(args.dir_result, emb_file_name))

        print("[!] The best checkpoint is updated")


def get_token_rationale(tokenizer, text, rationale, id):
    text_token = tokenizer.tokenize(' '.join(text))
    assert len(text) == len(rationale), '[!] len(text) != len(rationale) | {} != {}\n{}\n{}'.format(len(text), len(rationale), text, rationale)
    
    rat_token = []
    for t, r in zip(text, rationale):
        token = tokenizer.tokenize(t)
        rat_token += [r]*len(token)

    assert len(text_token) == len(rat_token), "#token != #target rationales of {}".format(id)
    return rat_token


def make_final_rationale(id, rats_list):
    rats_np = np.array(rats_list)
    sum_np = rats_np.sum(axis=0)
    try:
        avg_np = sum_np / len(rats_list)
        avg_rat = avg_np.tolist()
        bi_rat = []
        for el in avg_rat:
            if el >= 0.5:
                bi_rat.append(1)
            else:
                bi_rat.append(0)
    except:
        print(id)
    
    return avg_rat, bi_rat


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)