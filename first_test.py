import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForTokenClassification
import os
from sklearn.preprocessing import MultiLabelBinarizer

from first_train import get_args_1
from dataset import HateXplainDataset
from utils import get_device, add_tokens_to_tokenizer
from first_train import evaluate


def test(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    model = BertForTokenClassification.from_pretrained(args.model_path)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)
    
    test_dataset = HateXplainDataset(args, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    mlb = MultiLabelBinarizer()
    model.to(args.device)
    log = open(os.path.join(args.dir_result, 'test_res.txt'), 'a')
    
    losses, loss_avg, time_avg, acc, f1 = evaluate(args, model, test_dataloader, tokenizer, None, mlb)

    print("Checkpoint: ", args.model_path)
    print("Loss_avg: {} / min: {} / max: {} | Consumed_time: {}".format(loss_avg, min(losses), max(losses), time_avg))
    print("Acc: {} | F1: {} \n".format(acc[0], f1[0]))

    log.write("Checkpoint: {} \n".format(args.model_path))
    log.write("Loss_avg: {} / min: {} / max: {} | Consumed_time: {} \n".format(loss_avg, min(losses), max(losses), time_avg))
    log.write("Acc: {} | F1: {} \n".format(acc[0], f1[0]))

    log.close()


if __name__ == '__main__':
    args = get_args_1()
    args.test = True
    args.intermediate = 'rp'
    args.device = get_device()
    #args.device = 'cpu'
    args.batch_size = 1
    args.n_eval = 0

    args.dir_result = '/'.join(args.model_path.split('/')[:6])
    print("Checkpoint path: ", args.model_path)
    print("Result path: ", args.dir_result)

    test(args)





           