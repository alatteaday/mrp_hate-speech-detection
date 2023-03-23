import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
from tqdm import tqdm
import json

from second_train import get_args_2
from dataset import HateXplainDataset, HateXplainDatasetForBias
from utils import get_device, add_tokens_to_tokenizer, NumpyEncoder
from second_train import evaluate
from second_test_lime import TestLime
from bias_result import get_bias_results
from explain_result import get_explain_results


def test(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)
    
    if args.num_labels == 3:
        test_dataset = HateXplainDataset(args, 'test')
    elif args.num_labels == 2:
        test_dataset = HateXplainDatasetForBias(args, 'test')
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.to(args.device)
    model.config.output_attentions=True

    log = open(os.path.join(args.dir_result, 'test_res_performance.txt'), 'a')

    losses, loss_avg, acc, per_based_scores, time_avg, bias_dict_list, explain_dict_list = evaluate(args, model, test_dataloader, tokenizer)

    print("Loss_avg: {} / min: {} / max: {} | Consumed_time: {}\n".format(loss_avg, min(losses), max(losses), time_avg))
    print("** Performance-based Scores **")
    print("Acc: {} | F1: {} | AUROC: {} \n".format(acc[0], per_based_scores[0], per_based_scores[1]))

    log.write("Checkpoint: {}\n".format(args.model_path))
    log.write("Loss_avg: {} / min: {} / max: {} | Consumed_time: {}\n\n".format(loss_avg, min(losses), max(losses), time_avg))
    log.write("** Performance-based Scores **\n")
    log.write("Acc: {} | F1: {} | AUROC: {} \n".format(acc[0], per_based_scores[0], per_based_scores[1]))
    log.close()

    if args.num_labels == 2:
        print(bias_dict_list[:3])
        with open(args.dir_result + '/for_bias.json', 'w') as f:
            f.write('\n'.join(json.dumps(i,cls=NumpyEncoder) for i in bias_dict_list))

        print("** Bias-based Scores **")
        log = open(os.path.join(args.dir_result, 'test_res_bias.txt'), 'a')
        bias_score_dict, n_targets = get_bias_results(args, {'MODEL': args.dir_result+'/for_bias.json'})

        power_value = -5
        for each_model in bias_score_dict:
            for each_method in bias_score_dict[each_model]:
                temp_value =[]
                for each_community in bias_score_dict[each_model][each_method]:
                    temp_value.append(pow(bias_score_dict[each_model][each_method][each_community], power_value))
                print(each_model, each_method, pow(np.sum(temp_value)/n_targets, 1/power_value))
                log.write("* {}: {}\n".format(each_method, pow(np.sum(temp_value)/n_targets, 1/power_value)))
        log.close()

    if args.num_labels == 3:
        
        with open(args.dir_result + '/for_explain_union.json', 'w') as f:
            f.write('\n'.join(json.dumps(i,cls=NumpyEncoder) for i in explain_dict_list))

        print('[*] Start LIME test')
        lime_tester = TestLime(args)
        lime_dict_list = lime_tester.test(args)  # This could take a little long time
        with open(args.dir_result + '/for_explain_lime.json', 'w') as f:
            f.write('\n'.join(json.dumps(i,cls=NumpyEncoder) for i in lime_dict_list))
        
        get_explain_results(args)  # The test_res_explain.txt file will be written


if __name__ == '__main__':
    args = get_args_2()
    args.device = get_device()
    
    args.test = True
    args.batch_size = 1
    args.n_eval = 0

    if 'ncls2' in args.model_path.split('/')[7]:
        args.num_labels = 2
    elif 'ncls3' in args.model_path.split('/')[7]:
        args.num_labels = 3
    else:
        print("[!] should check num_labels your checkpoint was trained for and set the right value")
        exit()
    
    args.dir_result = '/'.join(args.model_path.split('/')[:7])
    assert args.dir_result

    print("Checkpoint: ", args.model_path)
    print("Result path: ", args.dir_result)

    test(args)