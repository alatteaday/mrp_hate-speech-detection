import json
import more_itertools as mit
import os
import emoji
import copy
import argparse
from transformers import BertTokenizer

from utils import get_device, get_token_rationale, add_tokens_to_tokenizer


# Reconstruct dataset
def get_dataset_for_explain(data_dir, tokenizer, method):
    with open(data_dir, 'r') as f:
        dataset = json.load(f)

    output_list = []
    for div in ['train', 'val', 'test']:
        for data in dataset[div]:
            post_id = data['post_id']
            label_list = [data['label1'], data['label2'], data['label3']]
            
            text = emoji.demojize(' '.join(data['text']), use_aliases=True)
            encoded_text = tokenizer(text)['input_ids']
            
            if method == 'union':
                bi_rat = [any(each) for each in zip(*data['rationales'])]
                bi_rat = [int(each) for each in bi_rat]
            
            bi_rat_token = get_token_rationale(tokenizer, copy.deepcopy(text.split()), bi_rat, copy.deepcopy(post_id))
            bi_rat_token = [0]+bi_rat_token+[0]

            assert len(bi_rat_token) == len(encoded_text)
            if 1 not in bi_rat_token and data['final_label'] in ('offensive', 'hatespeech'):
                continue
            output_list.append([post_id, data['final_label'], encoded_text, bi_rat_token, label_list])
    
    return output_list


# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, majority_label, anno_text, explanations):
    output = []
    
    indexes = sorted([i for i, each in enumerate(explanations) if each==1])
    span_list = list(find_ranges(indexes))
    # if span_list == []:
    #     print(post_id, majority_label, explanations, indexes)
    #     a += 1

    for each in span_list:
        if type(each)== int:
            start = each
            end = each+1
        elif len(each) == 2:
            start = each[0]
            end = each[1]+1
        else:
            print('error')

        output.append({"docid":post_id, 
              "end_sentence": -1, 
              "end_token": end, 
              "start_sentence": -1, 
              "start_token": start, 
              "text": ' '.join([str(x) for x in anno_text[start:end]])})
    return output


# To use the metrices defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, id_division):  
    final_output = []
    
    if save_split:
        train_fp = open(os.path.join(save_path, 'train.jsonl'), 'w')
        val_fp = open(os.path.join(save_path, 'val.jsonl'), 'w')
        test_fp = open(os.path.join(save_path, 'test.jsonl'), 'w')
            
    for tcount, eachrow in enumerate(dataset):
        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]
        majority_label = eachrow[1]
        
        if majority_label=='normal':
            continue
        
        all_labels = eachrow[4]
        final_explanation = eachrow[3]
        
        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, majority_label, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)
        
        if save_split:
            docs_dir = os.path.join(save_path, 'docs')
            if not os.path.exists(docs_dir):
                os.makedirs(docs_dir)
            
            with open(os.path.join(docs_dir, post_id), 'w+') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))
            
            if post_id in id_division['train']:
                train_fp.write(json.dumps(temp)+'\n')
            
            elif post_id in id_division['val']:
                val_fp.write(json.dumps(temp)+'\n')
            
            elif post_id in id_division['test']:
                test_fp.write(json.dumps(temp)+'\n')
            else:
                print(post_id)
    
    if save_split:
        train_fp.close()
        val_fp.close()
        test_fp.close()
        
    return final_output

def get_explain_results(args):
    data_dir = os.path.join(args.dir_hatexplain, 'hatexplain_thr_div.json')
    save_path = './metrics/'  # The dataset in Eraser format will be stored here
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # If should get dataset in Eraser format
    method = 'union'
    get_eval_data = False  # turn it to True
    if get_eval_data:  
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        tokenizer = add_tokens_to_tokenizer(args, tokenizer)
            
        recon_data = get_dataset_for_explain(data_dir, tokenizer, method)
        print(len(recon_data))

        with open(os.path.join(args.dir_hatexplain, 'post_id_divisions.json')) as fp:
            id_division = json.load(fp)

        save_split = True
        _ = convert_to_eraser_format(recon_data, method, save_split, save_path, id_division)
        print('done')
    
    log = open(os.path.join(args.dir_result, 'test_res_explain.txt'), 'a')

    # Attn
    result_file_dir = args.dir_result + '/for_explain_union.json'
    score_file_dir = os.path.join('/'.join(result_file_dir.split('/')[:-1]), 'explain_result.json')
    
    os.chdir('./eraserbenchmark')
    os.system('PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --strict --data_dir {} --results {} --score_file {}'.format(save_path, result_file_dir, score_file_dir))
    
    with open(score_file_dir) as fp:
        output_data = json.load(fp)

    print("** Explainability-based Scores **")
    print("[Attn]")
    print('Plausibility')
    print('* IOU F1 :', output_data['iou_scores'][0]['macro']['f1'])
    print('* Token F1 :', output_data['token_prf']['instance_macro']['f1'])
    print('* AUPRC :', output_data['token_soft_metrics']['auprc'])

    print('\nFaithfulness')
    print('* Comprehensiveness :', output_data['classification_scores']['comprehensiveness'])
    print('* Sufficiency', output_data['classification_scores']['sufficiency'])

    log.write("** Explainability-based Scores **\n")
    log.write("[Attn]\n")
    log.write('Plausibility\n')
    log.write('IOU F1: {}\n'.format(output_data['iou_scores'][0]['macro']['f1']))
    log.write('Token F1: {}\n'.format(output_data['token_prf']['instance_macro']['f1']))
    log.write('AUPRC: {}\n'.format(output_data['token_soft_metrics']['auprc']))

    log.write('\nFaithfulness\n')
    log.write('Comprehensiveness: {}\n'.format(output_data['classification_scores']['comprehensiveness']))
    log.write('Sufficiency: {}\n'.format(output_data['classification_scores']['sufficiency']))
    
    # LIME
    result_file_dir = args.dir_result + '/for_explain_lime.json'
    score_file_dir = os.path.join('/'.join(result_file_dir.split('/')[:-1]), 'explain_result_lime.json')

    os.system('PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --strict --data_dir {} --results {} --score_file {}'.format(save_path, result_file_dir, score_file_dir))
    
    with open(score_file_dir) as fp:
        output_data = json.load(fp)

    print("\n[LIME]")
    print('Plausibility')
    print('* IOU F1 :', output_data['iou_scores'][0]['macro']['f1'])
    print('* Token F1 :', output_data['token_prf']['instance_macro']['f1'])
    print('* AUPRC :', output_data['token_soft_metrics']['auprc'])

    print('\nFaithfulness')
    print('* Comprehensiveness :', output_data['classification_scores']['comprehensiveness'])
    print('* Sufficiency', output_data['classification_scores']['sufficiency'])

    log.write("\n[LIME]\n")
    log.write('Plausibility\n')
    log.write('IOU F1: {}\n'.format(output_data['iou_scores'][0]['macro']['f1']))
    log.write('Token F1: {}\n'.format(output_data['token_prf']['instance_macro']['f1']))
    log.write('AUPRC: {}\n'.format(output_data['token_soft_metrics']['auprc']))

    log.write('\nFaithfulness\n')
    log.write('Comprehensiveness: {}\n'.format(output_data['classification_scores']['comprehensiveness']))
    log.write('Sufficiency: {}\n'.format(output_data['classification_scores']['sufficiency']))

    log.close()


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(description='')

    # DATASET
    parser.add_argument('--dir_hatexplain', type=str, default="./dataset", help='the root directiory of the dataset')

    # PRETRAINED MODEL
    model_choices = ['bert-base-uncased']
    parser.add_argument('--pretrained_model', default='bert-base-uncased', choices=model_choices, help='a pretrained bert model to use')
    
    # TEST
    parser.add_argument('-m', '--model_path', type=str, required=True, help='the checkpoint path to test')  
    
    ## Explainability based metrics
    parser.add_argument('--top_k', default=5, help='the top num of attention values to evaluate on explainable metrics')
    parser.add_argument('--lime_n_sample', default=100, help='the num of samples for lime explainer')

    args = parser.parse_args()  

    args.test = True
    args.intermediate = False
    args.batch_size = 1
    args.n_eval = 0
    args.device = get_device()

    args.num_labels = 3
    
    args.dir_result = '/'.join(args.model_path.split('/')[:-1])
    assert args.dir_result

    get_explain_results(args)
    """
