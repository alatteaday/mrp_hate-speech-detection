import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch_optimizer as optim

from transformers import BertTokenizer, BertForTokenClassification, BertForSequenceClassification, BertForMaskedLM
import numpy as np
import argparse
import os
import sys
import gc
from tqdm import tqdm
import time 
from datetime import datetime
import random

from dataset import HateXplainDataset, HateXplainDatasetForBias
from utils import get_device, GetLossAverage, save_checkpoint, add_tokens_to_tokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_args_2():
    parser = argparse.ArgumentParser(description='')

    # DATASET
    parser.add_argument('--dir_hatexplain', type=str, default="./dataset", help='the root directiory of the dataset')

    # PRETRAINED MODEL
    model_choices = ['bert-base-uncased']
    parser.add_argument('--pretrained_model', default='bert-base-uncased', choices=model_choices, help='a pretrained bert model to use')

    # PRE-FINETUNED MODELS
    # model_paths = [
    #     '## pre_finetuned_model path ##',
    #     ]
    # parser.add_argument('--pre_finetuned_model', choices=model_paths, default=model_paths[-1])    
    parser.add_argument('-pf_m', '--pre_finetuned_model', required=True)

    # TRAIN
    parser.add_argument('--num_labels', choices=['2', '3'], default='2', required=True, help="3 = [hatespeech/offensive/normal], 2 = [toxic/nontoxic]")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--val_int', type=int, default=745)  
    parser.add_argument('--patience', type=int, default=3)

    ## Explainability based metrics
    parser.add_argument('--top_k', default=5, help='the top num of attention values to evaluate on explainable metrics')
    parser.add_argument('--lime_n_sample', default=100, help='the num of samples for lime explainer')

    args = parser.parse_args()  
    return args 


def load_model_train(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels)

    if 'mlm' in args.pre_finetuned_model:
        pre_finetuned_model = BertForMaskedLM.from_pretrained(args.pre_finetuned_model)
    else:
        pre_finetuned_model = BertForTokenClassification.from_pretrained(args.pre_finetuned_model)

    model_state = model.state_dict()
    finetuned_state = pre_finetuned_model.state_dict()
    
    # Initialize condition layer randomly 
    filtered_pretrained_state = {}
    for (k1, v1), (k2, v2) in zip(model_state.items(), finetuned_state.items()):
        if v1.size() == v2.size():
            filtered_pretrained_state[k1] = v2
        else:
            filtered_pretrained_state[k1] = v1

    model_state.update(filtered_pretrained_state)
    model.load_state_dict(model_state, strict=True)

    return model, tokenizer


def get_pred_cls(logits):
    probs = F.softmax(logits, dim=1)
    #labels = labels.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()
    max_probs = np.max(probs, axis=1).tolist()
    probs = probs.tolist()
    pred_clses = []
    for m, p in zip(max_probs, probs):
        pred_clses.append(p.index(m))
    
    return probs, pred_clses


def get_dict_for_bias(post_id, label, pred_cls, pred_prob):
    bias_dict = {}
    bias_dict['annotation_id'] = post_id
    
    if label == 1:
        bias_dict['ground_truth'] = 'toxic'
    elif label == 0:
        bias_dict['ground_truth'] = 'non-toxic'
    else:
        print('[!] gt error of test of classification by 2 | should be 1 or 0')
    
    if pred_cls == 1:
        bias_dict['classification'] = 'toxic'
    elif pred_cls == 0:
        bias_dict['classification'] = 'non-toxic'
    else:
        print('[!] prediction error of test of classification by 2 | should be 1 or 0')

    bias_dict['classification_scores'] = {'non-toxic': pred_prob[0], 'toxic': pred_prob[1]}

    return bias_dict


def get_dict_for_explain(args, model, tokenizer, in_tensor, gts_tensor, attns, id, pred_cls, pred_prob):
    explain_dict = {}
    explain_dict["annotation_id"] = id
    explain_dict["classification"] = pred_cls 
    explain_dict["classification_scores"] = {"hatespeech": pred_prob[0], "normal": pred_prob[1], "offensive": pred_prob[2]}
    
    attns = np.mean(attns[:,:,0,:].detach().cpu().numpy(),axis=1).tolist()[0]
    top_indices = sorted(range(len(attns)), key=lambda i: attns[i])[-args.top_k:]  # including start/end token ?
    temp_hard_rationale = []
    for ind in top_indices:
        temp_hard_rationale.append({'end_token':ind+1, 'start_token':ind})

    gt = gts_tensor.detach().cpu().tolist()[0]
    
    explain_dict["rationales"] = [{"docid": id, 
                                "hard_rationale_predictions": temp_hard_rationale, 
                                "soft_rationale_predictions": attns,
                                #"soft_sentence_predictions": [1.0],
                                #"truth": gts_tensor.detach().cpu().tolist()[0]}, 
                                "truth": gt, 
                                }]

    in_ids = in_tensor['input_ids'].detach().cpu().tolist()[0]
    
    in_ids_suf, in_ids_com = [], []
    for i in range(len(attns)):
        if i in top_indices:
            in_ids_suf.append(in_ids[i])
        else:
            in_ids_com.append(in_ids[i])

    suf_tokens = tokenizer.convert_ids_to_tokens(in_ids_suf)
    suf_text = tokenizer.convert_tokens_to_string(suf_tokens)
    suf_text = suf_text.lower()
    in_ids_suf = tokenizer.encode(suf_text)

    in_ids_com = [101]+in_ids_com[1:-1]+[102]
  
    in_ids_suf = torch.tensor(in_ids_suf)
    in_ids_suf = torch.unsqueeze(in_ids_suf, 0).to(args.device)
    in_ids_com = torch.tensor(in_ids_com)
    in_ids_com = torch.unsqueeze(in_ids_com, 0).to(args.device)
    
    in_tensor_suf = {'input_ids': in_ids_suf, 
                    'token_type_ids': torch.zeros(in_ids_suf.shape, dtype=torch.int).to(args.device), 
                    'attention_mask': torch.ones(in_ids_suf.shape, dtype=torch.int).to(args.device)}
    in_tensor_com = {'input_ids': in_ids_com, 
                    'token_type_ids': torch.zeros(in_ids_com.shape, dtype=torch.int).to(args.device), 
                    'attention_mask': torch.ones(in_ids_com.shape, dtype=torch.int).to(args.device)}
    
    out_tensor_suf = model(**in_tensor_suf, labels=gts_tensor)  
    prob_suf = F.softmax(out_tensor_suf.logits, dim=1).detach().cpu().tolist()[0]
    try:
        out_tensor_com = model(**in_tensor_com, labels=gts_tensor) 
    except:
        print(id)

    prob_com = F.softmax(out_tensor_com.logits, dim=1).detach().cpu().tolist()[0]
    
    explain_dict['sufficiency_classification_scores'] = {"hatespeech": prob_suf[0], "normal": prob_suf[1], "offensive": prob_suf[2]}
    explain_dict['comprehensiveness_classification_scores'] = {"hatespeech": prob_com[0], "normal": prob_com[1], "offensive": prob_com[2]}
    
    return explain_dict


def evaluate(args, model, dataloader, tokenizer):
    losses = []
    consumed_time = 0
    total_pred_clses, total_gt_clses, total_probs = [], [], []

    bias_dict_list, explain_dict_list = [], []
    label_dict = {0:'hatespeech', 2:'offensive', 1:'normal'}

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="EVAL | # {}".format(args.n_eval), mininterval=0.01)):
            texts, labels, ids = batch[0], batch[1], batch[2]
           
            in_tensor = tokenizer(texts, return_tensors='pt', padding=True)
            in_tensor = in_tensor.to(args.device)
            gts_tensor = labels.to(args.device)

            start_time = time.time()
            out_tensor = model(**in_tensor, labels=gts_tensor)
            consumed_time += time.time() - start_time 

            loss = out_tensor.loss
            logits = out_tensor.logits
            attns = out_tensor.attentions[11]

            losses.append(loss.item())
            
            probs, pred_clses = get_pred_cls(logits)
            labels_list = labels.tolist()
            
            total_gt_clses += labels_list
            total_pred_clses += pred_clses
            total_probs += probs

            if args.num_labels == 2 and args.test:
                bias_dict = get_dict_for_bias(ids[0], labels_list[0], pred_clses[0], probs[0])
                bias_dict_list.append(bias_dict)
            
            if args.num_labels == 3 and args.test:
                if labels_list[0] == 1:  # if label is 'normal'
                    continue                     
                explain_dict = get_dict_for_explain(args, model, tokenizer, in_tensor, gts_tensor, attns, ids[0], label_dict[pred_clses[0]], probs[0])
                if explain_dict == None:
                    continue
                explain_dict_list.append(explain_dict)
                    
    time_avg = consumed_time / len(dataloader)
    loss_avg = [sum(losses) / len(dataloader)]
    acc = [accuracy_score(total_gt_clses, total_pred_clses)]
    f1 = f1_score(total_gt_clses, total_pred_clses, average='macro')
    if args.num_labels == 2:
        auroc = -1
    else:  # args.num_labels == 3
        auroc = roc_auc_score(total_gt_clses, total_probs, multi_class='ovo')  
    per_based_scores = [f1, auroc]
    
    return losses, loss_avg, acc, per_based_scores, time_avg, bias_dict_list, explain_dict_list 


def train(args):
    model, tokenizer = load_model_train(args)
    model.resize_token_embeddings(len(tokenizer))
   
    if args.num_labels == 3:
        train_dataset = HateXplainDataset(args, mode='train')
        val_dataset= HateXplainDataset(args, mode='val')
    else:  # args.num_labels == 2
        train_dataset = HateXplainDatasetForBias(args, mode='train')
        val_dataset = HateXplainDatasetForBias(args, mode='val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    get_tr_loss = GetLossAverage()
    optimizer = optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    model.config.output_attentions=True

    model.to(args.device)
    model.train()
    
    log = open(os.path.join(args.dir_result, 'train_res.txt'), 'a')
    tr_losses, val_losses, val_f1s, val_accs = [], [], [], []
    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(train_dataloader, desc="TRAIN | Epoch: {}".format(epoch), mininterval=0.01)):  # data: (post_words, target_rat, post_id)
            texts, labels, ids = batch[0], batch[1], batch[2]
    
            in_tensor = tokenizer(texts, return_tensors='pt', padding=True)
            in_tensor = in_tensor.to(args.device)
            gts_tensor = labels.to(args.device)

            optimizer.zero_grad()

            out_tensor = model(**in_tensor, labels=gts_tensor)  
            loss = out_tensor.loss
            
            loss.backward()
            optimizer.step()
            get_tr_loss.add(loss)

            # Validation 
            if i==0 or (i+1) % args.val_int == 0:
                _, loss_avg, acc_avg, per_based_scores, time_avg, _, _ = evaluate(args, model, val_dataloader, tokenizer)
                
                args.n_eval += 1
                model.train()

                val_losses.append(loss_avg[0])
                val_accs.append(acc_avg[0])
                val_f1s.append(per_based_scores[0])

                tr_loss = get_tr_loss.aver()
                tr_losses.append(tr_loss) 
                get_tr_loss.reset()  
               
                print("[Epoch {} | Val #{}]".format(epoch, args.n_eval))
                print("* tr_loss: {}".format(tr_loss))
                print("* val_loss: {} | val_consumed_time: {}".format(loss_avg[0], time_avg))
                print("* acc: {} | f1: {} | AUROC: {}\n".format(acc_avg[0], per_based_scores[0], per_based_scores[1]))
                
                log.write("[Epoch {} | Val #{}]\n".format(epoch, args.n_eval))
                log.write("* tr_loss: {}\n".format(tr_loss))
                log.write("* val_loss: {} | val_consumed_time: {}\n".format(loss_avg[0], time_avg))
                log.write("* acc: {} | f1: {} | AUROC: {}\n\n".format(acc_avg[0], per_based_scores[0], per_based_scores[1]))
                
                save_checkpoint(args, val_losses, None, model)
        
            if args.waiting > args.patience:
                print("[!] Early stopping")
                break
        if args.waiting > args.patience:
            break 

    log.close()
            
    
if __name__ == '__main__':
    args = get_args_2()
    
    args.test = False
    args.intermediate = False
    args.device = get_device()
    
    lm = '-'.join(args.pretrained_model.split('-')[:-1])  # ex) 'bert-base' - check 'first_train.py'
    # assert lm in args.pre_finetuned_model, "[!] check | the two models are supposed to have the same size"

    print("Pre-finetuned model path: ", args.pre_finetuned_model)

    now = datetime.now()
    args.exp_date = now.strftime('%m%d-%H%M')
    args.exp_name = args.exp_date + "_" + lm + "_" + 'ncls' + str(args.num_labels) +  "_"  + str(args.lr) + "_" + str(args.batch_size) + "_" + str(args.val_int)
    folder_name = '.'.join(args.pre_finetuned_model.split('/')[-1].split('.')[:-1])
    dir_result = os.path.join("finetune_2nd", folder_name, args.exp_name)

    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    print("Checkpoint path: ", dir_result)
    args.dir_result = dir_result

    args.waiting = 0
    args.n_eval = 0
    args.num_labels = int(args.num_labels)

    gc.collect()
    torch.cuda.empty_cache()

    train(args)

    
    
    




