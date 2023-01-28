import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
import os
from tqdm import tqdm

from dataset import HateXplainDataset
from utils import get_device, add_tokens_to_tokenizer
# from second_test import get_args_2, NumpyEncoder


class TestLime():
    def __init__(self, args):
        self.args = args
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels)
        tokenizer = add_tokens_to_tokenizer(args, tokenizer)
        
        self.tokenizer = tokenizer
        model.to(args.device)
        model.eval()
        self.model = model

         # Define dataloader
        test_dataset = HateXplainDataset(args, 'test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        self.dataloader = test_dataloader

        self.explainer = LimeTextExplainer(class_names=['hatespeech', 'normal', 'offensive'], split_expression='\s+', random_state=333, bow=False)
        self.label_dict = {0:'hatespeech', 2:'offensive', 1:'normal'}

    def get_prob(self, texts):  # input -> list
        probs_list = []
        with torch.no_grad():
            for text in texts:
                in_tensor = self.tokenizer(text, return_tensors='pt', padding=True)
                in_tensor = in_tensor.to(self.args.device)
                
                out_tensor = self.model(**in_tensor)  # [batch_size * sequence_length, num_labels]
                logits = out_tensor.logits

                probs = F.softmax(logits, dim=1)
                probs = probs.squeeze(0)
                probs = probs.detach().cpu().numpy()
                probs_list.append(probs)

        return np.array(probs_list)

    def test(self, args):
        lime_dict_list = []
        for i, batch in enumerate(tqdm(self.dataloader, desc="EVAL | # {}".format(args.n_eval), mininterval=0.01)):
            texts, labels, ids = batch[0], batch[1], batch[2]
            label = labels[0]
            if label == 1:
                continue 
            
            exp = self.explainer.explain_instance(texts[0], self.get_prob, num_features=6, top_labels=3, num_samples=args.lime_n_sample)
            
            temp = {}
            pred_id = np.argmax(exp.predict_proba)
            pred_cls = self.label_dict[pred_id]
            gt_cls = label
            temp["annotation_id"] = ids[0]
            temp["classification"] = pred_cls
            temp["classification_scores"] = {"hatespeech": exp.predict_proba[0], "normal": exp.predict_proba[1], "offensive": exp.predict_proba[2]}

            attention = [0]*len(texts[0].split(" "))
            exp_res = exp.as_map()[pred_id]
            for e in exp_res:
                if e[1] > 0:
                    attention[e[0]]=e[1]

            final_explanation = [0]
            tokens = texts[0].split(" ")
            for i in range(len(tokens)):
                temp_tokens = self.tokenizer.encode(tokens[i],add_special_tokens=False)
                for j in range(len(temp_tokens)):
                     final_explanation.append(attention[i])
            final_explanation.append(0)
            attention = final_explanation

            #assert(len(attention) == len(row['Attention']))
            topk_indices = sorted(range(len(attention)), key=lambda i: attention[i])[-args.top_k:]

            temp_hard_rationales=[]
            for ind in topk_indices:
                temp_hard_rationales.append({'end_token':ind+1, 'start_token':ind})

            temp["rationales"] = [{"docid": ids[0], 
                                "hard_rationale_predictions": temp_hard_rationales, 
                                "soft_rationale_predictions": attention,
                                #"soft_sentence_predictions":[1.0],
                                "truth": 0}]

            in_ids = self.tokenizer.encode(texts[0])
    
            in_ids_suf, in_ids_com = [], []
            for i in range(len(attention)):
                if i in topk_indices:
                    in_ids_suf.append(in_ids[i])
                else:
                    in_ids_com.append(in_ids[i])
            
            suf_tokens = self.tokenizer.convert_ids_to_tokens(in_ids_suf)
            suf_text = self.tokenizer.convert_tokens_to_string(suf_tokens)
            suf_text = suf_text.lower()
            
            com_tokens = self.tokenizer.convert_ids_to_tokens(in_ids_com[1:-1])
            com_text = self.tokenizer.convert_tokens_to_string(com_tokens)

            suf_probs = self.get_prob([suf_text])
            com_probs = self.get_prob([com_text])

            temp["sufficiency_classification_scores"] = {"hatespeech": suf_probs[0][0], "normal": suf_probs[0][1], "offensive": suf_probs[0][2]}
            temp["comprehensiveness_classification_scores"] = {"hatespeech": com_probs[0][0], "normal": com_probs[0][1], "offensive": com_probs[0][2]}

            lime_dict_list.append(temp)

        return lime_dict_list


if __name__ == '__main__':
    """
    ### If you want to run the codes here, should import arguments from second_test.py and solve the circular import prob ###
    
    args = get_args_2()
    # args.device = torch.device('cpu')
    # print("device = cpu")
    args.device = get_device()
    args.test = True
    args.batch_size = 1
    args.n_eval = 0
    args.num_labels = 3

    model_path = '/home/jiyun/hatesp_detection_2/finetune_2nd/0718_4_bert_mrp0.5_5e-05_16_945/0719_3_bert_ori_ncls3_2e-05_16_745_False_False/for_explain_union.json'    
    args.model_path = model_path
    args.dir_result = '/'.join(model_path.split('/')[:7])

    lime_tester = TestLime(args)
    lime_dict_list = lime_tester.test(args)
    with open(args.dir_result + '/for_explain_lime.json', 'w') as f:
        f.write('\n'.join(json.dumps(i,cls=NumpyEncoder) for i in lime_dict_list))
    print('lime done')
    """

