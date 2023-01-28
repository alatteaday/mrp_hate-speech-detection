import torch

import pandas as pd
from transformers import BertTokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from .attentionCal import aggregate_attention
from .spanMatcher import returnMask,returnMaskonetime

from TensorDataset.datsetSplitter import createDatasetSplit
from TensorDataset.dataLoader import combine_features
#from metrics.dataCollect import get_test_data, convert_data, get_annotated_data, transform_dummy_data
from TensorDataset.datsetSplitter import encodeData


##### Data collection for test data
def get_test_data(data,params,message='text'): 
    '''input: data is a dataframe text ids labels column only'''
    '''output: training data in the columns post_id,text (tokens) , attentions (normal) and labels'''
    
    if(params['bert_tokens']):
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    else:
        tokenizer=None
    
    post_ids_list=[]
    text_list=[]
    attention_list=[]
    label_list=[]
    print('total_data',len(data))
    for index,row in tqdm(data.iterrows(),total=len(data)):
        post_id=row['post_id']
        annotation=row['final_label']
        tokens_all,attention_masks=returnMask(row,params,tokenizer)
        attention_vector= aggregate_attention(attention_masks,row, params) 
        attention_list.append(attention_vector)
        text_list.append(tokens_all)
        label_list.append(annotation)
        post_ids_list.append(post_id)
    
    # Calling DataFrame constructor after zipping 
    # both lists, with columns specified 
    training_data = pd.DataFrame(list(zip(post_ids_list,text_list,attention_list,label_list)), 
                   columns =['Post_id','Text', 'Attention' , 'Label']) 
    
    return training_data


def get_testloader_with_rational(params, test_data=None,extra_data_path=None, topk=2,use_ext_df=False):
    """
    # device = torch.device("cpu")
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        deviceID = get_gpu(params)
        torch.cuda.set_device(deviceID[0])
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    """
    
    embeddings=None
    if(params['bert_tokens']):
        train,val,test=createDatasetSplit(params)
        vocab_own=None    
        vocab_size =0
        padding_idx =0
    else:
        train,val,test,vocab_own=createDatasetSplit(params)
        params['embed_size']=vocab_own.embeddings.shape[1]
        params['vocab_size']=vocab_own.embeddings.shape[0]
        embeddings=vocab_own.embeddings
    if(params['auto_weights']):
        y_test = [ele[2] for ele in test] 
        encoder = LabelEncoder()
        encoder.classes_ = np.load('Data/classes.npy')
        params['weights']=class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_test), y=y_test).astype('float32')
    if(extra_data_path!=None):
        # params_dash={}
        # params_dash['num_classes']=3
        # params_dash['data_file']=extra_data_path
        # params_dash['class_names']=dict_data_folder[str(params['num_classes'])]['class_label']
        # temp_read = get_annotated_data(params_dash)
        
        temp_read = pd.read_json('/hdd/jiyun/hatesp_detect/hateXplain/hatexplain_three.json', orient ='records')

        with open('Data/post_id_divisions.json', 'r') as fp:
            post_id_dict=json.load(fp)
        temp_read=temp_read[temp_read['post_id'].isin(post_id_dict['test']) & (temp_read['final_label'].isin(['hatespeech','offensive']))]
        test_data=get_test_data(temp_read,params,message='text')
        test_extra=encodeData(test_data,vocab_own,params)
        test_dataloader=combine_features(test_extra,params,is_train=False)
    elif(use_ext_df):
        test_extra=encodeData(test_data,vocab_own,params)
        test_dataloader=combine_features(test_extra,params,is_train=False)
    else:
        test_dataloader=combine_features(test,params,is_train=False)



if __name__ == '__main__':
    params_path = '/home/jiyun/HateXplain/best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json'
    params = pd.read_json(params_path, orient ='records')

    get_testloader_with_rational(params, )


    