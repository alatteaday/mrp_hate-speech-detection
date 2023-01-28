from collections import Counter,defaultdict
from tqdm import tqdm
import json
import numpy as np


def generate_target_information(dataset):
    final_target_output = defaultdict(list)
    all_communities_selected = []
    
    for each in dataset.iterrows(): 
        # All the target communities tagged for this post
        all_targets = each[1]['target1']+each[1]['target2']+each[1]['target3']  
        community_dict = dict(Counter(all_targets))
        
        # Select only those communities which are present more than once.
        for key in community_dict:
            if community_dict[key]>1:  
                final_target_output[each[1]['post_id']].append(key)
                all_communities_selected.append(key)
        
        # If no community is selected based on majority voting then we don't select any community
        if each[1]['post_id'] not in final_target_output:
            final_target_output[each[1]['post_id']].append('None')
            all_communities_selected.append(key)

    return final_target_output, all_communities_selected


list_selected_community = ['African', 'Islam', 'Jewish', 'Homosexual', 'Women', 'Refugee', 'Arab', 'Caucasian', 'Asian', 'Hispanic']

method_list = ['subgroup', 'bpsn', 'bnsp']

community_list = list(list_selected_community)


# This function is used to convert the classification into a [0-1] score 
# with a value of 0 meaning non-toxic and 1 meaning toxic
def convert_to_score(label_name, label_dict):
    if label_name=='non-toxic':
        return 1-label_dict[label_name]
    else:
        return label_dict[label_name]


# Function to divide the ids into postive or class class based on the method.
def bias_evaluation_metric(dataset, method, community):
    positive_ids = []
    negative_ids = []
    if method=='subgroup':
        for eachrow in dataset.iterrows():
            if eachrow[1]['final_target_category'] == None:
                continue
            if community in eachrow[1]['final_target_category']:
                if eachrow[1]['final_label'] =='non-toxic':
                    negative_ids.append(eachrow[1]['post_id'])
                else:
                    positive_ids.append(eachrow[1]['post_id'])
            else:
                pass
    elif method=='bpsn':
        for eachrow in dataset.iterrows():
            if eachrow[1]['final_target_category'] == None:
                continue
            if community in eachrow[1]['final_target_category']:
#                 print(eachrow[1]['final_label'])
                if eachrow[1]['final_label'] =='non-toxic':
                    negative_ids.append(eachrow[1]['post_id'])
                else:
                    pass
            else:
                if eachrow[1]['final_label'] !='non-toxic':
                    positive_ids.append(eachrow[1]['post_id'])
                else:
                    pass
    elif method=='bnsp':
        for eachrow in dataset.iterrows():
            if eachrow[1]['final_target_category'] == None:
                continue
            if community in eachrow[1]['final_target_category']:
                if eachrow[1]['final_label'] !='non-toxic':
                    positive_ids.append(eachrow[1]['post_id'])
                else:
                    pass
            else:
                if eachrow[1]['final_label'] =='non-toxic':
                    negative_ids.append(eachrow[1]['post_id'])
                else:
                    pass
    else:
        print('Incorrect option selected!!!')
                
    return {'positiveID':positive_ids, 'negativeID':negative_ids}





if __name__ == '__main__':
    final_bias_dictionary = defaultdict(lambda: defaultdict(dict))
    d = defaultdict(dict)
    d['hi']
    final_bias_dictionary['hi']['hii']=1
    print(final_bias_dictionary)