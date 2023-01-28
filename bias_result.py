from collections import Counter,defaultdict
from tqdm.notebook import tqdm
import os
import json
import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def convert_to_score(label_name, label_dict):
    if label_name=='non-toxic':
        return 1-label_dict[label_name]
    else:
        return label_dict[label_name]


# Function to divide the ids into postive or class class based on the method.
def calculate_bias_metrics(dataset, method, community):
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


def get_bias_dict(bias_score_file_mapping, method_list, community_list, bias_test_data):
    final_bias_dictionary = defaultdict(lambda: defaultdict(dict))

    # We load each of the model bias output file and compute the bias score using each method for all the community
    for each_model in tqdm(bias_score_file_mapping):
        total_data ={}
        with open(bias_score_file_mapping[each_model]) as fp:
            for line in fp:
                data = json.loads(line)
                total_data[data['annotation_id']] = data

        for each_method in method_list:
            for each_community in community_list:
                community_data = calculate_bias_metrics(bias_test_data, each_method, each_community)
                truth_values = []
                prediction_values = []

                label_to_value = {'toxic':1.0, 'non-toxic':0.0}
                for each in community_data['positiveID']:
                    truth_values.append(label_to_value[total_data[each]['ground_truth']])
                    prediction_values.append(convert_to_score(total_data[each]['classification'], total_data[each]['classification_scores']))

                for each in community_data['negativeID']:
                    truth_values.append(label_to_value[total_data[each]['ground_truth']])
                    prediction_values.append(convert_to_score(total_data[each]['classification'], total_data[each]['classification_scores']))

                roc_output_value = roc_auc_score(truth_values, prediction_values)
                final_bias_dictionary[each_model][each_method][each_community] = roc_output_value

    return final_bias_dictionary


def get_bias_results(args, bias_score_file_mapping):
    with open(os.path.join(args.dir_hatexplain, 'hatexplain_two_div.json'), 'r') as fp:
        data = json.load(fp)

    bias_test_data = json_normalize(data['test']) 
    target_group_list = ['African', 'Islam', 'Jewish', 'Homosexual', 'Women', 'Refugee', 'Arab', 'Caucasian', 'Asian', 'Hispanic']
    method_list = ['subgroup', 'bpsn', 'bnsp']

    bias_dict = get_bias_dict(bias_score_file_mapping, method_list, target_group_list, bias_test_data)

    # To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:
    # power_value = -5
    # num_communities = len(target_group_list)

    # for each_model in bias_dict:
    #     for each_method in bias_dict[each_model]:
    #         temp_value =[]
    #         for each_community in bias_dict[each_model][each_method]:
    #             temp_value.append(pow(bias_dict[each_model][each_method][each_community], power_value))
    #         print(each_model, each_method, pow(np.sum(temp_value)/num_communities, 1/power_value))

    # get_plots(bias_dict)

    return bias_dict, len(target_group_list)
    

def get_plots(bias_dict):
    #tuple_community = []
    for each_method in ['bnsp', 'subgroup', 'bpsn']:
        tuple_community = []
        for each_model in bias_dict:
            # Select the metric which you want to view. Possible values are subgroup, bpsn, bnsp
            #each_method = 'bnsp' 
            
            for each_community in bias_dict[each_model][each_method]:
                tuple_community.append((each_model, each_community, bias_dict[each_model][each_method][each_community]))
        
        df_community_score = pd.DataFrame(tuple_community, columns=['Model', 'Community', 'AUCROC'])

        ax = sns.catplot(x="Community", y="AUCROC", hue="Model",
                    data=df_community_score,
                    legend=False,
                    kind="bar")
        ax.set(ylim=(0.3, 1.0))
        ax.set_xticklabels(rotation=45, size=13, horizontalalignment='right')

        # sns.set(font_scale = 0.1)

        handles = ax._legend_data.values()
        labels = ax._legend_data.keys()

        ax.fig.legend(handles=handles, labels=labels, loc='upper right', ncol=3)
        ax.fig.subplots_adjust(top=0.92)

        #ax.set(xlabel=each_method.upper(), fontdict={'fontsize': 15, 'fontweight': 'bold'}, pad=15)  
        plt.xlabel(xlabel=each_method.upper(), fontdict={'fontsize': 15, 'fontweight': 'bold'}, labelpad=5)  
        #plt.title(each_method.upper(), loc='left', fontdict={'fontsize': 15}, pad=15)
        #print(each_method)
        plt.savefig('/home/jiyun/hatesp_detection/vis/bias_'+each_method+'.png', dpi=300, transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    """ 
    ### You can draw plots by write the dictionary like {model name: for_bias.json file of the model} and evaluating several models at once ###
    ### Solve the circular import prob to run the codes

    args = get_args_2()
    bias_score_file_mapping = {
        'BERT-MLM': "path of 'for_bias.json",
        'BERT-RP': "path of 'for_bias.json",
        'BERT-MRP': "path of 'for_bias.json",
        }

    bias_dict = get_bias_results(args, bias_score_file_mapping)
    get_plots(bias_dict)

    """