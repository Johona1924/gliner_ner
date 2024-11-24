import numpy as np
import pandas as pd
import os
import json

from numpy.lib.stride_tricks import sliding_window_view

from data_preprocessing import tokenize_text

def token_range_per_entity(text : str, entity : str) -> list[list]: 

    start_end_pairs = []
    
    #Tokenize text string and convert to np.ndarray
    tokenized_text = np.array(tokenize_text(text))

    tokenized_entity = tokenize_text(entity)

    #Sliding window
    window_view = sliding_window_view(tokenized_text,len(tokenized_entity))

    matching_indices = np.argwhere(np.all(window_view == tokenized_entity,axis = 1)).flatten()

    for start_index in matching_indices:
        end_index = start_index + len(tokenized_entity) - 1
        start_end_pairs.append([start_index,end_index])

    return start_end_pairs
    

def ner_tags_per_entity(start_end_pairs : list[list], label : str) -> list[list]:
    return [[int(pair[0]),int(pair[1]),str(label)] for pair in start_end_pairs]


def get_datapoint(text: str,entities : list[str], labels : list[str]) -> dict:
    ner = []

    for entity,label in zip(entities,labels):
        start_end_pairs = token_range_per_entity(text,entity)
        ner.extend(ner_tags_per_entity(start_end_pairs,label))

    
    return {"tokenized_text" : tokenize_text(text),
            "ner" : ner 
            }


def ood_dataframe_to_dataset(data : pd.DataFrame) -> list[dict]:

    output = data.apply(lambda row : get_datapoint(row['text'],row['entities'],row['labels']),axis = 1).tolist()

    return output


if __name__ == '__main__':

    output_path = f"data/test/ood"
    data_path = f"data/ood/synthetic"

    os.makedirs(output_path,exist_ok = True)

    french_df = pd.read_parquet(os.path.join(data_path,"french_ood.parquet"))
    german_df = pd.read_parquet(os.path.join(data_path,"german_ood.parquet"))
    italian_df = pd.read_parquet(os.path.join(data_path,"italian_ood.parquet"))
    with open(os.path.join(output_path,"french_ood.json"),'w') as file:
        json.dump(ood_dataframe_to_dataset(french_df),file)
    with open(os.path.join(output_path,"german_ood.json"),'w') as file:
        json.dump(ood_dataframe_to_dataset(german_df),file)
    with open(os.path.join(output_path,"italian_ood.json"),'w') as file:
        json.dump(ood_dataframe_to_dataset(italian_df),file)




    













