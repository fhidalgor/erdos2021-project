import pandas as pd
import time
from icecream import ic
from transformers import BertTokenizer

def remove_extra_columns(data):
    important_columns = ["ABSTRACT_ID", "TEXT", "LOCATION", "LABEL", "ABBR"]
    return data[important_columns]

def filter_entries(data, abbr_list, column = "ABBR"):
    return data[data[column].isin(abbr_list)]

def remove_long_columns(data, column = "TEXT", length = 256):
    return data[data[column].apply(lambda string: len(string.split()) < length)]

def remove_big_long_loc(data, length = 256):
    # We need room for the CLS and SEP tokens.
    return data[data['LONG_LOC']<=length-2]

def check_long_location(ser):
    return ser['TOKEN_IDS'][ser['LONG_LOC']]
    
def find_long_location(ser):
    tokens = ser['TEXT']
    abbr = ser['ABBR']
    loc = ser['LOCATION']
    abbr_lower = abbr.lower()
    return tokens.index(abbr_lower, loc)

def main():
    ### Filter data to only choose ones with the follow abbreviations

    # abbr_list = ["AI","ER","BT","IT","VE","SO","BR", "RM"]
    # abbr_list = ["RR", "FM"]
    # abbr_list = ['BD', 'PCA']
    abbr_list = ['IP', 'AH', 'BC', 'ME', 'LC', 'GS', 'LT', 'TT', 'RT', 'DS', \
        'FA', 'IA', 'AL', 'PV', 'CI', 'PH', 'TE', 'ED', 'CH', 'SL', \
        'CN', 'DA', 'MG', 'PL', 'CB', 'NC', 'OP', 'OM', 'LL', 'MT', \
        'ID', 'AT', 'GP', 'RE', 'EE', 'TB', 'TI', 'FC', 'TR', 'GT', \
        'OR', 'IR', 'FT', 'GM', 'NE', 'RD', 'RI', 'BS', 'VR', 'AE', \
        'RC', 'AO', 'ML', 'RM', 'AI', 'ER', 'BT', 'IT', 'VE', 'SO', \
        'BR', 'RR', 'FM', 'SG']
    folder = "datasets/medal/64_abbr"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for type in ["valid", "test", "train"]:
        start = time.time()
        data = pd.read_csv(f"datasets/medal/{type}_total_labeled.csv", error_bad_lines=False)
        #data = pd.read_csv(f"{folder}/{type}.csv", error_bad_lines=False)
        end = time.time()
        print("Data downloaded", end-start)

        data_filtered = filter_entries(data, abbr_list)
        data_prime = remove_extra_columns(data_filtered)
        ic(data_prime)
        #ic(data)
        data_prime.to_csv(f"{folder}/{type}.csv", index = False)
        print("File created")

        start = time.time()
        tokens = data_prime['TEXT'].apply(tokenizer.tokenize)
        end = time.time()
        print(f"Tokenized in {end-start :> .1f} seconds")
        ic(tokens)
        data1 = pd.concat([tokens, data_prime['ABBR'], data_prime['LOCATION']], axis=1)
        ic(data1)
        data_prime['LONG_LOC'] = data1.apply(find_long_location, axis = 1)

        data2 = remove_big_long_loc(data_prime)
        data2.to_csv(f"{folder}/{type}_long_loc.csv", index = False)
        print("File created")
        ic(data2.head())


    ## Make a dictionary
    # adam =pd.read_csv("datasets/adam/train_2500_sort_AB_Exp.txt", sep = '\t')
    # adam_short = filter_entries(adam, abbr_list, "PREFERRED_AB")
    # adam_short = adam_short.reset_index(drop=True)
    # adam_short["LABEL"] = adam_short.index
    # print(adam_short.head())
    # adam_short.to_csv(f"{folder}/dict.txt", sep = '\t', index = False)

    # for type in ["valid", "test", "train"]:
    #     num_abbr = "eight_abbr"
    #     #data = pd.read_csv(f"datasets/medal/{num_abbr}/{type}_long_loc.csv")
    #     data = pd.read_csv(f"datasets/medal/{num_abbr}/{type}_long_loc.csv")
        
    #     data2 = remove_extra_columns(data1)
    #     max = data2['LONG_LOC'].max()
    #     # data2.to_csv(f"datasets/medal/{num_abbr}/{type}_max_256.csv", index = False)
    #     print(max)



if __name__ == "__main__":
    main()