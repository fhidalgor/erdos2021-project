import pandas as pd
import time
from icecream import ic

def remove_extra_columns(data):
    important_columns = ["ABSTRACT_ID", "TEXT", "LOCATION", "LABEL", "ABBR"]
    return data[important_columns]

def filter_entries(data, abbr_list, column = "ABBR"):
    return data[data[column].isin(abbr_list)]

def remove_long_columns(data, column = "TEXT", length = 256):
    return data[data[column].apply(lambda string: len(string.split()) < length)]

def main():


    ### Filter data to only choose ones with the follow abbreviations

    abbr_list = ["FM", "RR"]
    folder = "datasets/medal/two_abbr"

    # for type in ["valid", "test", "train"]:
    #     start = time.time()
    #     # data = pd.read_csv(f"datasets/medal/{type}_total_labeled.csv", error_bad_lines=False)
    #     data = pd.read_csv(f"datasets/medal/{type}_total_labeled.csv", error_bad_lines=False)
    #     end = time.time()
    #     print("Data downloaded", end-start)

    #     data_filtered = filter_entries(data, abbr_list)
    #     data_prime = remove_extra_columns(data_filtered)
    #     data_short = remove_long_columns(data_prime)
    #     data_short.to_csv(f"{folder}/{type}.csv", index = False)
    #     print("File created")

    ### Make a dictionary
    adam =pd.read_csv("datasets/adam/train_2500_sort_AB_Exp.txt", sep = '\t')
    adam_short = filter_entries(adam, abbr_list, "PREFERRED_AB")
    adam_short = adam_short.reset_index(drop=True)
    adam_short["LABEL"] = adam_short.index
    print(adam_short.head())
    adam_short.to_csv(f"{folder}/dict.txt", sep = '\t', index = False)

    # for type in ["valid", "test", "train"]:
    #     data = pd.read_csv(f"datasets/medal/one_abbr/{type}.csv")
    #     data1 = remove_long_columns(data)
    #     data2 = remove_extra_columns(data1)
    #     data2.to_csv(f"datasets/medal/one_abbr/{type}_max_256.csv", index = False)
    #     print(data2.head())



if __name__ == "__main__":
    main()