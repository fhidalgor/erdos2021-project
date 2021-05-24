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


    # for type in ["valid", "test", "train"]:
    #     start = time.time()
    #     # data = pd.read_csv(f"datasets/medal/{type}_total_labeled.csv", error_bad_lines=False)
    #     data = pd.read_csv(f"datasets/medal/{type}_total_labeled.csv", error_bad_lines=False)
    #     end = time.time()
    #     print("Data downloaded", end-start)

    #     data_filtered = filter_entries(data, ["SG"])
    #     data_prime = remove_extra_columns(data_filtered)
    #     data_prime.to_csv(f"datasets/medal/one_abbr/{type}.csv", index = False)
    #     print(data_prime.head())
    #     print("File created")

    # adam =pd.read_csv("datasets/adam/train_2500_sort_AB_Exp.txt", sep = '\t')
    # #adam = pd.read_csv("datasets/adam/train_2500_sort_AB_Exp.txt", sep="\t")
    # adam_short = filter_entries(adam, ["SG"], "PREFERRED_AB")
    # adam_short = adam_short.reset_index(drop=True)
    # adam_short["LABEL"] = adam_short.index
    # # print(adam_short.head())
    # adam_short.to_csv("datasets/medal/one_abbr/dict.txt", sep = '\t', index = False)

    for type in ["valid", "test", "train"]:
        data = pd.read_csv(f"datasets/medal/one_abbr/{type}.csv")
        data1 = remove_long_columns(data)
        data2 = remove_extra_columns(data1)
        data2.to_csv(f"datasets/medal/one_abbr/{type}_max_256.csv", index = False)
        print(data2.head())



if __name__ == "__main__":
    main()