import pandas as pd

def create_label_numbers():
    file_name = "datasets/adam/train_2500_sort_AB_Exp.txt"
    adam_pd = pd.read_csv(file_name, sep='\t')
    print(adam_pd.head())
    label_numbers = {}
    for (i,expansion) in enumerate(adam_pd["EXPANSION"]):
        label_numbers[expansion] = i
    label_pd = pd.Series(label_numbers)
    print(label_pd.head())
    label_pd.to_csv('datasets/adam/label_numbers.txt', sep='\t')


def main():
    file_name = "datasets/adam/label_numbers.txt"
    df = pd.read_csv(file_name, sep='\t', index_col = "EXPANSION")
    print(df.head())
    ser = df.squeeze()
    print(ser.head())
    print(ser[['active avoidance', 'acupuncture analgesia']].to_numpy())


if __name__ == "__main__":
    main()