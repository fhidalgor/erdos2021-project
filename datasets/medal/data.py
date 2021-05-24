import pandas as pd
import time
from icecream import ic

def main():
    type = "valid"
    # type = "test"
    type = "train"
    start = time.time()
    data = pd.read_csv(f"datasets/medal/{type}_total_labeled.csv", error_bad_lines=False)
    end = time.time()

    print("Data downloaded", end-start)
    
    # data["ABBR"] = data.apply(lambda row: row.TEXT.split()[row.LOCATION], axis=1)

    freq = data["ABBR"].value_counts()
    print(freq.head())
    
    top_series = freq.head(100)
    top_abb = top_series.keys()
    # ic(top_abb)

    # filter = data.ABBR.isin(top_abb)

    # filtered = data[filter]
    # ic(filtered)

    top_series.to_csv(f"datasets/medal/{type}_1000_keys.csv")
    # print("Success")
    # print(data.head())

if __name__ == "__main__":
    main()