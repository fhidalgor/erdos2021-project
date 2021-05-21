import pandas as pd
import time

def main():
    for n in [1000, 10000]:
        for type in ["test", "train", "valid"]:
            start = time.time()
            data = pd.read_csv(f"datasets/medal/{type}.csv", error_bad_lines=False, nrows = n)
            end = time.time()
            print(end - start)
            data.to_csv(f"datasets/medal/{type}{n}.csv")
            print("Success")
            print(data.head())

if __name__ == "__main__":
    main()