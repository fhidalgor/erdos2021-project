import pandas as pd
from icecream import ic
from transformers import BertTokenizer
import time

def check_long_location(ser):
    return ser['TOKEN_IDS'][ser['LONG_LOC']]
    

def find_long_location(ser):
    tokens = ser['TEXT']
    abbr = ser['ABBR']
    loc = ser['LOCATION']
    abbr_lower = abbr.lower()
    return tokens.index(abbr_lower, loc)

def tokens_to_ids(tokens, tokenizer):
    ids = tokenizer.convert_tokens_to_ids(tokens)
    # ids.append(102)
    # ids.insert(0, 101)
    return ids

def main():
    for type in ["train", "test", "valid"]:
        df = pd.read_csv(f"datasets/medal/two_abbr/{type}_all.csv")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # df = df.head(10)

        start = time.time()
        tokens = df['TEXT'].apply(tokenizer.tokenize)
        # df['TOKEN_IDS'] = tokens.apply(lambda t: tokens_to_ids(t, tokenizer))
        end = time.time()
        print(f"Tokenized in {end-start :> .1f} seconds")

        df2 = pd.concat([tokens, df['ABBR'], df['LOCATION']], axis=1)
        df['LONG_LOC'] = df2.apply(find_long_location, axis = 1)
        print(df.head())
        df.to_csv(f"datasets/medal/two_abbr/{type}_long_loc.csv")
        # print(df.apply(check_long_location, axis = 1 ))
    
    # df['TOKEN_LOC'] = tokens.apply(lambda t: find_long_location(t, ))


if __name__ == '__main__':
    main()