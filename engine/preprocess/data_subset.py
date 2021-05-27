import os, pickle
import numpy as np
import pandas as pd
from itertools import compress
import copy



def generate_subset(int: cutoff=1500, string: data_dir, string: save_dir, format=['train', 'valid', 'test']):

''' Generate subset of data from initial datasets corresponding to most occuring abbreviations

        Parameters
        --------------
        cutoff : int
                Cutoff to apply to reference dataset "train" for identifying most occuring ABBVs

        data_dir : string
                Path to directory containing .csv files with which to generate subsets

        save_dir : string
                Path to directory for saving subsets

        format: list
                Names of files in 'data_dir' for converting to subsets

        Returns
        --------------
        None

        '''
    train = pd.read_csv("data_dir" + format[0] + ".csv");

    text_list = train["TEXT"];
    location_list = train["LOCATION"];
    abbvs = []

    for i in range(len(location_list)):
        abbvs.append(text_list[i].split()[location_list[i]]);

    abbvs = np.array(abbvs);
    values, counts = np.unique(abbvs, return_counts=True);

    values_new = values[np.where(counts > cutoff)[0]];
    redux_abbv_index = [abbvs[i] in values_new for i in range(len(abbvs))];

    new_train = train[redux_abbv_index]

    print("Number of ABBVs above cutoff: %0.1d \n" % len(values_new));
    print("Number of unique expansions: %0.1d \n" % len(values_new));

    new_train.to_csv(save_dir + format[0] + "_" + cutoff + ".csv");

    if len(format) > 1:
        for files in format[1:]:
            reduce_dataset("data_dir", files + ".csv", save_dir, values_new, cutoff);

    generate_word_dict(new_train, format[0] + "_" + cutoff);


def reduce_dataset(string: data_dir, string: filename, string: save_dir, values, int: cutoff):
    ''' Following initial dataset scraping, reduce subsequent files in the same manner

            Parameters
            --------------

            data_dir : string
                    Path to directory containing .csv files with which to generate subsets

            filename : string
                    Name of subset file to save as.

            save_dir : string
                    Path to directory for saving subsets

            values: np.array()
                    Array of ABBVs that satisfy the cutoff applied to original reference data

            cutoff : int
                    Cutoff applied to original reference data, appended to output filename.
            Returns
            --------------
            None

            '''
    dataset = pd.read_csv(datadir + filename);
    text = dataset["TEXT"];
    location = dataset["LOCATION"];

    abbvs_test = [];

    for i in range(len(location_list_test)):
        abbvs_test.append(text[i].split()[location[i]]);

    redux_abbv_index = [abbvs_test[i] in values for i in range(len(abbvs_test))];

    new_test = dataset[redux_abbv_index]
    new_test.to_csv(save_dir + filename + "_" + cutoff + ".csv");

def generate_word_dict(DataFrame: dataset, string:filename):
    ''' Generate ADAM.txt file containin all ABBVs and Expansions used in reduced dataset. Data alphabetized first by abbreviation: "PREFERRED_AB" and secondly by "EXPANSION".

            Parameters
            --------------

            dataset : pd.DataFrame
                    DataFrame containing reduced reference data from original set.

            filename : string
                    Input name of file used as reference, and output ADAM file name

            --------------
            None

            '''
    df = copy.Deepcopy(dataset);

    uniques = df.LABEL.unique();
    texts = df.TEXT.to_numpy()
    locs = df.LOCATION.to_numpy()
    labels = df.LABEL.to_list()

    indexes = [labels.index(x) for x in set(labels)]

    dict_ray = np.zeros([len(uniques), 2], dtype=object);

    i=0;
    for index in indexes:
        AB = texts[index].split()[locs[index]];
        EX = labels[index];
        dict_ray[i] = AB, EX
        i+=1;

    df = pd.DataFrame(dict_ray, columns = ['PREFERRED_AB','EXPANSION'])
    df.sort_values(by=['PREFERRED_AB', "EXPANSION"]).to_csv(filename + '_sort_AB_Exp.txt', sep='\t', index=False)
