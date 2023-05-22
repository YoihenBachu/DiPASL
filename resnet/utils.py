import numpy as np
import pandas as pd
import os, warnings

import config
warnings.filterwarnings(action = "ignore")

def onehotencoder(value):
    one_hot_vector = np.zeros(26)

    # Set the corresponding index to 1
    one_hot_vector[value] = 1

    # Convert the one-hot vector to integer data type
    one_hot_vector = one_hot_vector.astype(int)

    return one_hot_vector

def make_df(all_path):
    c = 0
    alphabets = config.ALPHABETS
    df = pd.DataFrame(columns = ['img_path', 'label'])
    for data_path in all_path:
        base = os.path.split(data_path)[0]
        label = os.path.basename(base)
        if label in alphabets:
            df.loc[c] = [data_path, label]
            c+=1
    return df