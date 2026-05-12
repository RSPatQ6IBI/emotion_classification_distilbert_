from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os

def prepare_data_for_train_val_test_sets_(emodata_df_, goemotions_ml_loc_):
    base_path_ = os.getcwd()
    print(f"The base path -->> ", base_path_)
    train_location_ = os.path.join(base_path_, goemotions_ml_loc_)
    train_location_ = train_location_.replace('\\','/')
    print(train_location_)
    if not os.path.exists(train_location_):
        os.makedirs(train_location_)
    #--<><>----<><>----<><>----<><>----<><>----<><>----<><>--
    data = emodata_df_
    # Creating training and validation sets using an 80-20 split
    input_train, input_val, target_train, target_val = train_test_split(data.text.to_numpy(),data.emotions.to_numpy(), test_size=0.2)
    # Split the validataion further to obtain a holdout dataset (for testing) -- split 50:50
    input_val, input_test, target_val, target_test = train_test_split(input_val, target_val, test_size=0.5)
    ## create a dataframe for each dataset
    train_dataset = pd.DataFrame(data={"text": input_train, "class": target_train})
    val_dataset = pd.DataFrame(data={"text": input_val, "class": target_val})
    test_dataset = pd.DataFrame(data={"text": input_test, "class": target_test})
    final_dataset = {"train": train_dataset, "val": val_dataset , "test": test_dataset }

    train_path = goemotions_ml_loc_+'train.txt'
    test_path = goemotions_ml_loc_+'test.txt'
    val_path = goemotions_ml_loc_+'val.txt'
    train_dataset.to_csv(train_path, sep=";",header=False, index=False)
    val_dataset.to_csv(test_path, sep=";",header=False, index=False)
    test_dataset.to_csv(val_path, sep=";",header=False, index=False)
    print('Training, Validation and Test dataset created , at path : ', goemotions_ml_loc_)
    

class EmoDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data_column = "text"
        self.class_column = "class"
        self.data = pd.read_csv(path, sep=";", header=None, names=[self.data_column, self.class_column],
                               engine="python")
        print('READ THE FILE --- >> >> ', path, type(self.data))

    def __getitem__(self, idx):
        return self.data.loc[idx, self.data_column], self.data.loc[idx, self.class_column]

    def __len__(self):
        return self.data.shape[0]


