import pandas as pd
import numpy as np
import subprocess
import pathlib 
import os
import shutil 
def get_toml_info_():
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # Backport for older versions
    with open("pyproject.toml", "rb") as f:
        data_toml_ = tomllib.load(f)
    return data_toml_


def run_wget_(the_url_, the_op_loc_):
    import wget
    datafile_ = wget.download(the_url_)
    print(datafile_)
    # subprocess.run(["wget", "-P", the_op_loc_, the_url_])


## DOWNLOAD THE DATA AT PROPER DIRECTORIES 
def get_goemotions_data_from_googleapis_(data_toml_):
    db_linkA_ = data_toml_["dataset_goemotions"]["dataset_goemotions_linkA_"]
    db_linkB_ = data_toml_["dataset_goemotions"]["dataset_goemotions_linkB_"]
    db_linkC_ = data_toml_["dataset_goemotions"]["dataset_goemotions_linkC_"]
    goemotions_dl_loc_ = data_toml_["dataset_goemotions"]["goemotions_download_location_"]
    base_path_ = os.getcwd()
    base_path_ = pathlib.PureWindowsPath(base_path_)
    print(f"The base path -->> ", base_path_)
    dataset_location_ = os.path.join(base_path_, goemotions_dl_loc_)
    dataset_location_ = dataset_location_.replace('\\','/')
    print(dataset_location_)
    if not os.path.exists(dataset_location_):
        os.makedirs(dataset_location_)
    run_wget_(  the_url_=db_linkA_, the_op_loc_= goemotions_dl_loc_)
    run_wget_(  the_url_=db_linkB_, the_op_loc_= goemotions_dl_loc_)
    run_wget_(  the_url_=db_linkC_, the_op_loc_= goemotions_dl_loc_)
    print("data downloaded successfully")
    the_goemo_csvs_ = [y for y in os.listdir(base_path_) if '.csv' in y]
    for csv_file_ in the_goemo_csvs_:
        shutil.move(os.path.join(base_path_,csv_file_), os.path.join(dataset_location_,csv_file_))
        print('new file loc -- > ', os.path.join(base_path_,goemotions_dl_loc_,csv_file_))



def prepare_data_matrix_(emotion_matrix_, text_vec_):
  the_txt_vec_ = []
  the_lab_vec_ = []
  for idx_ in range(len(emotion_matrix_)):
    position_emotions_ = np.argwhere(emotion_matrix_[idx_,:] == 1)
    if len(position_emotions_)>=1:
      class_lab_ = position_emotions_[0]
      the_txt_vec_.append(text_vec_[idx_])
      the_lab_vec_.append(class_lab_[0])
  return the_txt_vec_, the_lab_vec_


def prepare_dataframe(data_toml_):
    import os
    goemotions_dl_loc_ = data_toml_["dataset_goemotions"]["goemotions_download_location_"]
    csv_dataset_files_ = [files_ for files_ in os.listdir(goemotions_dl_loc_) if '.csv' in files_]  
    final_df_data_ = []
    for csv_file_ in csv_dataset_files_:
        df_goemo_ = pd.read_csv(os.path.join(goemotions_dl_loc_, csv_file_))
        all_cols_ = df_goemo_.columns
        desired_emotions_ = [all_cols_[9:]]
        desired_emotions_lab_ = desired_emotions_[0].to_numpy()
        emotion_matrix_ = df_goemo_[desired_emotions_[0].to_numpy()[:]]
        emotion_matrix_ = emotion_matrix_.to_numpy()
        text_vec_ = df_goemo_["text"].to_numpy()
        the_txt_vec_, the_lab_vec_ = prepare_data_matrix_(emotion_matrix_, text_vec_)
        if len(final_df_data_) == 0:    
            # print('adding first frame -- ')
            final_df_data_ = pd.DataFrame(data={"text": the_txt_vec_, "emotions": the_lab_vec_})
            # print(final_df_data_.head())
        else: 
            temp_df_ = pd.DataFrame(data={"text": the_txt_vec_, "emotions": the_lab_vec_})
            final_df_data_ = pd.concat([final_df_data_, temp_df_], ignore_index=True)
            # print('adding subsequent frames -- ')
            # print(final_df_data_.head())

    return final_df_data_, desired_emotions_lab_


def plot_emotion_class_data_distribution(emo_data_frame_, desired_emotions_lab_):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 6))
    plt.grid()
    binwidth = 1
    plt.hist(emo_data_frame_.emotions, bins=range(0, len(desired_emotions_lab_) + binwidth, binwidth), color='lightsalmon', edgecolor='slategray')
    plt.title('Histogram of emotion classes')
    for ax_ in range(len(desired_emotions_lab_)):
      plt.text(ax_+0.25, 2e4, desired_emotions_lab_[ax_], rotation=90, fontweight='bold', fontsize = 20)
    plt.show()
