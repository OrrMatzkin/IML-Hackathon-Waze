import pandas as pd

def make_train_data(pre_process_df, not_processes_df):
    no_shift = pre_process_df
    shift1 = pre_process_df.shift(periods=-1)
    shift2 = pre_process_df.shift(periods=-2)
    shift3 = pre_process_df.shift(periods=-3)
    shift4_for_labels = not_processes_df.shift(periods=-4)

    flatted_4_samples = pd.concat([no_shift, shift1, shift2, shift3], axis=1)

    type_labels = shift4_for_labels['linqmap_type']
    subtype_labels = shift4_for_labels['linqmap_subtype']
    x_labels = shift4_for_labels['x']
    y_labels = shift4_for_labels['y']

    return flatted_4_samples.iloc[:-4], type_labels.iloc[:-4], subtype_labels.iloc[:-4], x_labels.iloc[:-4], y_labels.iloc[:-4]




