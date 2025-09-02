import pandas as pd
import wfdb
from src.utils import get_class_mapping, get_record_numbers
from src.preprocessing import split_data, extract_features_from_dataframe, feature_extracting
import os

def generateData(data=None, record_numbers=get_record_numbers(), window_size=187):
    """
    Generates all datasets that are necessary to train the model.
    First, it generates a interim dataset that contains the features for each heartbeat, 
    then it generates the filtered and scaled training, validation and testing datasets. 
    """
    if os.path.exists('data/interim/mitbih_combined_records.csv'):
        print("Interim dataset already exists. Skipping generation.")
    else:
        rows = []

        for record_number in record_numbers:
            record = wfdb.rdrecord(f'data/raw/mit-bih-arrhythmia-database-1.0.0/{record_number}')
            ann = wfdb.rdann(f'data/raw/mit-bih-arrhythmia-database-1.0.0/{record_number}', 'atr')
            class_mapping = get_class_mapping()
            
            for i in range(len(ann.sample)):
                symbol = ann.symbol[i]
                if symbol in class_mapping:
                    center = ann.sample[i]
                    w2 = window_size//2
                    start = max(0, center - (w2))
                    end = min(len(record.p_signal[:,0]), center + (w2) + (window_size % 2))
        
                    if end - start == window_size:
                        beat_samples = record.p_signal[start:end, 0].tolist()
                        beat_samples.append(class_mapping[symbol])
                        rows.append(beat_samples)
        
        column_names = [f'sample_{i}' for i in range(window_size)] + ['class']
        
        df = pd.DataFrame(rows, columns=column_names)

        df.to_csv('data/interim/mitbih_combined_records.csv', index=False)

    if data is None:
        split_data('data/interim/mitbih_combined_records.csv')
        feature_extracting()
        extract_features_from_dataframe()
        split_data('data/interim/mitbih_features.csv')
        split_data('data/interim/mitbih_features_only.csv')
    elif data == "no_feat":
        split_data('data/interim/mitbih_combined_records.csv')
    elif data == "cnn":
        split_data('data/interim/mitbih_combined_records.csv')
        feature_extracting()
    elif data == "feat":
        extract_features_from_dataframe()
        split_data('data/interim/mitbih_features.csv')
    elif data == "feat_only": 
        extract_features_from_dataframe()
        split_data('data/interim/mitbih_features_only.csv')



if __name__ == "__main__":
    generateData(data="no_feat")  # Options: None, , "no_feat", "cnn", "feat", "feat_only"