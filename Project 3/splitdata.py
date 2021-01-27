"""split_data.py
"""

from typing import Optional, Tuple
import pandas as pd


def get_raw_dataset():
    "returns the whole dataset downloaded from kaggle"
    inputs = pd.read_csv('input_training_set.csv', index_col=-1)
    inputs_label = pd.read_csv('output_training_set.csv')
    return inputs.join(inputs_label)


def shuffle(ds: pd.DataFrame, random_state: Optional[int] = None) -> pd.DataFrame:
    "shuffle a DataFrame with an optional random state"
    return ds.sample(frac=1., random_state=random_state).reset_index(drop=True)


def main():
    "split the dataset"

    ds = get_raw_dataset()
    ds = shuffle(ds, random_state=42)

    training_step = int(.7 * len(ds))
    testing_step = training_step + int(.15 * len(ds))
    # other 15% are for validation

    ds.iloc[0:training_step].to_csv('dataset/training.csv', index=False)
    ds.iloc[training_step:testing_step].to_csv('dataset/testing.csv', index=False)
    ds.iloc[testing_step:].to_csv('dataset/validation.csv', index=False)


if __name__ == "__main__":
    main()
