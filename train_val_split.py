import os

import pandas as pd

root_dir = os.path.dirname(os.path.realpath(__file__))


def split():
    qa_df = pd.read_parquet(os.path.join(root_dir, "data", "qa.parquet"))
    train_df = qa_df[:600]
    val_df = qa_df[600:]
    train_df.to_parquet(os.path.join(root_dir, "data", "qa_train.parquet"))
    val_df.to_parquet(os.path.join(root_dir, "data", "qa_val.parquet"))


if __name__ == "__main__":
    split()
