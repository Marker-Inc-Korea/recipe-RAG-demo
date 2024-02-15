import os
import ast
from datetime import datetime

import pandas as pd

root_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(root_dir, 'data')


def preprocess():
    raw_df = pd.read_csv(os.path.join(data_dir, 'RAW_recipes.csv'))
    raw_df = raw_df.dropna()
    raw_df = raw_df[raw_df['minutes'] >= 1]
    raw_df = raw_df[raw_df['n_steps'] >= 10]
    raw_df = raw_df[raw_df['n_steps'] < 20]
    raw_df = raw_df[raw_df['description'].str.len() > 20]
    raw_df = raw_df.drop(columns=['contributor_id', 'nutrition'])
    raw_df['tags'] = raw_df['tags'].apply(lambda x: ast.literal_eval(x))
    raw_df['steps'] = raw_df['steps'].apply(lambda x: ast.literal_eval(x))
    raw_df['ingredients'] = raw_df['ingredients'].apply(lambda x: ast.literal_eval(x))
    result_df = raw_df.copy()
    result_df['contents'] = result_df.apply(make_corpus, axis=1)
    result_df['metadata'] = result_df['submitted'].apply(
        lambda x: {'last_modified_datetime': datetime.strptime(x, '%Y-%m-%d')})
    result_df = result_df[['id', 'contents', 'metadata']]
    result_df.rename(columns={'id': 'doc_id'})
    result_df.sample(1000, random_state=42).to_parquet(os.path.join(data_dir, 'corpus.parquet'), index=False)


def make_corpus(row):
    step_str = '\n'.join([f"{i}. {val}" for i, val in enumerate(row['steps'])])

    return f"""
    # {row['name']} recipe
    
    Estimated time: {row['minutes']} minutes
    
    ## Ingredients
    {', '.join(row['ingredients'])}
    
    ## Steps
    {step_str}
    
    ## Description
    {row['description']}
    
    ## Tags
    {', '.join(row['tags'])}
    
    """


if __name__ == '__main__':
    preprocess()
