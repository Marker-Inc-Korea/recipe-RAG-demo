# AutoRAG-template
Template for a new AutoRAG project


# Installation

```bash
pip install -r requirements.txt
```

# Running the project

1. Download dataset to data folder.
You can download raw dataset from [kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv).
Download `RAW_recipes.csv` only. 
That's all we need.
2. Run `preprocess.py` to make corpus.parquet.
3. Make `.env` file using `.env.template` file.
4. Run evaluator with the following command.
```bash
python main.py --config /path/to/config.yaml
```
5. Check the result in the benchmark folder.

You can check the example config file at config folder.

And you can specify qa data path, corpus data path, and project dir if you want.
