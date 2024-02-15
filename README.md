# recipe-RAG-demo
You are the developer of a recipe app.
Your boss wants to create a new chatbot based on the recipe data.
Ultimately, the chatbot can replace search engine in your app. 

Now, you have to make the proof of concept for the chatbot in a few days. 
The only change to make it fast, yet powerful, is using [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG).
This is the journey to make the chatbot with the recipe.


# Installation

```bash
pip install -r requirements.txt
```

# Running the project

- Download dataset to data folder.
You can download raw dataset from [kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv).
Download `RAW_recipes.csv` only. 
That's all we need.
- Run `preprocess.py` to make corpus.parquet.
- Run `make_qa.py` and `train_val_split.py` to make qa.parquet.
- Make `.env` file using `.env.template` file.
- Run evaluator with the following command.
```bash
python main.py --config /path/to/config.yaml
```

- Check the result in the benchmark folder.

You can check the example config file at config folder.

And you can specify qa data path, corpus data path, and project dir if you want.
