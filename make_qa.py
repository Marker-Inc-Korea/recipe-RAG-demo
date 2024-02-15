import os

import click
import guidance
import pandas as pd
from autorag.data.qacreation.simple import generate_simple_qa_dataset
from dotenv import load_dotenv
from guidance import models, gen

root_dir = os.path.dirname(os.path.realpath(__file__))


# Example for LLM API
def generate_qa_row(llm: models.Model, corpus_data_row):
    temp_llm = llm

    # make template and synthetic data with guidance
    with guidance.user():
        temp_llm += f"""
    You have to make a question and answer pair about the recipe.
    You need to build a clean and clear set of (question, passage, answer) in json format
    
    ## Making Question 
    You can ask "How to make a cake?" or "What can I make with eggs, flour, and sugar?".
    It is okay to ask about the specific style of the food or flavor, 
    such as "How to make spicy fried chicken in Portuguese style?".
    Question must end with question mark("?").
    Be creative and make a question that is not too simple or too complex.
    Plus, do not ask a question that is too similar with the recipe itself.
    
    ## Making Answer
    You have to make a clear and concise answer to the question.
    The information for making the answer should be in the given recipe.
    Do not make up a new information. 
    Be kind and make a clear and concise answer.
    
   "Recipe": {corpus_data_row["contents"]}\n
   "Question": 
    """

    with guidance.assistant():
        temp_llm += gen('query', stop="?")
    with guidance.user():
        temp_llm += f"""
        "Answer":
        """
    with guidance.assistant():
        temp_llm += gen('generation_gt')

    # add metadata in the function
    corpus_data_row["metadata"]["qa_generation"] = "recipe_simple"

    # make response dictionary
    response = {
        "query": temp_llm["query"],
        "generation_gt": temp_llm["generation_gt"]
    }
    return response


@click.command()
@click.option('--output_filepath', default=os.path.join(root_dir, "data", "qa.parquet"),
              help='Output filepath for QA dataset')
def generate_qa(output_filepath):
    load_dotenv()
    corpus_df = pd.read_parquet(os.path.join(root_dir, "data", "corpus.parquet"))
    llm = models.OpenAI("gpt-3.5-turbo")
    qa_dataset = generate_simple_qa_dataset(corpus_data=corpus_df, llm=llm, output_filepath=output_filepath,
                                            generate_row_function=generate_qa_row)
    qa_dataset.to_parquet(output_filepath)


if __name__ == "__main__":
    generate_qa()
