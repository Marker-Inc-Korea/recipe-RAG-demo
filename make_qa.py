import os
import uuid

import click
import guidance
import pandas as pd
from dotenv import load_dotenv
from guidance import models, gen

root_dir = os.path.dirname(os.path.realpath(__file__))


def generate_qa_row(llm: models.Model, corpus_data_row):
    qa_llm = llm

    # make template and synthetic data with guidance
    with guidance.user():
        qa_llm += f"""
    You have to make a question and answer pair about the recipe.
    You need to build a clean and clear set of (question, passage, answer) in json format
    
    ## Making Question 
    You can ask "How to make a cake?" or "What can I make with eggs, flour, and sugar?".
    It is okay to ask about the specific style of the food or flavor, 
    such as "How to make spicy fried chicken in Portuguese style?".
    Question must end with question mark("?").
    Be creative and make a question that is not too simple or too complex.
    Plus, do not ask a question that is too similar with the recipe itself.
    Think that you don't know about the recipe at all when you make a question.

   "Recipe": {corpus_data_row["contents"]}\n
   "Question": 
    """

    with guidance.assistant():
        qa_llm += gen(stop="?", temperature=0.0, name='q1')
    with guidance.user():
        qa_llm += "\nBe more Creative.\nQuestion: "
    with guidance.assistant():
        qa_llm += gen(stop="?", temperature=0.5, name='q2')
    with guidance.user():
        qa_llm += "\nStill simple and not creative enough.\nQuestion: "
    with guidance.assistant():
        qa_llm += gen(stop="?", temperature=1.0, name='q3')
    with guidance.user():
        qa_llm += "\nBe more Creative. Don't juse change the words. Write a whole new question.\nQuestion: "
    with guidance.assistant():
        qa_llm += gen(stop="?", temperature=1.0, name='q4')
    with guidance.user():
        qa_llm += "\nBe more Creative!! Think about random question about this recipe!\nQuestion: "
    with guidance.assistant():
        qa_llm += gen(stop="?", temperature=1.0, name='q5')

    # add metadata in the function
    corpus_data_row["metadata"]["qa_generation"] = "recipe_simple"

    queries = [qa_llm[f"q{i}"] for i in range(1, 6)]

    def generate_answer(question, corpus_data_row):
        gen_llm = llm
        with guidance.user():
            gen_llm += f"""
        ## Making Answer
        You have to make a clear and concise answer to the question.
        The information for making the answer should be in the given recipe.
        Do not make up a new information.
        Be kind and make a clear and concise answer.
        
        Recipe:
        {corpus_data_row["contents"]}
        
        Question: {question}
        
        Answer:
        """
        with guidance.assistant():
            gen_llm += gen('answer', temperature=0.5, max_tokens=512)

        return gen_llm["answer"]

    answers = list(map(lambda q: generate_answer(q, corpus_data_row), queries))
    response = list(map(lambda q, a: {"query": q, "generation_gt": a}, queries, answers))

    return response


def generate_simple_qa_dataset(llm: models.Model, corpus_data: pd.DataFrame,
                               output_filepath: str, generate_row_function, **kwargs):
    """
    corpus_data to qa_dataset
    qa_dataset will be saved to filepath(file_dir/filename)

    :param llm: guidance.models.Model
    :param corpus_data: pd.DataFrame. refer to the basic structure
    :param output_filepath: file_dir must exist, filepath must not exist. file extension must be .parquet
    :param generate_row_function: input(llm, corpus_data_row, kwargs) output(dict[columns contain "query" and "generation_gt"])
    :param kwargs: if generate_row_function requires more args, use kwargs
    :return: qa_dataset as pd.DataFrame
    """
    qa_data_lst = []
    for _, corpus_data_row in corpus_data.iterrows():
        response = generate_row_function(llm=llm, corpus_data_row=corpus_data_row, **kwargs)
        qa_data_lst.extend(list(map(lambda res: {
            'qid': str(uuid.uuid4()),
            'query': res["query"],
            'retrieval_gt': [[corpus_data_row["doc_id"]]],
            'generation_gt': [res["generation_gt"]],
            'metadata': corpus_data_row["metadata"]
        }, response)))

    qa_dataset = pd.DataFrame(qa_data_lst)
    qa_dataset.to_parquet(output_filepath, index=False)

    return qa_dataset


@click.command()
@click.option('--output_filepath', default=os.path.join(root_dir, "data", "qa.parquet"),
              help='Output filepath for QA dataset')
def generate_qa(output_filepath):
    load_dotenv()
    corpus_df = pd.read_parquet(os.path.join(root_dir, "data", "corpus.parquet"))
    llm = models.OpenAI("gpt-3.5-turbo")
    corpus_df = corpus_df.sample(1)
    qa_dataset = generate_simple_qa_dataset(corpus_data=corpus_df, llm=llm, output_filepath=output_filepath,
                                            generate_row_function=generate_qa_row)
    qa_dataset.to_parquet(output_filepath)


if __name__ == "__main__":
    generate_qa()
