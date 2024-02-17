import os

import click
from autorag import generator_models
from autorag.deploy import Runner
from dotenv import load_dotenv
from llama_index.llms.vllm import Vllm

root_path = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.join(root_path, 'benchmark')


@click.command()
@click.option('--config', type=click.Path(exists=True))
@click.option('--query', type=str)
def run(config, query):
    load_dotenv()
    generator_models['vllm'] = Vllm
    runner = Runner.from_yaml(config, project_dir=project_dir)
    answer = runner.run(query)
    print(answer)


if __name__ == '__main__':
    run()
