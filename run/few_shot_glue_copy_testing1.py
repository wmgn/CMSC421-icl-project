from batcon.config import DatasetConfig, NetConfig
from batcon.net import Net
from batcon.consts import *
from batcon.dataset import MiniBatchDataset, RandomDataset, Dataset
from batcon.pipeline import Pipeline, EntangledPipeline, RNNPipeline

import argparse
from datasets import load_dataset
from evaluate import load
import random
import torch
import time

def main(dataset_config_path, net_config_path, dataset_name, verbose, seed, single_step, valid_limit):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    start_time = time.time()

    train_source = list(load_dataset("glue", dataset_name, split="train", cache_dir="cache"))
    for entry in train_source:
        entry["label_text"] = prompt_dict[dataset_name]["label_map_reverse"][entry["label"]]
    if valid_limit > 0:
        dev_source = load_dataset("glue", dataset_name, split=f"validation[:{valid_limit}]", cache_dir="cache")
    else:
        dev_source = load_dataset("glue", dataset_name, split="validation", cache_dir="cache")
    labels = dev_source["label"]
    dataset_config_train = DatasetConfig(dataset_config_path)
    dataset_config_dev = DatasetConfig()
    #                             source_dataset  prompt_template: str,                                     config: dataset_config
    train_dataset = MiniBatchDataset(train_source, prompt_dict[dataset_name]["single_example_prompt"], dataset_config_train)
    test_dataset = MiniBatchDataset(dev_source, prompt_dict[dataset_name]["question_prompt"], dataset_config_dev) #using minibatch dataset makes this take 2x as long
    
    net_config = NetConfig(net_config_path)
    net = Net(net_config)
    #if single_step:
    first_prompt = prompt_dict[dataset_name]["CMSC421_test_prompt_comedy_w_memo_before_examples"]
    first_prompt_backup = prompt_dict[dataset_name]["CMSC421_test_prompt_comedy_w_out_memo"]
    #print(f'!!!!!first_prompt!!!!!\n\n\n: {first_prompt}\n\n\n')
    second_prompt = None
    #else:
    #    first_prompt = prompt_dict[dataset_name]["CMSC421_test_prompt_1a"]
    #   second_prompt = prompt_dict[dataset_name]["CMSC421_test_prompt_1a"]
    #pipeline = Pipeline(net, train_dataset, test_dataset, single_step=single_step, first_prompt=first_prompt, second_prompt=second_prompt)
    pipeline = RNNPipeline(net, train_dataset, test_dataset, single_step=single_step, first_prompt=first_prompt, second_prompt=second_prompt, first_prompt_backup=first_prompt_backup)

    all_results = pipeline.evaluate(verbose=verbose, label_map=prompt_dict[dataset_name]["label_map"])
    #print(f'\n!!!test print of all_results!!!\n\n\n: {all_results}\n\n\n')
    # can use results to do mini-batches, will have to make a new pipeline????, though??? maybe
    # figure out how to get pipeline.py to use different in-context examples from dataset



    #results = [prompt_dict[dataset_name]["label_map"][r] for r in results]
    #metric = load('glue', dataset_name)
    #evaluation_results = metric.compute(predictions=results, references=labels)
    #print(f'Evaluation results on {dataset_name}: {evaluation_results}')
    end_time = time.time()
    print(f'Used time: {end_time-start_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """configs/dataset_config_few_shot.json
    {
    "max_steps": 1,
    "example_size": 3 #changed this to 3 from 4
    }
    """
    
    #parser.add_argument('--prompt_name', default="single_example_prompt", type=str)
    parser.add_argument('--dataset_config_path', default="configs/dataset_config_few_shot.json", type=str)
    parser.add_argument('--net_config_path', default="configs/net_config_basic.json", type=str)
    parser.add_argument('--dataset_name', default="sst2", type=str)
    parser.add_argument('--single_step', action=argparse.BooleanOptionalAction)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--valid_limit', default=100, type=int)
    args = parser.parse_args()
    main(**vars(args))

    
'''
class Llama2Runner():
    def __init__(self, net):
        self.net = net

    def run_prompt(self, prompt):
        return self.net.inference(prompt)
#if __name__ == "__main__":
    # Assuming you have a pre-trained Llama2-7b model (net)
    net = PreTrainedLlama2Model()  # Replace with your actual pre-trained model instantiation

    # Create the Llama2Runner
    llama2_runner = Llama2Runner(net)

    # Define prompts you want to run
    
    prompts = [
        "Prompt 1: {examples}",
        "Prompt 2: {question} - {examples}",
        # Add more prompts as needed
    ]
    #from old def main
    first_prompt = prompt_dict[dataset_name]["CMSC421_test_prompt_1a4comedy"]

    # Run prompts and collect results
    results = []
    for prompt in prompts:
        result = llama2_runner.run_prompt(prompt)
        results.append(result)
        print(f"Prompt: {prompt}")
        print(f"Result: {result}\n")

    # Save or process the results as needed
    # For example, you can write the results to a file for manual evaluation
    with open("results.txt", "w") as f:
        for prompt, result in zip(prompts, results):
            f.write(f"Prompt: {prompt}\nResult: {result}\n\n")
'''