import io
import json
import logging
import math
import os
import re
import sys

from functools import partial
from multiprocessing import Pool
from random import sample
from string import punctuation
from time import sleep, time
from typing import Sequence, Union

import numpy as np
from openai import OpenAI, OpenAIError
from rouge_score import rouge_scorer
from tqdm import tqdm


class SelfInstruct:
    def __init__(
        self,
        api_key=None,
        model_provider="openai",
        model_name="text-davinci-003",
        prompt_template_path="./prompt.txt",
        seed_tasks_path="./seed_tasks.jsonl",
        num_instructions_to_generate=100,
        num_prompt_instructions=3,
        request_batch_size=5,
        num_cpus=16,
    ):
        """
        SelfInstruct class.

        Args:
          api_key (str): OpenAI API key.
          model_provider (str): Model provider.
          model_name (str): Model name.
          prompt_template_path (str): Prompt template path.
          seed_tasks_path (str): Seed tasks path.
          num_instructions_to_generate (int): Number of instructions to generate.
          num_prompt_instructions (int): Number of prompt instructions to select from seed tasks.
          request_batch_size (int): Request batch size.
          num_cpus (int): Number of CPUs.
        """
        if model_provider == "openai":
            self.client = OpenAI(api_key=api_key)
        else:
            raise NotImplementedError(
                f"Model provider {model_provider} not implemented."
            )

        self.prompt_template_path = prompt_template_path
        self.prompt_template = None
        self.seed_tasks_path = seed_tasks_path
        self.num_instructions_to_generate = num_instructions_to_generate
        self.model_provider = model_provider
        self.model_name = model_name
        self.num_prompt_instructions = num_prompt_instructions
        self.request_batch_size = request_batch_size
        self.num_cpus = num_cpus

    def generate(self, output_dir="./"):
        """
        main program logic

        Steps:
          1. Load seed tasks
          2. Synthesize instructions
          3. Filter out bad instructions
          4. Save instructions to file (seed tasks and generated tasks are not mixed)
        """

        seed_tasks = [json.loads(l) for l in open(self.seed_tasks_path, "r")]
        seed_instruction_data = [
            {"instruction": t["instruction"], "response": t["response"]} for t in seed_tasks
        ]

        print(f"Loaded {len(seed_instruction_data)} seed instructions")

        os.makedirs(output_dir, exist_ok=True)
        request_idx = 0

        # load the LM-generated instructions (first run shouldn't trigger this) -tim
        machine_instruction_data = []

        if os.path.exists(os.path.join(output_dir, "regen.json")):
            machine_instruction_data = self.jload(
                os.path.join(output_dir, "regen.json")
            )
            print(
                f"Loaded {len(machine_instruction_data)} machine-generated instructions"
            )

        # similarities
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        # now let's generate new instructions!
        progress_bar = tqdm(total=self.num_instructions_to_generate)

        if machine_instruction_data:
            progress_bar.update(len(machine_instruction_data))

        # first we tokenize all the seed instructions and generated machine instructions
        all_instructions = [d["instruction"] for d in seed_instruction_data] + [
            d["instruction"] for d in machine_instruction_data
        ]
        all_instruction_tokens = [
            scorer._tokenizer.tokenize(inst) for inst in all_instructions
        ]
        
        while len(machine_instruction_data) < self.num_instructions_to_generate:
            request_idx += 1
            batch_inputs = []
            window_size = 3  # hardcoded sliding window size of 5 samples
            step_size = 3 # hardcoded step size of 5 elements
            
            for i in range(self.request_batch_size):
                # random sample
                # prompt_inst = sample(
                #     seed_instruction_data, self.num_prompt_instructions
                # )
                # prompt = self.encode_prompt(prompt_inst)
                # batch_inputs.append(prompt)
            
                # sliding window 
                start_index = i * step_size % len(seed_instruction_data)
                end_index = (start_index + window_size) % len(seed_instruction_data)
                prompt_inst = seed_instruction_data[start_index:end_index]

                if len(prompt_inst) < window_size:  # wrap around when reaching end of list
                    prompt_inst += seed_instruction_data[:window_size - len(prompt_inst)]

                prompt = self.encode_prompt(prompt_inst)
                    
                batch_inputs.append(prompt)

            request_start = time()

            if self.model_provider == "openai":
                decoding_args = {
                    "temperature": 0.7,
                    "n": 1,
                    "max_tokens": 2000,
                    "top_p": 1.0,
                    # "stop": [f"\n{self.num_instructions_to_generate}", f"{self.num_instructions_to_generate}."],
                }

                results = self.openai_completion(
                    prompts=batch_inputs,
                    decoding_args=decoding_args,
                )

                request_duration = time() - request_start
                process_start = time()

                instruction_data = []

                for result in results:
                    new_instructions = self.post_process(
                        self.num_prompt_instructions, result
                    )
                    instruction_data += new_instructions

                total = len(instruction_data)
                keep = 0

                for instruction_data_entry in instruction_data:
                    # computing similarity with the pre-tokenzied instructions
                    new_instruction_tokens = scorer._tokenizer.tokenize(
                        instruction_data_entry["instruction"]
                    )
                    with Pool(self.num_cpus) as p:
                        rouge_scores = p.map(
                            partial(rouge_scorer._score_lcs, new_instruction_tokens),
                            all_instruction_tokens,
                        )

                    rouge_scores = [score.fmeasure for score in rouge_scores]
                    most_similar_instructions = {
                        all_instructions[i]: rouge_scores[i]
                        for i in np.argsort(rouge_scores)[-10:][::-1]
                    }

                    if max(rouge_scores) >= 0.7:
                        continue
                    else:
                        keep += 1

                    instruction_data_entry[
                        "most_similar_instructions"
                    ] = most_similar_instructions
                    instruction_data_entry["avg_similarity_score"] = float(
                        np.mean(rouge_scores)
                    )

                    machine_instruction_data.append(instruction_data_entry)

                    all_instructions.append(instruction_data_entry["instruction"])
                    all_instruction_tokens.append(new_instruction_tokens)

                    progress_bar.update(1)

                process_duration = time() - process_start

                print(
                    f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s"
                )
                print(f"Generated {total} instructions, kept {keep} instructions")

                self.jdump(
                    machine_instruction_data, os.path.join(output_dir, "regen.json")
                )


            else:
                raise NotImplementedError(
                    f"Model provider {self.model_provider} not implemented"
                )
        else:
            print("\nGeneration complete!")

    def configure_prompt(self, topic
                         # , difficulty
                        ):
        """Configure the prompt template."""
        prompt = open(self.prompt_template_path).read() + "\n"
        prompt = prompt.format(
            num_questions=self.num_instructions_to_generate,
            topic=topic,
            # difficulty=difficulty,
        )

        self.prompt_template = prompt

    def encode_prompt(self, prompt_instructions):
        """Encode multiple prompt instructions into a single string."""
        prompt = self.prompt_template + "\n"

        for idx, task_dict in enumerate(prompt_instructions):
            (instruction, output) = task_dict["instruction"], task_dict["response"]
            instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
            prompt += f"###\n"
            prompt += f"{idx + 1}. Instruction: {instruction}\n"
            prompt += f"{idx + 1}. Response: {output}\n"

        prompt += f"###\n"
        prompt += f"{idx + 2}. Instruction:"
        return prompt

    def openai_completion(
        self,
        prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
        decoding_args={},
        batch_size=1,
        max_instances=sys.maxsize,
        sleep_time=2,
    ):
        is_single_prompt = isinstance(prompts, (str, dict))

        if is_single_prompt:
            prompts = [prompts]

        prompts = prompts[:max_instances]
        num_prompts = len(prompts)
        prompt_batches = [
            prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
            for batch_id in range(int(math.ceil(num_prompts / batch_size)))
        ]

        completions = []
        
        for batch_id, prompt_batch in tqdm(
            enumerate(prompt_batches),
            desc="prompt_batches",
            total=len(prompt_batches),
            position=0, leave=True
        ):
            batch_decoding_args = decoding_args

            while True:
                try:
                    # batched completion requests
                    completion_batch = self.client.completions.create(
                        prompt=prompt_batch,
                        model=self.model_name,
                        **batch_decoding_args,
                    )
                    choices = completion_batch.choices

                    completions.extend(choices)
                    break

                except OpenAIError as e:
                    logging.warning(f"OpenAIError: {e}.")
                    if "Please reduce your prompt" in str(e):
                        batch_decoding_args.max_tokens = int(
                            batch_decoding_args.max_tokens * 0.8
                        )
                        logging.warning(
                            f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying..."
                        )
                    else:
                        logging.warning("Hit request rate limit; retrying...")
                        sleep(sleep_time)  # Annoying rate limit on requests.

        if decoding_args["n"] > 1:
            # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
            completions = [
                completions[i : i + decoding_args["n"]]
                for i in range(0, len(completions), decoding_args["n"])
            ]
        if is_single_prompt:
            # Return non-tuple if only 1 input and 1 generation.
            (completions,) = completions

        return completions

    def find_word_in_string(w, s):
        return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)

    def post_process(self, num_prompt_instructions, response):
        if response is None:
            return []
        
        raw_instructions = (
            f"{num_prompt_instructions+1}. Instruction:" + response.text
        )
        
        raw_instructions = re.split("###", raw_instructions)
        instructions = []
        
        for idx, inst in enumerate(raw_instructions):
            # if the decoding stops due to length, the last example is likely truncated so we discard it            
            if (
                idx == len(raw_instructions) - 1
                and response.finish_reason == "length"
            ):
                continue
            
            idx += num_prompt_instructions + 1
            splitted_data = re.split(f"{idx}\.\s+(Instruction|Response):", inst)
            
            if len(splitted_data) != 5:
                continue
            else:
                inst = splitted_data[2].strip()
                output = splitted_data[4].strip()

            if len(inst.split()) <= 3: #or len(inst.split()) > 150:
                continue

            if inst.startswith("Write a program"):
                continue

            # filter those starting with punctuation
            if inst[0] in punctuation:
                continue

            # filter those starting with non-english character
            if not inst[0].isascii():
                continue

            instructions.append({"instruction": inst, "context": "", "response": output})
        
        return instructions

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode)
        return f
        
    def _make_w_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f_dirname = os.path.dirname(f)
            if f_dirname != "":
                os.makedirs(f_dirname, exist_ok=True)
            f = open(f, mode=mode)
        return f

    def jdump(self, obj, f, mode="w", indent=4, default=str):
        """Dump a str or dictionary to a file in json format.

        Args:
            obj: An object to be written.
            f: A string path to the location on disk.
            mode: Mode for opening the file.
            indent: Indent for storing json dictionaries.
            default: A function to handle non-serializable entries; defaults to `str`.
        """
        f = self._make_w_io_base(f, mode)
        if isinstance(obj, (dict, list)):
            json.dump(obj, f, indent=indent, default=default)
        elif isinstance(obj, str):
            f.write(obj)
        else:
            raise ValueError(f"Unexpected type: {type(obj)}")
        f.close()

    def jload(self, f, mode="r"):
        """Load a .json file into a dictionary."""
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict