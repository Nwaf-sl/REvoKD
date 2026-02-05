import argparse
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default="/mnt/data/verl/datasets/stage2/stage1_mix_0.3_no_solution.jsonl",  # 你的 jsonl 路径
        help="Path to the FVQA raw dataset"
    )
    parser.add_argument(
        "--local_save_dir",
        default="/mnt/data/verl/datasets/stage2",
        help="Where to save parquet files"
    )

    args = parser.parse_args()

    data_source = "self_rl"

    dataset = datasets.load_dataset("json", data_files=args.local_dataset_path)["train"]

    total_len = len(dataset)
    split_idx = int(total_len * 0.9)
    train_dataset = dataset.select(range(split_idx))
    system_prompt = (
    """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

        Start an iterative loop for each question:

        - **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
        - **Next, find information:** Use a tool to research the things you need to find out.
        - **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

        Continue this loop until your research is complete.

        To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question."""
    )

    instruction_following = (
        """Multiple tool invocations are permitted. Prior to each tool invocation, you must articulate your analytical reasoning.

        Critical reminder: Tools operate in a stateless manner and will not preserve any variables, functions, or definitions from prior executions.

        Reasoning step by step, write ONLY the final answer enclosed in <answer> and </answer> tags. Do not include any extra text outside the tags after the final answer. /no_think
        """
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example["question"]
            prompt = question + "\n" + instruction_following
            image_path = example["image"]      
            answer = example["true_answer"]

            data = {
                "agent_name": "tool_agent",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "image": [{"image":image_path}], 
                "ability": "vqa",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                    "category": example.get("category", ""),
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "image_zoom_tool": {  
                            "create_kwargs": {'image':image_path} },
                        "search": { 
                            "create_kwargs": None
                        }
                    }
                }
            }

            return data
        return process_fn
    
  
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    
    local_save_dir = args.local_dir if args.local_dir else args.local_save_dir
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "trian.parquet"))