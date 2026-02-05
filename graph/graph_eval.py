import os
import re
import json
from tkinter import Image, image_names
import yaml
import threading
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

from configuration import ModelConfiguration
from functions import generate_answer, answer_evaluate

# Add a lock for thread-safe file writing
log_file_lock = threading.Lock()

class AgentState(TypedDict):
    seed_task_info: Dict[str, Any]  # To store original task information
    seed_answer: str
    seed_answer_full_process: List
    is_correct: bool

def create_step_config(
        base_config: RunnableConfig, step_name: str, 
    ) -> RunnableConfig:
    """Create a new configuration for a specific step with its designated model"""
    # cfg = AgentConfiguration.from_runnable_config(base_config)
    step_model_config = base_config["configurable"]["step_models"][step_name]
    
    # Create a new config with the specific model for this step
    step_config = {}
    if "configurable" not in step_config:
        step_config["configurable"] = {}
        
    # Apply the step-specific model configuration
    step_config["configurable"]["model_name"] = step_model_config["name"]
    if "temperature" in step_model_config:
        step_config["configurable"]["temperature"] = step_model_config["temperature"]
    if "max_tokens" in step_model_config:
        step_config["configurable"]["max_tokens"] = step_model_config["max_tokens"]
    if "use_tools" in step_model_config:
        step_config["configurable"]["use_tools"] = step_model_config["use_tools"]
    if "use_thinking" in step_model_config:
        step_config["configurable"]["use_thinking"] = step_model_config["use_thinking"]

    step_config["configurable"]["api_configs"] = base_config["configurable"]["api_configs"]
    
    return step_config

def seed_answer_node(state: AgentState, config: RunnableConfig):
    # Create step-specific configuration
    step_config = create_step_config(config, "seed_answer")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    
    original_question = state["seed_task_info"]["question"]
    image_path = state["seed_task_info"]["image"]
    
    # Generate the answer of the seed task
    answer_messages = generate_answer(cfg=cfg, question=original_question,image_path=image_path)

    match = re.search(r"<answer>(.*?)</answer>", answer_messages[-1]["content"], re.DOTALL)
    if match:
        seed_answer = match.group(1).strip()
    else:
        seed_answer = ""

    return {
        "seed_answer_full_process": answer_messages,
        "seed_answer": seed_answer,
    }


def evaluation_answer_node(state: AgentState, config: RunnableConfig):
    """Evaluate the correctness of the seed task answer."""
    if state["seed_answer"] == "" or state["seed_answer"] == "N/A":
        return {
            "is_correct": False
        }
    
    # Create step-specific configuration
    step_config = create_step_config(config, "evaluation")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    
    # Extract necessary information
    seed_task_info = state["seed_task_info"]
    evaluated_answer = state["seed_answer"]
    
    question = seed_task_info["question"]
    true_answer = seed_task_info["true_answer"]

    evaluation_result_str = answer_evaluate(
        cfg=cfg, question=question,
        answers=[true_answer, evaluated_answer],
    )

    return {
        # Only create a new question if the answer is correct, excluding over-difficult questions
        "is_correct": "true" in evaluation_result_str
    }


# Build the graph
builder = StateGraph(AgentState, config_schema=RunnableConfig)
builder.add_node("seed_answer", seed_answer_node)
builder.add_node("evaluate_answer", evaluation_answer_node)

builder.set_entry_point("seed_answer")
builder.add_edge("seed_answer", "evaluate_answer")
graph = builder.compile()

# --- 运行入口 ---
def run_agent(seed_task_info: dict, run_config: dict = None):
    output_path = run_config["logging"]["eval_file_path"]
    
    log_dir = os.path.dirname(output_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    run_config = {"configurable": run_config or {}}
    
    initial_state = {
        "seed_task_info": seed_task_info,
    }
    
    final_state = graph.invoke(initial_state, config=run_config)

    task_eval = {
        "id": seed_task_info["id"],
        "question": seed_task_info["question"],
        "image": seed_task_info['image'],
        "solution": final_state["seed_answer_full_process"],
        "answer": final_state["seed_answer"],
        "true_answer": seed_task_info["true_answer"],
        "is_correct": final_state["is_correct"]
    }

    with log_file_lock:
        # Save basic task data
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(task_eval, ensure_ascii=False) + '\n')
    
    return final_state


if __name__ == "__main__":
    with open("configs/eval.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)

    # Example usage
    # task = {"id": "2e9f925f-0299-4255-82ed-b8bb2dc9627e", "question": "Find all functions \\( f: \\mathbb{R} \\rightarrow \\mathbb{R} \\) such that\n\n\\[ f(x + f(y)) + f(xy) = y f(x) + f(y) + f(f(x)) \\]", "true_answer": "The solutions are the zero function and the identity function. Thus, the functions are \\(\\boxed{f(x) = 0}\\) and \\(\\boxed{f(x) = x}\\)."}
    # task = {"question": "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$", "true_answer": 70, "id": "0"}
    task = {"question": "Let $ABCDE$ be a convex pentagon with $AB=14,$ $BC=7,$ $CD=24,$ $DE=13,$ $EA=26,$ and $\\angle B=\\angle E=60^{\\circ}.$ For each point $X$ in the plane, define $f(X)=AX+BX+CX+DX+EX.$ The least possible value of $f(X)$ can be expressed as $m+n\\sqrt{p},$ where $m$ and $n$ are positive integers and $p$ is not divisible by the square of any prime. Find $m+n+p.$", "true_answer": 60, "id": "13"}
    run_agent(task, run_config=agent_config)
