import os
import re
import json
import yaml
import threading
from typing import TypedDict, Annotated, List, Dict, Any
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from configuration import ModelConfiguration
from functions import generate_answer, answer_evaluate, generate_new_question

# Add a lock for thread-safe file writing
log_file_lock = threading.Lock()

class AgentState(TypedDict):
    seed_task_info: Dict[str, Any]  # To store original task information
    
    seed_answer: str
    seed_answer_full_process: List
    answer_1: str
    answer_1_full_process: List
    answer_2: str
    answer_2_full_process: List
    
    create_new_question: bool
    new_question: str
    new_question_create_process: List
    new_answer: str
    new_answer_full_process: List[List]

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

    step_config["configurable"]["api_configs"] = base_config["configurable"]["api_configs"]
    
    return step_config

def seed_answer_node(state: AgentState, config: RunnableConfig):
    # Create step-specific configuration
    step_config = create_step_config(config, "seed_answer")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    
    original_question = state["seed_task_info"]["question"]
    
    # Generate the answer of the seed task
    answer_messages = generate_answer(cfg=cfg, question=original_question)

    match = re.search(r"<answer>(.*?)</answer>", answer_messages[-1]["content"], re.DOTALL)
    if match:
        seed_answer = match.group(1).strip()
    else:
        seed_answer = ""

    return {
        "seed_answer_full_process": answer_messages,
        "seed_answer": seed_answer,
    }

def answer_1_node(state: AgentState, config: RunnableConfig):
    # Create step-specific configuration
    step_config = create_step_config(config, "answer_1")
    cfg = ModelConfiguration.from_runnable_config(step_config)
    
    # Generate the answer for the constructed question
    answer_messages = generate_answer(cfg=cfg, question=state["new_question"])

    match = re.search(r"<answer>(.*?)</answer>", answer_messages[-1]["content"], re.DOTALL)
    if match:
        answer_1 = match.group(1).strip()
    else:
        answer_1 = ""

    return {
        "answer_1_full_process": answer_messages,
        "answer_1": answer_1,
    }

def answer_2_node(state: AgentState, config: RunnableConfig):
    # Create step-specific configuration
    step_config = create_step_config(config, "answer_2")
    cfg = ModelConfiguration.from_runnable_config(step_config)
        
    # Generate the answer for the constructed question
    answer_messages = generate_answer(cfg=cfg, question=state["new_question"])

    match = re.search(r"<answer>(.*?)</answer>", answer_messages[-1]["content"], re.DOTALL)
    if match:
        answer_2 = match.group(1).strip()
    else:
        answer_2 = ""

    return {
        "answer_2_full_process": answer_messages,
        "answer_2": answer_2,
    }

def evaluation_answer_node(state: AgentState, config: RunnableConfig):
    """Evaluate the correctness of the seed task answer."""
    if state["seed_answer"] == "" or state["seed_answer"] == "N/A":
        return {
            "create_new_question": False
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
        "create_new_question": "true" in evaluation_result_str
    }

def eval_consistency_node(state: AgentState, config: RunnableConfig):
    """Evaluate the consistency of the answers."""
    if state["answer_1"] == "" or state["answer_2"]  == "":
        return {
            "new_answer": "The answer is not finished or not found.",
            "new_answer_full_process": [state["answer_1_full_process"], state["answer_2_full_process"]]
        }
    step_config = create_step_config(config, "evaluation")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    question = state["new_question"]
    answer1 = state["answer_1"]
    answer2 = state["answer_2"]

    evaluation_result_str = answer_evaluate(
        cfg=cfg, question=question,
        answers=[answer1, answer2],
    )

    if "true" not in evaluation_result_str:
        return {
            "new_answer": "The answer is not consistent.",
            "new_answer_full_process": [state["answer_1_full_process"], state["answer_2_full_process"]]
        }
    return {
        "new_answer": state["answer_2"],
        "new_answer_full_process": [state["answer_1_full_process"], state["answer_2_full_process"]]
    }

def create_new_question_node(state: AgentState, config: RunnableConfig):
    """Create a new question based on the previous answer."""
    # Create step-specific configuration
    step_config = create_step_config(config, "new_question_generator")
    cfg = ModelConfiguration.from_runnable_config(step_config)

    seed_question = state["seed_task_info"]["question"]
    seed_question_solution = f"Question: {seed_question}\n\n"
    for msg in state["seed_answer_full_process"][1:]:
        if msg.get("content"):
            seed_question_solution += f"{msg['role']}: {msg['content']}\n"
        elif "function_call" in msg:
            seed_question_solution += f"{msg['role']} calls {msg['function_call']['name']} with {msg['function_call']['arguments']}\n"

    new_question, new_question_create_process = generate_new_question(cfg=cfg, question_solution=seed_question_solution)

    return {
        "new_question": new_question,
        "new_question_create_process": new_question_create_process,
    }

def should_create_new_question(state: AgentState):
    if state["create_new_question"]:
        return "create_new_question"
    else:
        return "end"

def is_new_question_created(state: AgentState):
    if state["new_question"] is not None:
        return "consistent_check"
    else:
        return "end"

def is_new_question_created2(state: AgentState):
    if state["new_question"] is not None:
        return "consistent_check"
    else:
        return "end"

# Build the graph
builder = StateGraph(AgentState, config_schema=RunnableConfig)
builder.add_node("seed_answer", seed_answer_node)
builder.add_node("evaluate_answer", evaluation_answer_node)
builder.add_node("create_new_question", create_new_question_node)
builder.add_node("answer_1", answer_1_node)
builder.add_node("answer_2", answer_2_node)
builder.add_node("eval_consistency", eval_consistency_node)


builder.set_entry_point("seed_answer")
builder.add_edge("seed_answer", "evaluate_answer")
builder.add_conditional_edges(
    "evaluate_answer",
    should_create_new_question,
    {"create_new_question": "create_new_question", "end": END}
)
builder.add_conditional_edges(
    "create_new_question",
    is_new_question_created,
    {"consistent_check": "answer_1", "end": END}
)
builder.add_conditional_edges(
    "create_new_question",
    is_new_question_created2,
    {"consistent_check": "answer_2", "end": END}
)
builder.add_edge("answer_1", "eval_consistency")
builder.add_edge("answer_2", "eval_consistency")
graph = builder.compile()

def save_steps_log(
        seed_task_info: dict,
        new_task_info: dict,
        new_task_solution: dict,
        new_question_create: dict,
        new_task_file_path: str="output/new_task.jsonl",
        new_task_solution_file_path: str="output/new_task_solution.jsonl",
        new_task_create_path: str = "output/new_task_create_path.jsonl"
    ):
    """Save all steps to respective JSONL files (thread-safe)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data with timestamp
    new_task_data = {
        "timestamp": timestamp,
        "seed_task_info": seed_task_info,
        "new_task_info": new_task_info
    }
    
    new_task_solution_data = {
        "timestamp": timestamp,
        **new_task_solution
    } if new_task_solution else None
    
    new_question_create_data = {
        "timestamp": timestamp,
        **new_question_create
    } if new_question_create else None

    # Thread-safe writes to respective JSONL files
    with log_file_lock:
        # Save basic task data
        with open(new_task_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_task_data, ensure_ascii=False) + '\n')
            
        # Save solution data if exists
        if new_task_solution_data:
            with open(new_task_solution_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_task_solution_data, ensure_ascii=False) + '\n')
                
        # Save creation path data if exists
        if new_question_create_data:
            with open(new_task_create_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_question_create_data, ensure_ascii=False) + '\n')
    
# --- 运行入口 ---
def run_agent(seed_task_info: dict, run_config: dict = None):
    new_task_file_path = run_config["logging"]["new_task_file_path"]
    new_task_solution_file_path = run_config["logging"]["new_task_solution_file_path"]
    new_task_create_path = run_config["logging"]["new_task_create_path"]
    
    log_dir = os.path.dirname(new_task_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    run_config = {"configurable": run_config or {}}
    
    initial_state = {
        "seed_task_info": seed_task_info,
    }
    
    final_state = graph.invoke(initial_state, config=run_config)
    seed_task_info["seed_answer_full_process"] = final_state["seed_answer_full_process"]
    
    if "new_question" not in final_state.keys():
        new_task_info = {}
        new_task_solution = {}
        new_question_create = {}

    else:
        if final_state["new_question"] is not None:
            final_state["new_question"] = final_state["new_question"].strip()
            new_id = str(uuid.uuid4())
            new_task_info = {
                "new_id": new_id,
                "new_question": final_state["new_question"],
                "new_answer": final_state["new_answer"],
            }
            new_task_solution = {
                "new_id": new_id,
                "new_question": new_task_info["new_question"],
                "new_answer": new_task_info["new_answer"],
                "new_answer_full_process": final_state["new_answer_full_process"]
            }
            new_question_create = {
                "new_id": new_id,
                "new_question": new_task_info["new_question"],
                "new_answer": new_task_info["new_answer"],
                "new_question_create_process": final_state["new_question_create_process"],
            }
        else:
            new_task_info = {
                "new_id": "N/A",
                "new_question": "Question Creation Failed",
                "new_answer": "N/A"
            }
            new_task_solution = {}
            new_question_create = {}

    save_steps_log(
        seed_task_info=seed_task_info,
        new_task_info=new_task_info,
        new_task_solution=new_task_solution,
        new_question_create=new_question_create,
        new_task_file_path=new_task_file_path,
        new_task_solution_file_path=new_task_solution_file_path,
        new_task_create_path=new_task_create_path
    )
    
    return final_state


if __name__ == "__main__":
    with open("configs/test.yaml", 'r', encoding='utf-8') as f:
        agent_config = yaml.safe_load(f)

    # Example usage
    task = {"id": "2e9f925f-0299-4255-82ed-b8bb2dc9627e", "question": "Find all functions \\( f: \\mathbb{R} \\rightarrow \\mathbb{R} \\) such that\n\n\\[ f(x + f(y)) + f(xy) = y f(x) + f(y) + f(f(x)) \\]", "true_answer": "The solutions are the zero function and the identity function. Thus, the functions are \\(\\boxed{f(x) = 0}\\) and \\(\\boxed{f(x) = x}\\)."}
    # task = {"question": "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$", "true_answer": 70, "id": "0"}
    # task = {"question": "Let $ABCDE$ be a convex pentagon with $AB=14,$ $BC=7,$ $CD=24,$ $DE=13,$ $EA=26,$ and $\\angle B=\\angle E=60^{\\circ}.$ For each point $X$ in the plane, define $f(X)=AX+BX+CX+DX+EX.$ The least possible value of $f(X)$ can be expressed as $m+n\\sqrt{p},$ where $m$ and $n$ are positive integers and $p$ is not divisible by the square of any prime. Find $m+n+p.$", "true_answer": 60, "id": "13"}
    run_agent(task, run_config=agent_config)
