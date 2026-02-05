import re
import json
import random
from .call_llms import call_qwen_agent

with open("configs/persona.jsonl") as f:
    persona_list = [json.loads(line) for line in f]
def generate_new_question_based_persona(cfg, question_solution):    
    random_selected_persona = random.sample(persona_list, 5)
    random_selected_persona = "\n".join([persona["persona"] for persona in random_selected_persona])
    prompt = f"""You are a math problem contextual rewriting assistant.

    Given a pure math problem and a list of personas, rewrite the problem so it fits naturally into a realistic scenario related to one chosen persona.  
    You may adjust the problem's constants, conditions, or parameter ranges to make it sound natural in the chosen context while keeping the core mathematical model essentially the same.

    ### Required Thinking Process:
    1. Analyze the original math problem's structure, variables, and type.
    2. Check the provided persona list:
    - For each persona, check if it could plausibly be connected to the problem scenario.
    - If none fit well, invent a new persona whose profession or interests naturally match the problem.
    3. Select the final persona (either from the list or the invented one).
    4. Based on the chosen persona, design a realistic scenario that uses the same math structure.
    5. **MANDATORY:** Slightly modify the constants, conditions, OR variable meanings so they naturally fit the context.  
    - You MUST describe exactly what you modified and why, in the Reasoning Process.
    - Examples: change distances, times, weights, capacities, speeds, prices, etc. to 
    6. Rewrite the problem statement:
    - Integrate the scenario description into the question.
    - Explain what each variable represents in that scenario.
    - State the mathematical question clearly.
    
    **Output Requirements:**
    - First, write "Persona Choice or Invented Persona".
    - Then write "Reasoning Process:" and show detailed step-by-step thinking following Guidelines 1–6.
    - Finally, enclose your rewritten scenario-based math problem with variable meanings between `<New Question Begin>` and `<New Question End>` tags.

    Original Question and Solution:
    {question_solution}

    Possible Personas:
    {random_selected_persona}

    IMPORTANT: The originality of the question is crucial. Please try to create new questions instead of simply copying or rewriting the original questions.
    """
    # Generate the first answer
    messages = call_qwen_agent(
        user_prompt=prompt,
        system_prompt="",
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        use_tools=cfg.use_tools
    )
    new_question = messages[-1]["content"]
    matches = re.findall(r"<New Question Begin>(.+?)<New Question End>", new_question, re.DOTALL)
    if matches:
        last_match = matches[-1]
        return last_match, messages
    else:
        return None, messages
