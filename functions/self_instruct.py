import re
from .call_llms import call_qwen_agent

def generate_new_question(cfg, question_solution):
    prompt = f"""You are a reasoning-question brainstorming assistant. Given a seed question, create **at least 5** novel, self-contained reasoning questions of similar difficulty — without solving them.

Guidelines:
1. Study the seed question's solving approach, core concepts, and structure.  
2. Consider potential variations to this problem in order to create a novel problem. Such as:
   - Changing numbers or parameters.  
   - Adding or modifying conditions to increase logical depth if appropriate.
   - Introducing alternative but related concepts and deeper theory.
   - Keeping difficulty comparable and answer type precise (integer, fraction, radical, yes/no, multiple-choice). 
   - Any other creative transformation that maintains or slightly increases the question's difficulty. 
3. Transform the seed question base on the variations.
4. Select the single best question: novel, clear, original, challenging, with an unambiguous answer.  

**Output Requirements:**
- First, write "Reasoning Process:" and show detailed step-by-step thinking following Guidelines 1–4.
- Then write "Candidate Questions:" and list at least 5 possible variations you created.
- Finally, enclose the most novel new question between `<New Question Begin>` and `<New Question End>` tags.

Generate a new question that is slightly more difficult than the given question, Question and Solution:
{question_solution}

IMPORTANT:
1. The originality of the question is crucial. Please try to create new questions instead of simply copying or rewriting the original questions.
2. Consider as many novel variations as possible and try to apply them to new problems
3. Spend most of your effort on building the question. Only solve parts when doing so is necessary to check or improve its quality.
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
