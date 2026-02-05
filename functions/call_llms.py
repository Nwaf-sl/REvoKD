import re
import copy
import json
from typing import Optional, Dict, List, Union, Literal
from openai import OpenAI
from openai.types.completion_usage import PromptTokensDetails
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from typing import List, Literal, Union
from qwen_agent.llm.schema import ContentItem
import json5

from qwen_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, ContentItem, FunctionCall, Message
from qwen_agent.log import logger

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import extract_fn, SPECIAL_CODE_MODE, NousFnCallPrompt

class NousFnCallPrompt2(NousFnCallPrompt):
    def postprocess_fncall_messages(
        self,
        messages: List[Message],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal['auto'], str] = 'auto',
        thought_in_content: bool = False,
    ) -> List[Message]:
        if function_choice != 'auto':
            raise NotImplementedError
        # Convert plaintext responses to function_call responses:
        new_messages = []
        tool_id = 1
        for msg in messages:
            role, content, reasoning_content, extra = msg.role, msg.content, msg.reasoning_content, msg.extra
            extra = extra or {}
            assert isinstance(content, list)

            if role in (SYSTEM, USER):
                new_messages.append(
                    Message(role=role, content=content, reasoning_content=reasoning_content, extra=extra))
                continue

            # Reasoning content is placed in a separate message
            if reasoning_content:
                new_messages.append(Message(role=role, content='', reasoning_content=reasoning_content, extra=extra))

            new_content = []
            for item in content:
                item_type, item_text = item.get_type_and_value()

                if item_type != 'text':  # multimodal
                    new_content.append(item)
                    continue
                # Do not parse <tool_call> in thought!!!
                if '<think>' in item_text:
                    thought_in_content = True
                if thought_in_content:
                    _item_text = item_text.split('</think>')
                    # assert len(_item_text) == 2
                    new_content.append(ContentItem(text='</think>'.join(_item_text[:-1]) + '</think>'))
                    item_text = _item_text[-1]

                i = item_text.find('<tool_call>')
                # If no function call:
                if i < 0:
                    show_text = item_text
                    if show_text:
                        new_content.append(ContentItem(text=show_text))
                    continue

                # split tool-call to separate assistant msg
                tool_call_list = item_text.split('<tool_call>')
                pre_thought = tool_call_list[0]
                if pre_thought.strip():
                    new_content.append(ContentItem(text=pre_thought))
                for txt in tool_call_list[1:]:
                    if not txt.strip():
                        continue

                    if '</tool_call>' not in txt:
                        # incomplete </tool_call>: This is to better represent incomplete tool calls in streaming output
                        fn_name, fn_args = extract_fn(txt)
                        if fn_name:  # need to call function
                            if new_content:
                                new_messages.append(Message(
                                    role=role,
                                    content=new_content,
                                    extra=extra,
                                ))  # split thought and function call
                                new_content = []
                            # TODO: process incomplete tool-call messages
                            _extra = copy.deepcopy(extra) if extra else {'function_id': ''}
                            _extra['function_id'] = str(tool_id)
                            tool_id += 1
                            new_messages.append(
                                Message(
                                    role=ASSISTANT,
                                    content=[],
                                    function_call=FunctionCall(
                                        name=fn_name,
                                        arguments=fn_args,
                                    ),
                                    extra=_extra,
                                ))
                        continue

                    one_tool_call_txt = txt.split('</tool_call>')

                    # The complete tool-call response
                    if new_content:
                        new_messages.append(Message(
                            role=role,
                            content=new_content,
                            extra=extra,
                        ))  # split thought and function call
                        new_content = []
                    fn = None
                    if SPECIAL_CODE_MODE and '<code>' in one_tool_call_txt[0] and '</code>' in one_tool_call_txt[0]:
                        _snips = one_tool_call_txt[0].split('<code>')
                        for i, _s in enumerate(_snips):
                            if i == 0:
                                fn = json5.loads(_s)
                            else:
                                # TODO: support more flexible params
                                code = _s.replace('</code>', '')
                                fn['arguments']['code'] = code
                    else:
                        try:
                            fn = json5.loads(one_tool_call_txt[0].strip())
                        except Exception:
                            logger.warning('Invalid json tool-calling arguments')
                            fn_name, fn_args = extract_fn(one_tool_call_txt[0].strip())
                            _extra = copy.deepcopy(extra) if extra else {'function_id': ''}
                            _extra['function_id'] = str(tool_id)
                            tool_id += 1
                            new_messages.append(
                                Message(
                                    role=ASSISTANT,
                                    content=[],
                                    function_call=FunctionCall(
                                        name=fn_name,
                                        arguments=fn_args,
                                    ),
                                    extra=_extra,
                                ))
                    if fn and 'name' in fn and 'arguments' in fn:
                        _extra = copy.deepcopy(extra) if extra else {}
                        _extra['function_id'] = str(tool_id)
                        tool_id += 1
                        new_messages.append(
                            Message(
                                role=ASSISTANT,
                                content=[],
                                function_call=FunctionCall(
                                    name=fn['name'],
                                    arguments=json.dumps(fn['arguments'], ensure_ascii=False),
                                ),
                                extra=_extra,
                            ))
                    # Expected not to output extra tails
                    # if one_tool_call_txt[1].strip():
                    #     new_content.append(ContentItem(text=one_tool_call_txt[1]))

            if new_content:
                new_messages.append(Message(role=role, content=new_content, extra=extra))
        return new_messages


def call_llm_api(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
):
    client = OpenAI(api_key=api_key, base_url=api_base)
    dynamic_model_id = model_name
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(
        model=dynamic_model_id,
        messages=messages,
        temperature=temperature,
        extra_body={"max_completion_tokens": max_tokens}
    )
    response_content = response.choices[0].message.content
    messages.append(
        {"role": "assistant", "content": response_content}
    )

    return messages


def call_qwen_agent(
    user_prompt: str,
    system_prompt: str,
    image_path: str,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
    use_tools: bool = True,
    use_thinking: bool = True
):
    # client = OpenAI(api_key=api_key, base_url=api_base)
    # try:
    #     models = client.models.list()
    #     dynamic_model_id = models.data[0].id if models.data else model_name
    # except Exception:
    #     dynamic_model_id = model_name
    dynamic_model_id = model_name

    llm_cfg = {
        'model_type': 'qwenvl_oai',
        'model': dynamic_model_id,
        'model_server': api_base,
        'api_key': api_key,
        # (Optional) LLM hyperparameters for generation:
        'generate_cfg': {
            'temperature': temperature,
            'max_tokens': max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": use_thinking}},
        },
    }
    analysis_prompt = """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

    Start an iterative loop for each question:

    - **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
    - **Next, find information:** Use a tool to research the things you need to find out.
    - **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

    Continue this loop until your research is complete.

    To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question."""

    # analysis_prompt = "You are a helpful assistant."

    tools = ['image_zoom_in_tool','web_search']if use_tools else []
    bot = Assistant(llm=llm_cfg,
                    system_message=analysis_prompt,
                    function_list=tools)
                
    
    bot.llm.fncall_prompt = NousFnCallPrompt2()
    
    if image_path:
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": user_prompt},
                    {"image": image_path}
                ]
            }
        ]
   
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": user_prompt}
                ]
            }
        ]

    
    responses = []
    for response in bot.run_nonstream(messages=messages):
        responses.append(response)
    messages.extend(responses)
    
    return messages

def generate_answer(cfg, question,image_path):
    prompt = f"""Question: {question}

Multiple tool invocations are permitted. Prior to each tool invocation, you must articulate your analytical reasoning.

Critical reminder: Tools operate in a stateless manner and will not preserve any variables, functions, or definitions from prior executions.

Reasoning step by step, write ONLY the final answer enclosed in <answer> and </answer> tags. Do not include any extra text outside the tags after the final answer. /no_think
"""
    answer_messages = call_qwen_agent(
        user_prompt=prompt,
        system_prompt="",
        image_path=image_path,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        use_tools=cfg.use_tools,
        use_thinking=cfg.use_thinking
    )

    return answer_messages

def answer_evaluate(cfg, question, answers):

    prompt = f"""
    Your task: Compare the evaluated answer to the true answer and determine if they are factually and semantically equivalent.
    Question: {question}
    True answer (factually correct): {answers[0]}
    Evaluated answer: {answers[1]}
    
    Rules:
    1. The true answer is correct. Mark as equivalent only if both answers have the same facts and conclusion.
    Even if they use different wording or provide different levels of explanation, they are equivalent if the meaning and final conclusion are the same.
    2. If the evaluated answer has any factual error, missing information, or contradiction with the true answer, mark as {{"equivalent": false}}.
    3. Ignore wording/style differences, focus on factual equality.

    Respond ONLY in strict JSON format with the 'equivalent' field. 
    Examples:
    {{"equivalent": true}}
    {{"equivalent": false}}
    """
    evaluation_messages = call_qwen_agent(
        user_prompt=prompt,
        system_prompt="",
        image_path=None,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        use_tools=cfg.use_tools
    )

    return evaluation_messages[-1]["content"]
