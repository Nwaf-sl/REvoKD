import json
from typing import Dict, List
from collections import Counter

tool_parameters_schema = {
    "image_zoom_in_tool": {
        "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner"
                },
                "label": {
                    "type": "string",
                    "description": "The name or label of the object in the specified bounding box"
                },
                "img_idx": {
                    "type": "number",
                    "description": "The index of the zoomed-in image (starting from 0)"
                }
            },
            "required": ["bbox_2d", "label", "img_idx"]
        }
    },
    "web_search":{
        "description":'Search for information from the internet.',
        "parameters" :{
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'A fully-formed semantic search query'
                },
                'num_results': {
                    'type': 'integer',
                    'description': 'Number of results returned (default 5)'
                }
            },
            'required': ['query'],
        }}
    }

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue  
            if not (raw_line.startswith("{") or raw_line.startswith("[")):
                continue 
            try:
                data.append(json.loads(raw_line))
            except json.JSONDecodeError as e:
                continue
    return data

def generate_all_tools(tool_parameters_schema: Dict) -> List[Dict]:
    tools_list = []
    for name, tool_info in tool_parameters_schema.items():
        description = tool_info.get("description", "")
        parameters_schema = tool_info.get("parameters", {})
        tools_list.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters_schema
            }
        })
    return tools_list

def fvqa_to_target_format(fvqa_item: Dict, tool_parameters_schema: Dict) -> Dict:
    tools_list = generate_all_tools(tool_parameters_schema)
    tools_str = json.dumps(tools_list, ensure_ascii=False)  

    messages = []
    images = []

    for step in fvqa_item.get("solution", []):
        role = step.get("role")

        if role == "user":
            for content in step.get("content", []):
                if "text" in content:
                    messages.append({"role": "user", "content": content["text"]})
                if "image" in content:
                    images.append(content["image"])
                    messages.append({"role": "user", "content": "<image>"})

        elif role == "assistant":
            if step.get("content"):
                messages.append({"role": "assistant", "content": step["content"]})
            if "function_call" in step:
                fc = step["function_call"]
                arguments = fc.get("arguments", {})
                if isinstance(arguments, str):
                    if arguments.strip():
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            print(f"⚠️ function_call.arguments: {arguments}")
                            arguments = {}
                messages.append({
                    "role": "tool_call",
                    "content": json.dumps({
                        "name": fc.get("name", ""),
                        "arguments": arguments
                    }, ensure_ascii=False)
                })

        elif role == "function":
            tool_name = step.get("name", "")
            step_content = step.get("content")

            if isinstance(step_content, list):
                for c in step_content:
                    if "image" in c:
                        images.append(c["image"])
                        messages.append({
                            "role": "tool_response",
                            "content": json.dumps({
                                "images": "<image>",
                                "status": "success"
                            }, ensure_ascii=False)
                        })
                    elif "text" in c:
                        messages.append({
                            "role": "tool_response",
                            "content": json.dumps({
                                "output": c["text"]
                            }, ensure_ascii=False)
                        })
            elif isinstance(step_content, str):
                messages.append({
                    "role": "tool_response",
                    "content": json.dumps({
                        "output": step_content
                    }, ensure_ascii=False)
                })

    return {
        "tools": tools_str,
        "messages": messages,
        "images": images
    }



def has_tool_call(fvqa_item: Dict) -> bool:
    for step in fvqa_item.get("solution", []):
        if step.get("role") == "assistant" and "function_call" in step:
            return True
    return False

def count_categories(jsonl_path: str):
    counter = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                cat = obj.get("category")
                if cat:
                    counter[cat] += 1
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    input_path = "you/path"
    output_path = "you/out/path"

    fvqa_data = load_jsonl(input_path)

    count_total = len(fvqa_data)
    count_saved = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in fvqa_data:
            if item.get("is_correct", False) and has_tool_call(item):
                converted = fvqa_to_target_format(item, tool_parameters_schema)
                converted["answer"] = item.get("answer")
                converted["true_answer"] = item.get("true_answer")
                converted["category"] = item.get("category")
                f_out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count_saved += 1
    count_categories(output_path)