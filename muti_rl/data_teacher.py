
import json
import base64
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

model_name = "Qwen3-VL-235B-A22B-Instruct"
openai_api_key = ""
openai_api_base = ""

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base
)

# ===== Prompt =====
prompt_template = """
You are a visual reasoning teacher model.
You will be given:
- An image related to a visual reasoning task.
- The original question.
- The student's reasoning process.
- The student's answer.
- The correct answer.

Your task:
1. Review the reasoning process and compare it with the correct answer.
2. Identify specific mistakes or missing observations in the student's reasoning.
3. Provide a short guidance tip to help the student improve 
   their reasoning or visual observation next time.
4. DO NOT reveal the correct answer directly.
5. Focus on actionable, concrete advice specific to the given example.

Output Requirements:
Only output a single JSON object:
{
    "teacher_prompt": "<your short guidance tip>"
}
No additional commentary or text outside the JSON.
"""

def encode_image_to_data_url(img_path):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(img_path)[1].lower().strip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"

# ===== 单条处理函数 =====
def process_record(data):
    rec_id = data.get("id")
    image_path = data.get("image")
    question = data.get("question", "")
    solution_text = json.dumps(data.get("solution", ""), ensure_ascii=False)  # 保留推理过程的结构
    student_answer = data.get("answer", "")
    true_answer = data.get("true_answer", "")

    if data.get("is_correct") is not False:
        return None, f"⏩ Skip correct sample: {rec_id}", rec_id

    if not image_path or not os.path.exists(image_path):
        return None, f"❌ Image not found: {image_path}", rec_id

    image_data_url = encode_image_to_data_url(image_path)

    user_prompt_text = f"""{prompt_template}

Question: {question}

Student reasoning: {solution_text}

Student answer: {student_answer}

Correct answer: {true_answer}
"""

    messages = [
        {"role": "system", "content": "You are a helpful teacher."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url}
                },
                {
                    "type": "text",
                    "text": user_prompt_text
                }
            ]
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=512,
            timeout=30
        )
        result = json.loads(resp.choices[0].message.content)
        merged = {**data, "teacher_prompt": result.get("teacher_prompt")}
        return merged, None, rec_id
    except Exception as e:
        return None, str(e), rec_id


input_jsonl = "train.json"
output_jsonl = "teacher_guidance.jsonl"

processed_ids = set()
if os.path.exists(output_jsonl):
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                processed_ids.add(json.loads(line.strip()).get("id"))
            except:
                pass

records = []
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line.strip())
        if rec.get("id") not in processed_ids:
            records.append(rec)
        else:
            print(f"⏩ Skip {rec.get('id')} (already processed)")

print(f"ℹ {len(records)} tasks to run.")

with ThreadPoolExecutor(max_workers=6) as executor, open(output_jsonl, "a", encoding="utf-8") as fout:
    future_map = {executor.submit(process_record, r): r for r in records}
    for future in as_completed(future_map):
        data, err, rec_id = future.result()
        if data:
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()
            print(f" Done {rec_id}")
        if err:
            print(f"⚠ Error {rec_id}: {err}")