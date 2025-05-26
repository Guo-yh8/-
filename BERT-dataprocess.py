import json
import re

txt_path = "问答对.txt"
json_path = "问答对.json"

def txt_to_json(filepath_in, filepath_out):
    with open(filepath_in, 'r', encoding='utf-8') as f:
        text = f.read()

    # 去掉章节标记，比如 “（一）” 和 “（二）”
    text = re.sub(r'（.*?）', '', text)

    # 按空行分割，得到问答块列表
    blocks = [block.strip() for block in text.split('\n\n') if block.strip()]

    data = []
    for block in blocks:
        # 确保block里有“问：”和“答：”
        if '问：' in block and '答：' in block:
            # 提取原始 context（不去换行）
            context = block.replace('\n', '').strip()

            # 正则提取问和答
            q_match = re.search(r'问：(.*?)(?:答：|$)', block, re.S)
            a_match = re.search(r'答：(.*)', block, re.S)

            if q_match and a_match:
                question = q_match.group(1).strip()
                answer_text = a_match.group(1).strip()

                # 找到 answer_text 在 context 中的位置
                answer_start = context.find(answer_text)

                if answer_start == -1:
                    print(f"警告：在 context 中找不到 answer：{answer_text}")
                    continue

                data.append({
                    "context": context,
                    "question": question,
                    "answer_text": answer_text,
                    "answer_start": answer_start
                })

    # 保存为 JSON
    with open(filepath_out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"转换完成，共 {len(data)} 条，保存到 {filepath_out}")

if __name__ == "__main__":
    txt_to_json(txt_path, json_path)
