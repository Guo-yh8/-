from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch

# 加载训练好的模型和分词器
model = BertForQuestionAnswering.from_pretrained('./bert_qa_model')
tokenizer = BertTokenizerFast.from_pretrained('./bert_qa_model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def answer_question(question, context):
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)

    if start_idx > end_idx:
        # 防止异常情况，保证start<=end
        return ""

    answer_ids = inputs['input_ids'][0, start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return answer.strip()

if __name__ == "__main__":
    print("请输入上下文文本：")
    context = input().strip()
    print("上下文加载完毕。现在你可以提问，输入exit或空行退出。")

    while True:
        question = input("请输入问题：").strip()
        if question == "" or question.lower() == "exit":
            print("退出程序。")
            break
        answer = answer_question(question, context)
        print("回答：", answer if answer else "[无回答]")


