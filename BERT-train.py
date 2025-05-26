import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from BERT import get_model
from tqdm import tqdm

# 数据集类（改为接受列表数据）
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer_text = item['answer_text']
        answer_start = item['answer_start']

        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        offset_mapping = encoding['offset_mapping'].squeeze()

        start_char = answer_start
        end_char = answer_start + len(answer_text)

        start_token_idx, end_token_idx = 0, 0
        for idx, (start, end) in enumerate(offset_mapping.tolist()):
            if start <= start_char < end:
                start_token_idx = idx
            if start < end_char <= end:
                end_token_idx = idx
                break

        if start_token_idx == 0 and end_token_idx == 0:
            start_token_idx = 0
            end_token_idx = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor(start_token_idx),
            'end_positions': torch.tensor(end_token_idx)
        }

def train():
    # 设置
    data_path = '问答对.json'
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 加载并划分数据
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42)

    # 构建数据集和加载器
    train_dataset = QADataset(train_data, tokenizer)
    val_dataset = QADataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # 开始训练
    model.train()
    epochs = 20
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_loss = 0

        # ------- 训练 -------
        model.train()
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        print(f"训练loss: {avg_train_loss:.4f}")

        # ------- 验证 -------
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        print(f"验证loss: {avg_val_loss:.4f}")

    # 保存模型
    model.save_pretrained('./bert_qa_model')
    tokenizer.save_pretrained('./bert_qa_model')
    print("训练完成，模型已保存到 ./bert_qa_model")

    # 绘制训练和验证loss曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_loss_list, marker='o', label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss_list, marker='s', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == '__main__':
    train()



