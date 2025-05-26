import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from model import LanguageTransformer
from collections import Counter
import matplotlib.pyplot as plt
import time

# 加载数据
def load_qa_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    qa_pairs = []
    question, answer = None, None

    for line in lines:
        line = line.strip()
        if line.startswith("问："):
            # 遇到新问之前，先保存上一组（如果完整）
            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
                question, answer = None, None
            question = line[2:].strip()
        elif line.startswith("答："):
            answer = line[2:].strip()
        else:
            # 其它非空行忽略
            pass

    # 文件结尾补充最后一条
    if question and answer:
        qa_pairs.append({
            "question": question,
            "answer": answer
        })

    return qa_pairs


def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text)
    tokens = [token for token, freq in counter.items() if freq >= min_freq]
    vocab = {token: idx for idx, token in enumerate(tokens)}
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab) + 1
    return vocab


def encode(text, vocab, pad_size):
    ids = [vocab.get(ch, vocab['<UNK>']) for ch in text]
    if len(ids) < pad_size:
        ids += [vocab['<PAD>']] * (pad_size - len(ids))
    else:
        ids = ids[:pad_size]
    return ids


def decode(ids, idx2word):
    # 将id序列解码为字符串
    tokens = [idx2word.get(i, '') for i in ids]
    return ''.join(tokens).replace('<PAD>', '').replace('<UNK>', '')


class Tokenizer:
    def __init__(self, vocab):
        self.word2idx = vocab
        self.idx2word = {idx: word for word, idx in vocab.items()}

    def encode(self, text):
        return [self.word2idx.get(ch, self.word2idx['<UNK>']) for ch in text]

    def decode(self, ids):
        return decode(ids, self.idx2word)

# 自定义数据集
class QADataset(Dataset):
    def __init__(self, texts, vocab, pad_size):
        self.vocab = vocab
        self.pad_size = pad_size
        self.data = [torch.tensor(encode(text, vocab, pad_size), dtype=torch.long) for text in texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][:-1]  # 输入序列
        y = self.data[idx][1:]   # 目标序列（下一个字符）
        return x, y


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    model.to(device)
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()  # 记录epoch开始时间

        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        end_time = time.time()  # 记录epoch结束时间
        epoch_time = end_time - start_time

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Time = {epoch_time:.2f} seconds")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'config': {
                    'embed': embed,
                    'dim_model': dim_model,
                    'num_layer': num_layer,
                    'num_head': num_head,
                    'hidden': hidden,
                    'pad_size': pad_size,
                    'dropout': dropout
                }
            }, "best_model.pth")
            print("模型已保存！")

    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train & Validation Loss')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # 参数配置
    file_path ="扩充问答对10万条.txt"
    pad_size = 256
    embed = 256
    dim_model = 256
    num_layer = 2
    num_head = 4
    hidden = 512
    dropout = 0.3
    batch_size = 16
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    qa_pairs = load_qa_data(file_path)
    texts = [f"问：{q['question']} 答：{q['answer']}" for q in qa_pairs]
    vocab = build_vocab(texts, min_freq=1)
    n_vocab = len(vocab)
    tokenizer = Tokenizer(vocab)

    # 准备数据集
    dataset = QADataset(texts, vocab, pad_size)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    model = LanguageTransformer(
        embed=embed,
        n_vocab=n_vocab,
        num_layer=num_layer,
        dim_model=dim_model,
        num_head=num_head,
        hidden=hidden,
        pad_size=pad_size,
        dropout=dropout,
        device=device
    )

    # 训练设置
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 开始训练
    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)