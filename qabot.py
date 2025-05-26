import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from model import LanguageTransformer
from train import Tokenizer

# 设置中文字体
# 这里用SimHei作为示范
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_token_probs(logits, tokenizer, top_k=10):
    """
    绘制当前step预测的top_k词概率条形图
    logits: 1D Tensor, vocab大小
    tokenizer: 具有 idx2word 属性（索引转词）
    """
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    top_indices = probs.argsort()[-top_k:][::-1]
    top_probs = probs[top_indices]
    top_tokens = [tokenizer.idx2word[idx] for idx in top_indices]

    plt.figure(figsize=(10, 4))
    sns.barplot(x=top_tokens, y=top_probs, palette="viridis")
    plt.title('Top-{} 词的预测概率'.format(top_k))
    plt.ylabel('概率')
    plt.xlabel('词')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

class QABot:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(checkpoint_path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.vocab = checkpoint['vocab']
        self.tokenizer = Tokenizer(self.vocab)

        if not hasattr(self.tokenizer, 'idx2word'):
            self.tokenizer.idx2word = {idx: word for word, idx in self.tokenizer.word2idx.items()}

        self.model = LanguageTransformer(
            embed=self.config['embed'],
            n_vocab=len(self.vocab),
            num_layer=self.config['num_layer'],
            dim_model=self.config['dim_model'],
            num_head=self.config['num_head'],
            hidden=self.config['hidden'],
            pad_size=self.config['pad_size'],
            dropout=self.config['dropout'],
            device=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def generate_answer(self, question, max_len=100, visualize_topk=10):
        input_text = f"问：{question} 答："
        input_ids = self.tokenizer.encode(input_text)

        for step in range(max_len):
            inputs = input_ids[-self.config['pad_size']:]
            input_tensor = torch.tensor(inputs, dtype=torch.long).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs[0, -1, :]
                next_token = torch.argmax(logits).item()

            input_ids.append(next_token)
            if next_token == self.tokenizer.word2idx.get('。', -1):
                break

        # 只画最后一步的概率分布
        plot_token_probs(logits, self.tokenizer, top_k=visualize_topk)

        answer = self.tokenizer.decode(input_ids)
        return answer.split("答：")[-1].strip()

if __name__ == "__main__":
    bot = QABot("best_model.pth")
    print("问答系统已加载，输入'退出'结束对话")

    while True:
        question = input("\n请输入问题：")
        if question == '退出':
            break
        answer = bot.generate_answer(question)
        print(f"回答：{answer}")

