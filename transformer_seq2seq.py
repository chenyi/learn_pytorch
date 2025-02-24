import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import argparse

# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, emb_size=256, n_heads=8, num_layers=6):
        super(Transformer, self).__init__()
        self.encoder = nn.Embedding(src_vocab_size, emb_size)
        self.decoder = nn.Embedding(trg_vocab_size, emb_size)
        self.transformer = nn.Transformer(d_model=emb_size, nhead=n_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(emb_size, trg_vocab_size)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def forward(self, src, trg):
        src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
        trg_mask = self._generate_square_subsequent_mask(len(trg)).to(trg.device)
        
        src_emb = self.encoder(src)
        trg_emb = self.decoder(trg)
        
        output = self.transformer(src_emb, trg_emb, src_mask=src_mask, tgt_mask=trg_mask)
        return self.fc_out(output)

    def _generate_square_subsequent_mask(self, sz):
        """生成自注意力掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)  # 批次大小
    parser.add_argument('--learning_rate', type=float, default=0.0001)  # 学习率
    parser.add_argument('--epochs', type=int, default=10)  # 训练轮数
    parser.add_argument('--print_freq', type=int, default=50)  # 打印频率
    parser.add_argument('--data_dir', type=str, default='./data')  # 数据集路径
    parser.add_argument('--save_dir', type=str, default='./checkpoints')  # 模型保存路径
    return parser.parse_args()

def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    if not os.environ.get('MASTER_ADDR'):
        os.environ['MASTER_ADDR'] = 'localhost'  # 主节点地址
    if not os.environ.get('MASTER_PORT'):
        os.environ['MASTER_PORT'] = '12355'  # 通信端口
    
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)  # 初始化进程组

def train(rank, world_size, args):
    """主训练循环"""
    setup_distributed(rank, world_size)  # 设置分布式环境

    # 定义源语言和目标语言的字段
    SRC = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = Field(tokenize='spacy', tokenizer_language='fr_core_news_sm', init_token='<sos>', eos_token='<eos>', lower=True)

    # 加载数据集
    train_data, valid_data, test_data = Multi30k.splits(exts=('.en', '.fr'), fields=(SRC, TRG))

    # 构建词汇表
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    # 创建数据加载器
    train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), batch_size=args.batch_size, device=torch.device(f'cuda:{rank % torch.cuda.device_count()}'))

    # 初始化模型
    model = Transformer(len(SRC.vocab), len(TRG.vocab), SRC.vocab.stoi[SRC.pad_token], TRG.vocab.stoi[TRG.pad_token]).to(torch.device(f'cuda:{rank % torch.cuda.device_count()}'))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])  # 忽略填充标记

    model.train()
    for epoch in range(args.epochs):
        for batch in train_iterator:
            src = batch.src  # 源语言
            trg = batch.trg  # 目标语言

            optimizer.zero_grad()  # 清零梯度
            output = model(src, trg[:-1, :])  # 目标语言去掉最后一个标记

            # 计算损失
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)  # 展平
            trg = trg[1:, :].view(-1)  # 目标语言去掉第一个标记并展平
            loss = criterion(output, trg)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if batch.batch_idx % args.print_freq == 0 and rank == 0:
                print(f'Epoch: {epoch}, Batch: {batch.batch_idx}, Loss: {loss.item()}')

def main():
    args = parse_args()
    
    # 从环境变量获取分布式信息
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 直接调用训练函数
    train(rank, world_size, args)

if __name__ == "__main__":
    main() 