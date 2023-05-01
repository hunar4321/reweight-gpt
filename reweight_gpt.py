'''
Reweight GPT
Author: Hunar Ahmad @ brainxyz.com
This method uses learnable lateral connections to reweight the inputs instead of the self-attention mechanism (which are commented).
To learn more about the method, watch this video (from 41:26): https://youtu.be/l-CjXFmcVzY
'''
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.nn import functional as F

with open('data/file.txt', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.lower()

chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
data = [stoi[c] for c in text]
vocab_size = len(chars)

device = 'cpu'
ins = 16
outs = vocab_size
nodes = 32
lr = 0.001
n_emb = 32

embed = torch.randn(vocab_size, n_emb)
pos = torch.randn(ins, n_emb)
embed = embed.to(device)
pos = pos.to(device)
data = torch.tensor(data).long()
params = []

def weights(ins, outs):
    ws = torch.randn(ins, outs)*0.1
    ws = ws.to(device)
    ws = ws.requires_grad_(True)
    params.append(ws)
    return ws

class Head():
    def __init__(self):
        '''
        if you want to compare this method to self-attention, uncomment the comments and remove attn = x @ self.wr
        '''
        self.wv = weights(n_emb, n_emb//4)
        # self.wq = weights(n_emb, n_emb//4)
        # self.wk = weights(n_emb, n_emb//4)
        self.wr = weights(n_emb, ins)
        
    def forward(self, x):
        v = x @ self.wv
        # q = x @ self.wq
        # k = x @ self.wk
        # attn = (q @ k.transpose(-2,-1)) / k.shape[0]**0.5
        attn = x @ self.wr
        tril = torch.tril(attn)
        tril = tril.masked_fill(tril==0, -1e10)
        rew = F.softmax(tril, dim=-1)
        x = rew @ v
        return x     
    
class Block():
    def __init__(self):
        self.heads = [Head(), Head(), Head(), Head()]
        self.w0 = weights(n_emb, nodes)
        self.w1 = weights(nodes, n_emb)

    def forward(self, x):        
        x = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        x = torch.relu(x @ self.w0)
        x = torch.relu(x @ self.w1)
        return x
        
class Model():
    def __init__(self):
        self.blocks = [Block(), Block(), Block()]
        self.w2 = weights(n_emb, outs)
        
    def forward(self, x):
        x = embed[x] + pos
        x = x + self.blocks[0].forward(x)
        x = x + self.blocks[1].forward(x)
        x = x + self.blocks[2].forward(x)
        yh = (x @ self.w2)
        return yh
        
model = Model()   
optimizer = torch.optim.Adam(params, lr)
print("params:", sum(p.numel() for p in params))

import time
t = time.time()

ers = []
for i in range(5000):

    b = torch.randint(len(data)-ins, (100, ))
    xs = torch.stack([data[i:i+ins] for i in b])
    ys = torch.stack([data[i+1:i+ins+1] for i in b]) 
    xs = xs.to(device)
    ys = ys.to(device)

    yh = model.forward(xs)
    
    loss = F.cross_entropy(yh.view(-1, vocab_size) , ys.long().view(-1)) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    e = loss.item()
    if i % 500 == 0:
        print("loss:", e)
    ers.append(e)

print("time:", time.time()-t)    

s = xs[0]
gen_text = ""
for i in range(3000):
    yh = model.forward(s)
    prob = F.softmax(yh[-1, :]*1, dim=0)
    # pred = torch.argmax(yh[-1, :]).item()
    pred = torch.multinomial(prob, num_samples=1).item()
    s = torch.roll(s, -1)
    s[-1] = pred
    gen_text += itos[pred]

print(gen_text)      
