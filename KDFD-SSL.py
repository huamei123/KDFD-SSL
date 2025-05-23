import torch
import numpy as np
import openml
from torch import nn, einsum
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.optim as optim
import os
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt_sne
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 0
np.random.seed(seed)
data = pd.read_csv("C:/Users\HNU\Desktop\先验知识\code\summer3.csv")
data = data.sample(n=len(data['label']),random_state=0, axis=0)
data = data.reset_index(drop=True)
data_test = data

datasplit = [0.7, 0.3]
data["Set"] = np.random.choice(["train", "test"], p=datasplit, size=(data.shape[0],))
train_indices = data[data.Set == "train"].index
test_indices = data[data.Set == "test"].index
data = data.drop(columns=['Set'])

X1 = data.iloc[:, 1:]
y1 = data.iloc[:, 0]
X1 = X1.reset_index(drop=True).values
y1 = y1.reset_index(drop=True).values
x_train = X1[train_indices]
y_train = y1[train_indices].flatten()
x_test = X1[test_indices]
y_test = y1[test_indices].flatten()
X = []
y = []
x_train = pd.DataFrame(x_train)
y_train = pd.Series(y_train.tolist())

for i in range(7):
    a = y_train[y_train == i].index[:5]
    tempx = x_train[y_train == i][:5]
    tempy = pd.Series(1*i, index=list(range(5)))
    X.append(tempx)
    y.append(tempy)
    x_train = x_train.drop(a, axis=0)
    y_train = y_train.drop(a, axis=0)
Labeled_train = pd.concat(X)
Labeled_train = Labeled_train.reset_index(drop=True).values
Label = pd.concat(y)
Label = Label.reset_index(drop=True).values
x_train = x_train.values
pv = x_train[:, :11]

x_train = x_train.astype(float)
Labeled_train = Labeled_train.astype(float)
x_train = torch.from_numpy(x_train).to(torch.float32).to(device)
Labeled_train = torch.from_numpy(Labeled_train).to(torch.float32).to(device)
Label = torch.from_numpy(Label).to(device)
pv_t = torch.from_numpy(pv).to(torch.float32).to(device)

# clean_X_test = [float(s.strip().replace(',', '')) if s.strip() != '#NAME?' else np.nan for s in X_test]
x_test = x_test.astype(float)
testdata = torch.from_numpy(x_test).to(torch.float32).to(device)
testlabel = torch.from_numpy(y_test).to(device)


bs = 32
train_ds = Data.TensorDataset(x_train, pv_t)
trainloader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)

class layer1(nn.Module):
    def __init__(self):
        super(layer1,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(144,72),

        )
        self.decoder=nn.Sequential(
            nn.Linear(72,144),

        )
    def forward(self,x):
        encode=self.encoder(x)
        decode=self.decoder(encode)
        return encode, decode

class layer2(nn.Module):
    def __init__(self,layer1):
        super(layer2,self).__init__()
        self.layer1=layer1
        self.encoder=nn.Sequential(
            nn.Linear(72, 11),
        )
        self.decoder=nn.Sequential(
            nn.Linear(11, 144),
        )

    def forward(self,x):
        self.layer1.eval()
        x,_=self.layer1.forward(x)
        encode=self.encoder(x)
        decode=self.decoder(encode)
        return encode, decode


def train_layer(layer,k):#layer1,1
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001, betas=(0.99, 0.999))

    for epoch in range(65):
        # i = 0
        for data, _ in trainloader:
            data = data
            encoded, decoded = layer.forward(data)
            loss = loss_fn(decoded, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if i % 50 == 0:
            #     print(loss)
            # i += 1
        # print("训练次数：", epoch)
        # print("重建损失：", loss)
        print("epoch:%d,loss:%f"%(epoch,loss))
        torch.save(layer.state_dict(), 'C:/Users\HNU\Desktop\先验知识\code\代码\SAE_layer/layer%d.pkl')

layer1 = layer1().to(device)
if os.path.exists("C:/Users\HNU\Desktop\先验知识\code\代码/layer1.pkl"):
    layer1.load_state_dict(torch.load('C:/Users\HNU\Desktop\SAE1/result/layer11.pkl'))
else:
    train_layer(layer1, 1)

layer2 = layer2(layer1).to(device)

if os.path.exists("C:/Users\HNU\Desktop\先验知识\code\代码/layer2.pkl"):
    layer2.load_state_dict(torch.load("C:/Users\HNU\Desktop\先验知识\code\代码/layer21.pkl"))
else:
    train_layer(layer2, 2)


N = 4 #编码器个数
input_dim = 144
seq_len = 16 #句子长度
d_model = 64 #词嵌入维度
d_ff = 512 #全连接层维度
head = 4 #注意力头数
dropout = 0.1
lr = 3E-5 #学习率
batch_size = 16

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = rearrange(x, 'b 1 n l -> b n l')
        # [batch_size, num_patches+1, inner_dim*3] --> ([batch_size, num_patches+1, inner_dim], -->(q,k,v)
        #                                               [batch_size, num_patches+1, inner_dim],
        #                                               [batch_size, num_patches+1, inner_dim])
        #将x转变为qkv
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 对tensor进行分块

        q, k, v = \
            map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'b n l -> b 1 n l')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, ResNetBlock()),
                # PreNorm(dim//(2**i), ConvAttention(dim//(2**i), heads, dim_head//(2**i), dropout)),
                # PreNorm(dim//(2**(i+1)), FeedForward(dim//(2**(i+1)), mlp_dim, dropout))
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


class TimeTransformer(nn.Module):
    def __init__(self, *, input_dim,  num_patches=9, dim, depth, heads, mlp_dim,
                 pool='cls', channels=1, dim_head, emb_dropout=0., dropout=0.):
        super(TimeTransformer, self).__init__()

        # self.to_patch_embedding = Embedding(input_dim, dim)
        self.to_patch_embedding = self.to_patch_embedding = nn.Sequential(
            Rearrange('b 1 (n d) -> b 1 n d', n=num_patches),
            nn.Linear(input_dim//num_patches, dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))  # [1, 1, 1, dim] 随机数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # [1, num_patches+1, dim] 随机数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()  # 这个恒等函数，就如同名字占位符，并没有实际操作

    def forward(self, rawdata):
        TimeSignals = rawdata   # Get Time Domain Signals
        TimeSignals = rearrange(TimeSignals, 'b l -> b 1 l')
        # print(TimeSignals.shape, rawdata.shape)

        x = self.to_patch_embedding(TimeSignals)
        b, _, n, _ = x.shape      # x: [batch_size, channels, num_patches, dim]

        cls_tokens = repeat(self.cls_token, '() c n d -> b c n d', b=b)  # cls_tokens: [batch_size, c, num_patches, dim]
        x = torch.cat((cls_tokens, x), dim=2)  # x: [batch_size, c, num_patches+1, dim]
        # print(x.shape)
        #x += self.pos_embedding[:, :(n + 1)]  # 添加位置编码：x: [batch_size, c, num_patches+1, dim]
        x = self.dropout(x)

        x = self.transformer(x)     # x: [batch_size, c, num_patches+1, dim]
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, :, 0]  # x: [batch_size, c, 1, dim]
        x = self.to_latent(x)
        return x


class DSCTransformer(nn.Module):
    def __init__(self, *, input_dim, dim, depth, heads, mlp_dim, pool='cls',
                 num_classes, channels=1, dim_head, emb_dropout=0., dropout=0.):
        super(DSCTransformer, self).__init__()
        self.in_dim_time = input_dim
        self.TimeTrans = TimeTransformer(input_dim=self.in_dim_time, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                         pool=pool, dim_head=dim_head, emb_dropout=emb_dropout, dropout=dropout)

        self.mlp_head = nn.Sequential(
            Rearrange('b c l -> b (l c)'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        TimeSignals = x
        TimeFeature = self.TimeTrans(TimeSignals)
        y = self.mlp_head(TimeFeature)

        return y  # [batch_size, 1, num_classes]


model = DSCTransformer(input_dim=input_dim, num_classes=22, dim=d_model, depth=N,
                       heads=head, mlp_dim=d_ff, dim_head=d_model, emb_dropout=dropout, dropout=dropout).to(device)
optimizerT = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.99,0.999))
criterion1 = nn.MSELoss().to(device)

for epoch in range(65):
    running_loss = 0.0
    for data, pv_t in trainloader:
        model.train()
        optimizerT.zero_grad()
        output = model(data)
        gv_t, _ = layer2(data)
        pro_labels = torch.cat((pv_t, gv_t), dim=1)
        loss = criterion1(output, pro_labels)/1000
        loss.backward()
        optimizerT.step()
        running_loss += loss.item()
    print(running_loss)

criterion2 = nn.CrossEntropyLoss().to(device)

class Cls(nn.Module):
    def __init__(self):
        super(Cls,self).__init__()
        # self.l1 = nn.Linear(11,7)
        self.l0 = model
        self.l1 = nn.Linear(22,100)
        self.l2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100,22)
        )
    def forward(self,x):

        x_t = self.l0(x)
        e_feature = self.l1(x_t)
        return x_t, e_feature, F.log_softmax(self.l2(e_feature), dim=1)

finl_model = Cls().to(device)
optimizerF = optim.Adam(finl_model.parameters(), lr=0.0008)
l_his = []
best_acc = 0
best_pea = 0

for i in range(100):
    for epoch in range(50):
        optimizerF.zero_grad()
        _, _, pred_labels = finl_model(Labeled_train)
        loss = F.nll_loss(pred_labels, Label)
        loss.backward()
        optimizerF.step()
        # print("训练分类损失：", loss)

    feature1, _, test_pre = finl_model(testdata)
    feature1 = pd.DataFrame(feature1.cpu().detach().numpy())
    test_pre1 = pd.DataFrame(test_pre.cpu().detach().numpy())
    # pea = feature1.corrwith(test_pre1, axis=0)
    # mean_pea = pea.mean()
    test_loss = F.nll_loss(test_pre, testlabel)
    pre = test_pre.data.max(1, keepdim=True)[1]
    correct = pre.eq(testlabel.data.view_as(pre)).cpu().sum() / len(testlabel)
    l_his.append(correct)
    if correct > best_acc:
        best_acc = correct
    # if mean_pea > best_pea:
    #     best_pea = mean_pea
    print("测试分类损失：", test_loss)
    print("测试分类精度：", correct)

print("最终分类精度：", best_acc)



