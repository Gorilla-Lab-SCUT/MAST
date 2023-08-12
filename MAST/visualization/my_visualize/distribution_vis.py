import matplotlib.pyplot as plt
import torch
from sklearn import manifold, datasets
import numpy as np 


file = 'dis_final_z.pth'
dis = torch.load(file)
X_t = dis['ft'].detach().numpy()
label_t = dis['lbt'].detach().numpy()
X_s = dis['fs'].detach().numpy()
label_s = dis['lbs'].detach().numpy()


## cal feature norm
# bins = np.unique(label_s)
# print(bins)
# for i in bins:
#     bin = np.where(label_t == i)
#     bin_len = np.linalg.norm(X_t[bin],axis=-1).mean()
#     print(f'{i}: {bin_len}')


## choose samples
num = 500
choose = np.array(range(len(X_s)))
choose = np.random.choice(choose, size=num)
X_s = X_s[choose]
label_s = label_s[choose]
X_t = X_t[choose]
label_t = label_t[choose]

X = np.concatenate((X_s, X_t))
label = np.concatenate((label_s, label_t))


## generate color
color = np.ones(2 * num)
color[:num] = 0
color = color.tolist()
cm = {0:'green', 1:'red'}
for i in range(len(color)):
    color[i] = cm[color[i]]
## green is source, red is target 


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(X)  # 转换后的输出

plt.scatter(Y[:, 0], Y[:, 1], c='white')
for i in range(Y.shape[0]):
    plt.text(Y[i, 0], Y[i, 1], str(label[i]),fontdict={'size': 7}, c=color[i])

plt.title(f"t-SNE\n(green: source, red: target )\n{file}")
plt.show()




# zfeat.append(featz.detach().cpu())
# Rfeat.append(featr.detach().cpu())
# zl.append(lblz)
# rl.append(lblr)
# if n == 500:
#     zfeat = torch.stack(zfeat).reshape(-1,featz.size(-1))
#     Rfeat = torch.stack(Rfeat).reshape(-1,featz.size(-1))
#     zl = torch.stack(zl).reshape(-1)
#     rl = torch.stack(rl).reshape(-1)
#     dis = dict(
#         zf=zfeat,
#         rf=Rfeat,
#         zl=zl,
#         rl=rl
#     )
#     torch.save(dis,'dis.pth')
#     exit()