import os
import os.path as osp
import copy
import argparse
import random

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, average_precision_score
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader, ImbalancedSampler
from torch_geometric import seed_everything

from sklearn.metrics import average_precision_score

# custom modules
from logger import setup_logger
from metapath import drop_metapath
from model import HeteroGNN

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="../data/icdm2022_session1.pt")
parser.add_argument("--labeled-class", type=str, default="item")
parser.add_argument(
    "--batch-size",
    type=int,
    default=1024,
    help="Mini-batch size. If -1, use full graph training.",
)
parser.add_argument(
    "--fanout", type=int, default=150, help="Fan-out of neighbor sampling."
)
parser.add_argument(
    "--n-layers", type=int, default=2, help="number of propagation rounds"
)
parser.add_argument("--h-dim", type=int, default=512, help="number of hidden units")
parser.add_argument("--in-dim", type=int, default=256, help="number of hidden units")
parser.add_argument("--early-stopping", type=int, default=100)
parser.add_argument('--full', action='store_true')
parser.add_argument('--lp', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument("--n-epoch", type=int, default=30)
parser.add_argument("--record-file", type=str, default="session1_record.txt")
parser.add_argument("--model-file", type=str, default="model_sess1.pth")
parser.add_argument("--resume-file", type=str, default="model_resume.pth")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--seed", type=int, default=2022)

args = parser.parse_args()

logger = setup_logger(output=args.record_file)

logger.info(args)

def pick_step(idx_train, y_train, adj_list, size):
    degree_train = [len(adj_list[node]) for node in idx_train]
    lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    smp_prob = np.array(degree_train) / lf_train
    return random.choices(idx_train, weights=smp_prob, k=size)



if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

device = torch.device(device)

# 加载异构图数据
hgraph = torch.load(args.dataset)
# 获取类别标签
labeled_class = args.labeled_class
# 检索并移除训练集和验证集的索引
train_idx = hgraph[labeled_class].pop("train_idx")
val_idx = hgraph[labeled_class].pop("val_idx")
# 如果'full'参数为真，则合并训练和验证索引
if args.full:
    train_idx = torch.cat([train_idx, val_idx])
# 记录节点特征的统计信息
logger.info("=" * 70)
logger.info("Node features statistics")
for name, value in hgraph.x_dict.items():
    logger.info(f"Name: {name}, feature shape: {value.size()}")

logger.info("=" * 70)

logger.info("Edges statistics")
for name, value in hgraph.edge_index_dict.items():
    logger.info(f"Relation: {name}, edge shape: {value.size()}")
logger.info("=" * 70)

hgraph_train = copy.copy(hgraph)
hgraph_test = copy.copy(hgraph)

if args.lp:
    logger.info("Add labels for label propagation...")

    y = hgraph_train[labeled_class].y.clone()
    y[y == -1] = 2  # mask unknown nodes

    hgraph_test[labeled_class].y_emb = y.clone()

    y[val_idx] = 2  # mask validation nodes
    hgraph_train[labeled_class].y_emb = y.clone()
    del y

logger.info("Initializing NeighborLoader...")

seed_everything(args.seed)

# Mini-Batch
sampler = ImbalancedSampler(hgraph_train[labeled_class].y, input_nodes=train_idx)

train_loader = NeighborLoader(hgraph_train, input_nodes=(labeled_class, train_idx),
                              num_neighbors=[args.fanout] * args.n_layers,
                              sampler=sampler, batch_size=args.batch_size)

val_loader = NeighborLoader(hgraph_train, input_nodes=(labeled_class, val_idx),
                            num_neighbors=[args.fanout] * args.n_layers,
                            shuffle=False, batch_size=args.batch_size)

logger.info("NeighborLoader Initialized.")

if args.resume:
    logger.info(f"Resume from {args.resume_file}")
    model = torch.load(f"{args.resume_file}").to(device)
else:
    model = HeteroGNN(metadata=hgraph.metadata(),
                      hidden_channels=args.h_dim, out_channels=2, n_layers=args.n_layers
                      ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

logger.info(model)
logger.info(optimizer)


def to_device(d, device):
    for k, v in d.items():
        d[k] = v.to(device)
    return d


metapaths_to_drop = [
    [('f', 'a'), ('a', 'e')],
    [('f', 'e'), ('e', 'a')],
    [('f', 'item'), ('item', 'b')],
    [('b', 'item'), ('item', 'f')],
]

def f1_score_binary(y_true, y_pred, threshold=0.5):
    # 将预测概率转换为二元预测（0或1）
    y_pred_binary = (y_pred > threshold).float()
    # 计算F1-score（需要确保y_true是一维的）
    return f1_score(y_true.cpu().numpy().flatten(), y_pred_binary.cpu().numpy().flatten())
def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in train_loader:
        optimizer.zero_grad()  #在每个批次开始时，清除之前批次的梯度，以便进行新一轮的梯度计算和更新。
        batch_size = batch[labeled_class].batch_size  # 获取当前批次中标签数据的批次大小。
        y = batch[labeled_class].y[:batch_size].to(device)  # 从批次中获取标签数据，并将其移动到指定的计算设备(device)上

        batch = drop_metapath(batch, metapaths_to_drop, r=[0.5, 0.5])  # 对批次中的元路径进行丢弃(drop)操作

        y_emb = getattr(batch[labeled_class], 'y_emb', None)  # 从批次中获取标签的嵌入表示(y_emb)，如果不存在则为None。
        if y_emb is not None:
            y_emb = y_emb.clone().to(device)  # 如果存在标签的嵌入表示，则克隆该表示并将其移动到指定的计算设备上。
            y_emb[:batch_size] = 2  # mask current batch nodes

        y_hat = model(to_device(batch.x_dict, device),  # 使用模型进行前向传播，得到模型的预测值y_hat。
                      to_device(batch.edge_index_dict, device), y_emb)[:batch_size]
        loss = F.cross_entropy(y_hat, y)  # 计算预测值y_hat与真实标签y之间的交叉熵损失。
        loss.backward()  # 反向传播，计算损失相对于模型参数的梯度。
        optimizer.step()  # 根据优化器的配置，更新模型参数，以减小损失。

        # 将模型的预测概率值（经过softmax处理后的第二列）添加到y_pred列表中，并将其从计算设备移动到CPU上。
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        # 将真实标签添加到y_true列表中，并将其从计算设备移动到CPU上。
        y_true.append(y.cpu())
        # 累加当前批次的损失
        total_loss += float(loss) * batch_size
        # 统计当前批次中预测正确的样本数量。
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())

        # 累加当前批次的样本数量。
        total_examples += batch_size
        # 更新进度条，表示当前批次已处理完毕。
        pbar.update(batch_size)
        # 关闭进度条，表示训练循环结束。
    pbar.close()

    y_true = torch.hstack(y_true).numpy()
    y_pred = torch.hstack(y_pred).numpy()
    # ap_score = average_precision_score(y_true, y_pred)

    y_pred_binary = (y_pred > 0.5).astype(int)
    # 计算 F1-macro
    f1_macro = f1_score(y_true, y_pred_binary, average='macro')

    # 计算 Gmean
    cm = confusion_matrix(y_true, y_pred.round())
    tp = cm[1, 1]  # 假设1是正类
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    sensitivity = tp / (tp + fn)  # 召回率或真正率（TPR）
    specificity = tn / (fp + tn)  # 真负率（TNR）或特异性
    if sensitivity > 0 and specificity > 0:
        gmean = np.sqrt(sensitivity * specificity)
    else:
        gmean = 0  # 避免除以零的错误

    # 计算 AUC
    auc = roc_auc_score(y_true, y_pred)

    # 计算 AP
    ap_score = average_precision_score(y_true, y_pred)

    return total_loss / total_examples, total_correct / total_examples, ap_score, f1_macro, gmean, auc


@torch.no_grad()
def val():
    model.eval()
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in val_loader:
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        y_emb = getattr(batch[labeled_class], 'y_emb', None)
        if y_emb is not None:
            y_emb = y_emb.to(device)

        y_hat = model(to_device(batch.x_dict, device),
                      to_device(batch.edge_index_dict, device), y_emb)[:batch_size]
        loss = F.cross_entropy(y_hat, y)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
    pbar.close()

    y_true = torch.hstack(y_true).numpy()
    y_pred = torch.hstack(y_pred).numpy()

    y_pred_binary = (y_pred > 0.5).astype(int)
    # 计算 F1-macro
    f1_macro = f1_score(y_true, y_pred_binary, average='micro')

    # 计算 Gmean
    cm = confusion_matrix(y_true, y_pred.round())
    tp = cm[1, 1]  # 假设1是正类
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    sensitivity = tp / (tp + fn)  # 召回率或真正率（TPR）
    specificity = tn / (fp + tn)  # 真负率（TNR）或特异性
    if sensitivity > 0 and specificity > 0:
        gmean = np.sqrt(sensitivity * specificity)
    else:
        gmean = 0  # 避免除以零的错误

    # 计算 AUC
    auc = roc_auc_score(y_true, y_pred)

    ap_score = average_precision_score(y_true, y_pred)

    return total_loss / total_examples, total_correct / total_examples, ap_score,f1_macro, gmean, auc


val_ap_list = []
best_result = 0

logger.info("Start training")

for epoch in range(1, args.n_epoch + 1):
    train_loss, train_acc, train_ap, f1_macro, gmean, auc = train(epoch)
    logger.info(
        f"Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AP_Score: {train_ap:.4f},f1_macro: {f1_macro:.4f},gmean: {gmean:.4f},auc: {auc:.4f}"
    )
    val_loss, val_acc, val_ap ,f1_macro, gmean, auc= val()
    val_ap_list.append(float(val_ap))

    if best_result < val_ap:
        best_result = val_ap
        torch.save(model, f"{args.model_file}")

    logger.info(
        f"Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}, Best AP: {best_result:.4f},f1_macro: {f1_macro:.4f},gmean: {gmean:.4f},auc: {auc:.4f}"
    )

    if epoch >= args.early_stopping:
        ave_val_ap = np.average(val_ap_list)
        if val_ap <= ave_val_ap:
            logger.info(f"Early Stopping at {epoch}")
            break
