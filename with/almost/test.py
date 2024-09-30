import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import StepLR
from torch.nn import BatchNorm1d
import gc
# from sklearn.model_selection import train_test_split

gc.collect()
print(f'GPU available : {torch.cuda.is_available()}')

# 데이터 로드
df = pd.read_csv('C:\\big18\\final\with\\almost\\data\\000660전체데이터Vector.csv')
# df = df.iloc[:,2:]
# df = df.astype('float16')

print(df)
# 노드 특징 행렬 (Node feature matrix)
x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float)

# 엣지 리스트 (Edge list)
edge_index = []
num_rows = len(df)
for i in range(1, num_rows):
    edge_index.append([i, i - 1])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 레이블
y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

# 그래프 데이터 객체
# graph_data = Data(x=x, edge_index=edge_index, y=y)

# 데이터셋 분리
# train_indices, val_indices = train_test_split(range(num_rows), test_size=0.2, random_state=42)
# train_mask = torch.zeros(num_rows, dtype=torch.bool)
# val_mask = torch.zeros(num_rows, dtype=torch.bool)
# train_mask[train_indices] = 1
# val_mask[val_indices] = 1

# 데이터셋 분리
split_index = int(num_rows * 0.8)  # 80%를 훈련 데이터로 사용

train_mask = torch.zeros(num_rows, dtype=torch.bool)
val_mask = torch.zeros(num_rows, dtype=torch.bool)

train_indices = range(split_index)  # 0부터 split_index까지의 인덱스
val_indices = range(split_index, num_rows)  # split_index부터 num_rows까지의 인덱스


train_mask[train_indices] = 1
val_mask[val_indices] = 1

train_data = Data(x=x, edge_index=edge_index, y=y)
train_data.train_mask = train_mask
train_data.val_mask = val_mask

#      복잡 모델
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.bn1 = BatchNorm1d(16)
        self.conv2 = GCNConv(16, 32)
        self.bn2 = BatchNorm1d(32)
        self.conv3 = GCNConv(32, 64)
        self.bn3 = BatchNorm1d(64)
        self.conv4 = GCNConv(64, num_classes)
        # self.bn4 = BatchNorm1d(128)
        # self.conv5 = GCNConv(128, num_classes)
        # self.bn5 = BatchNorm1d(256)
        # self.conv6 = GCNConv(256, num_classes)
        # self.bn6 = BatchNorm1d(512)
        # self.conv7 = GCNConv(512, num_classes)
        # 1D Convolution 레이어 추가
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)
        # x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)
        # x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = self.conv4(x, edge_index)
        # x = self.bn4(x)
        # x = F.gelu(x)
        # x = self.dropout(x)
        
        # x = self.conv5(x, edge_index)
        # x = self.bn5(x)
        # x = F.gelu(x)
        # x = self.dropout(x)
        
        # x = self.conv6(x, edge_index)
        # x = self.bn6(x)
        # x = F.gelu(x)
        # x = self.dropout(x)
        
        # x = self.conv7(x, edge_index)

        return F.log_softmax(x, dim=1)

# 모델 초기화
model = GCN(num_node_features=x.size(1), num_classes=len(y.unique()))


# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_data = train_data.to(device)

# 옵티마이저 정의
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)  #   optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # 학습률&정규화 조정 (model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

# 조기 종료 설정
early_stopping_patience = 1000    # *10
early_stopping_counter = 0
best_acc = 0.0

# 정확도 계산 함수
def accuracy(pred, labels):
    _, pred_classes = pred.max(dim=1)
    correct = (pred_classes == labels).sum().item()
    return correct / len(labels)

# 모델 훈련 및 평가
model.train()
for epoch in range(100000):
    optimizer.zero_grad()
    out = model(train_data)
    loss = F.nll_loss(out[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()

    # 평가 모드로 전환하여 테스트 셋에서의 성능을 평가
    model.eval()
    with torch.no_grad():
        val_out = model(train_data)
        val_loss = F.nll_loss(val_out[train_data.val_mask], train_data.y[train_data.val_mask])
        val_acc = accuracy(val_out[train_data.val_mask], train_data.y[train_data.val_mask])

    # 모델을 다시 훈련 모드로 전환
    model.train()

    if epoch % 10 == 0:
        train_acc = accuracy(out[train_data.train_mask], train_data.y[train_data.train_mask])
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}     Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_acc:.4f}    Lr : {current_lr:.6f}')
        if epoch > 500:
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'C:\\big18\\final\with\\almost\gcn_model\gcn_new.pth')
                early_stopping_counter = 0
                print(f'New best model saved with val accuracy: {best_acc:.4f}')
            else:
                early_stopping_counter += 1
    if early_stopping_counter >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch}')
        break

# 학습 완료 후 최종 평가
model.load_state_dict(torch.load('C:\\big18\\final\with\\almost\gcn_model\gcn_new.pth'))
model.eval()
with torch.no_grad():
    final_out = model(train_data)
    final_loss = F.nll_loss(final_out[train_data.val_mask], train_data.y[train_data.val_mask])
    final_acc = accuracy(final_out[train_data.val_mask], train_data.y[train_data.val_mask])
    val_out = model(train_data)
    val_pred = val_out[train_data.val_mask].max(dim=1)[1].cpu().numpy()
    val_true = train_data.y[train_data.val_mask].cpu().numpy()

conf_matrix = confusion_matrix(val_true, val_pred)
class_report = classification_report(val_true, val_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

print(f'Final Eval Loss: {final_loss.item():.4f}, Final Eval Accuracy: {final_acc:.4f}')