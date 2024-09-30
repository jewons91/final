import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import StepLR
import torch
torch.cuda.empty_cache()

code = '000660_2'
model_name = 'gcn_반도체10'

print(f'GPU available : {torch.cuda.is_available()}')
# 데이터 로드
df = pd.read_csv(f'C:\\big18\\final\with\\almost\\test_data\{code}체결강도Vector.csv')
print(df)
# 노드 특징 행렬 (Node feature matrix)
x = torch.tensor(df.iloc[:,:-1].values, dtype=torch.float)

# 엣지 리스트 (Edge list)
edge_index = []
num_rows = len(df)
# for i in range(1, num_rows):
#     edge_index.append([i, i -1 ])
#     edge_index.append([i -1, i ])
for i in range(1, num_rows):
    edge_index.append([i, i - 1])
    edge_index.append([i - 1, i])
    # edge_index.append([i, i - 3])
    # edge_index.append([i, i - 4])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 레이블
y = torch.tensor(df['18'].values, dtype=torch.long)

# 그래프 데이터 객체
# graph_data = Data(x=x, edge_index=edge_index, y=y)

# 데이터셋 분리
# train_indices, val_indices = train_test_split(range(num_rows), test_size=0.2, random_state=42)
# train_mask = torch.zeros(num_rows, dtype=torch.bool)
# val_mask = torch.zeros(num_rows, dtype=torch.bool)
# train_mask[train_indices] = 1
# val_mask[val_indices] = 1

train_data = Data(x=x, edge_index=edge_index, y=y)
# train_data.train_mask = train_mask
# train_data.val_mask = val_mask

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

import torch.optim as optim

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_data = train_data.to(device)

# 옵티마이저 정의
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)  #   optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # 학습률&정규화 조정 (model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

# 조기 종료 설정
early_stopping_patience = 100    # *10
early_stopping_counter = 0
best_acc = 0.0

# 정확도 계산 함수
def accuracy(pred, labels):
    _, pred_classes = pred.max(dim=1)
    correct = (pred_classes == labels).sum().item()
    return correct / len(labels)

# # 모델 훈련 및 평가
# model.train()
# for epoch in range(10000):
#     optimizer.zero_grad()
#     out = model(train_data)
#     loss = F.nll_loss(out, train_data.y)
#     loss.backward()
#     optimizer.step()
#     scheduler.step()

#     # 평가 모드로 전환하여 테스트 셋에서의 성능을 평가
#     model.eval()
#     with torch.no_grad():
#         val_out = model(train_data)
#         val_loss = F.nll_loss(val_out, train_data.y)
#         val_acc = accuracy(val_out, train_data.y)

#     # 모델을 다시 훈련 모드로 전환
#     model.train()

#     if epoch % 10 == 0:
#         train_acc = accuracy(out, train_data.y)
#         current_lr = scheduler.get_last_lr()[0]
#         print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}     Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_acc:.4f}    Lr : {current_lr:.6f}')
#         if epoch > 500:
#             if val_acc > best_acc:
#                 best_acc = val_acc
#                 torch.save(model.state_dict(), 'gcn.pth')
#                 early_stopping_counter = 0
#                 print(f'New best model saved with val accuracy: {best_acc:.4f}')
#             else:
#                 early_stopping_counter += 1
#     if early_stopping_counter >= early_stopping_patience:
#         print(f'Early stopping at epoch {epoch}')
#         break

# 학습 완료 후 최종 평가
model.load_state_dict(torch.load(f'C:\\big18\\final\with\\almost\gcn_model\{model_name}.pth'))
model.eval()
with torch.no_grad():
    final_out = model(train_data)
    final_loss = F.nll_loss(final_out, train_data.y)
    final_acc = accuracy(final_out, train_data.y)
    # val_out = model(train_data)
    val_pred = final_out.max(dim=1)[1].cpu().numpy()
    val_true = train_data.y.cpu().numpy()


conf_matrix = confusion_matrix(val_true, val_pred)
class_report = classification_report(val_true, val_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)
# print(f'종목명 : {name}')
print(f'Final Eval Loss: {final_loss.item():.4f}, Final Eval Accuracy: {final_acc:.4f}')

real_close_data = pd.read_csv(f'C:\\big18\\final\with\\almost\\test_data\{code}.csv',encoding='cp949')
# real_close_data = real_close_data[:32]
real_close_data = real_close_data['종가']

val_pred = val_pred.astype(float)
val_pred = np.insert(val_pred, 0, [np.nan] * 12)

df = pd.DataFrame({
    '종가': real_close_data,
    '예측 행동': val_pred
})
# 결과 데이터프레임 출력
print(df)
# df.to_excel('./result.xlsx')

# results_df.to_excel('C:/big18/dl-dev/dl-dev/project/변동률과 체결강도/000660_2급등주행동예측.xlsx')

# 백테스팅
# 초기 자본금
initial_capital = 200000
capital = initial_capital
position = 0  # 현재 보유 주식 수
buy_price = 0  # 매수 가격
sell_count = 0  # 총 매도 횟수
buy_count = 0  # 총 매수 횟수
sell_fee = 0.00195
buy_fee = 0.00015

fee = 0 # 수수료

# 결과를 저장할 리스트
results = []

# 수익률 계산
for i in range(len(df)):
    if df['예측 행동'][i] == 2 and position == 0:  # 매수 조건
        position = capital // df['종가'][i-2]  # 최대 매수 가능 수량
        buy_price = df['종가'][i-2]  # 매수 가격
        buy_count += 1  # 매수 횟수
        fee += (buy_price * position) * buy_fee  # 매수 수수료
        capital -= (position * buy_price) + ((buy_price * position) * buy_fee)  # 자본금 감소
        results.append({'행동': '매수', '가격': buy_price, '잔고': capital})

    elif df['예측 행동'][i] == 0 and position > 0:  # 매도 조건
        sell_price = df['종가'][i-2]  # 매도 가격
        capital += (position * sell_price) - ((sell_price * position) * sell_fee)  # 매도 후 자본금 증가
        fee += (sell_price * position) * sell_fee  # 매도 수수료
        sell_count += 1  # 매도 횟수
        
        # 수익률 계산
        profit_per_share = sell_price - buy_price
        profit_rate = profit_per_share-buy_price
        
        results.append({'행동': '매도', '가격': sell_price, '잔고': capital, '수익금': profit_per_share})
        position = 0  # 보유 주식 수 초기화

# 결과를 데이터프레임으로 변환
results_df = pd.DataFrame(results)
# results_df.to_excel('./하이닉스수익확인.xlsx')
# 최종 자산 가치 계산
final_value = capital + position * df['종가'].iloc[-1]
profit = final_value - initial_capital
profit_rate = (profit / initial_capital) * 100

print(f'초기 자본금: {initial_capital}원')
print(f'최종 자산 가치: {final_value:.2f}원')
print(f'총 수익: {profit:.2f}원')
print(f'수익률: {profit_rate:.2f}%')
print(f'매도 횟수: {sell_count}, 매수 횟수: {buy_count}')
print(f'총 수수료 : {fee:.2f}원')

# 결과 출력
results_df
