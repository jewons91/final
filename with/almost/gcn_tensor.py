import numpy as np
import pandas as pd
from spektral.data import Graph
from spektral.layers import GCNConv
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from scipy.sparse import coo_matrix
import tensorflow as tf

code = '000660'

# 데이터 읽기
df = pd.read_csv(f'C:\\big18\\final\\with\\almost\\data\\{code}전체데이터Vector.csv')

# 노드 특징 행렬 (Node feature matrix)
x = df.iloc[:, :-1].values
# 레이블
y = df.iloc[:, -1].values

# 데이터 샘플링 (예: 1000개 데이터만 사용)
sample_size = 1000
num_nodes = min(len(x), sample_size)
x = x[:num_nodes]
y = y[:num_nodes]

# 그래프 구성 (인접 행렬)
edges = np.array([[i, i + 1] for i in range(num_nodes - 1)])  # 시간 순서로 연결
row_indices = []
col_indices = []
for edge in edges:
    row_indices.append(edge[0])
    col_indices.append(edge[1])

# 희소 행렬 생성
a = coo_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(num_nodes, num_nodes)).toarray()

# 그래프 데이터 생성
graph = Graph(x=x, a=a)

# GCN 모델 정의
def build_model(num_features):
    gcn_input = Input(shape=(num_nodes, num_features))  # 수정된 입력 형태
    a_input = Input(shape=(num_nodes, num_nodes), sparse=True)  # 희소 인접 행렬 입력

    # GCNConv에 mask를 None으로 설정
    x = GCNConv(32, activation='gelu')([gcn_input, a_input], training=True)  # training=True로 설정
    x = Dropout(0.5)(x)
    x = Dense(16, activation='gelu')(x)
    output = Dense(3, activation='softmax')(x)  # 3개의 클래스 (상승, 보합, 하락)

    model = Model(inputs=[gcn_input, a_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 모델 구축
model = build_model(x.shape[1])

# 라벨을 2D 형태로 변환
labels_2d = y.reshape(-1, 1)

# 모델 훈련
model.fit([graph.x, graph.a], labels_2d, epochs=50, batch_size=1)

# 예측
predictions = model.predict([graph.x, graph.a])
predicted_classes = np.argmax(predictions, axis=1)

print("예측 결과:", predicted_classes)
