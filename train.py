import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from lstm_model import LSTMModel


# 대화형 그래프 생성

# plotly - 대화형 그래프 생성
# df - Pandas DataFrame (시계열 데이터)
# title - 그래프 제목
import plotly.graph_objs as go
from plotly.offline import iplot

def plot_dataset(df, title):
    data = []
    # Scatter - 시계열 데이터를 그래프로
    value = go.Scatter(
        x=df.index,
        y=df.value,
        # mode - 점(line) 또는 선(lines)의 모양 지정
        mode="lines",
        # name - 데이터에 대한 이름을 지정
        name="values",
        marker=dict(),
        text=df.index,
        # line - 그래프의 선에 대한 스타일 설정, 그래프의 선 색상과 불투명도 설정
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )
    
    # fig - 데이터와 레이아웃을 결합 -> 그래프 객제 생성
    fig = dict(data=data, layout=layout)
    # iplot - 그래프 출력
    iplot(fig)

# 데이터 전처리

limit_max = 1500.0 # 데이터 필터링 상한선
limit_min = 0.0 # 데이터 필터링 하한선

# 데이터가 적정한지 여부를 판별하는 함수
def is_clear_data(value, index): 
    if((float(value[index])<=limit_min) or (pd.isna(value[index])) or (float(value[index])>=limit_max)):
        return False
    else:
        return True

# 간단한 평균값으로 데이터 재작성하는 함수
def data_rewriting1(front, value, index): 
    if(is_clear_data(value, index)):
        return index
    else:
        tail_index = data_rewriting1(front, value, index+1)
        value[index] = (float(front) + float(value[tail_index]))/2
        return tail_index

# 데이터 전처리 유형 1: 전후 평균값으로 대체
def data_clearing_1(v): 
    front = v[1] 
    i = 2
    while(i<len(v)):
        if(is_clear_data(v, i)):
            front = v[i]
            i += 1
        else:
            tail_index = data_rewriting1(front, v, i)
            print(time[i],"~",time[tail_index],"평준화")
            i = tail_index

# 데이터 전처리 유형 2: 다른 년도의 값으로 대체 (권장)
def data_clearing_2(time, value):
    i = 2
    while(i<len(value)):
        if(is_clear_data(value, i)):
            date = time[i][5:10]
            i = i+1
            continue
        else:
            year = time[i][:4]
            if(year == "2017"):
                value[i] = float(value[i+(24*365)])+float(value[i+2*(24*365)])+float(value[i+3*(24*365)])/ 3
            elif(year == "2018"):
                value[i] = float(value[i-(24*365)])+float(value[i+1*(24*365)])+float(value[i+2*(24*365)])/ 3
            elif(year == "2019"):
                value[i] = float(value[i-2*(24*365)])+float(value[i-(24*365)])+float(value[i+(24*365)])/ 3
            elif(year == "2020"):
                value[i] = float(value[i-3*(24*365)])+float(value[i-2*(24*365)])+float(value[i-(24*365)])/ 3
            print(time[i],"수정")
            i = i+1

# 부적합한 데이터를 탐색하고 결과를 출력하는 함수
def print_check_Csv(time, value): 
    count = 1
    for i in range(1,len(value)):
        if(float(value[i]) <= limit_min):
            print(str(time[i]),"의 데이터에서 ",count,"번째 오류 데이터 (기준치이하) 탐색됨!")
            count = count+1
        if(pd.isna(value[i])):
            print(str(time[i]), "의 데이터에서 ",count,"번째 오류 데이터 (공란) 탐색됨!")
            count = count+1
        if(float(value[i]) > limit_max):
            print(str(time[i]), "의 데이터에서 ",count,"번재 오류 데어터 (기준치이상) 탐색됨!")
            count = count+1
    if(count == 1):
        print(i,"개 데이터 검사결과 문제 없음!")

def isoweekday(t, wkday): #요일 피쳐 초기화
    for i in range(0, len(wkday)):
        d = datetime.date(int(t[i][:4]), int(t[i][5:7]), int(t[i][8:10]))
        wkday[i] = d.isoweekday()

df = pd.read_csv('./data_tr_city.csv') #트레이닝 데이터 읽기
df = df.rename(columns = {"datetime": "time", '구미 혁신도시배수지 유출유량 적산차': 'value'})

time = df['time']
value = df['value']
df['weekday'] = 0
weekday = df['weekday']

day = datetime.date(int(time[1][:4]), int(time[1][5:7]), int(time[1][8:10]))

print_check_Csv(time, value)
data_clearing_2(time, value) # 전처리
isoweekday(time, weekday) #요일 데이터 계산 => 7: 일 1: 월 2: 화 3: 수 4: 목 5: 금 6: 토

# 데이터 전처리 확인
print(df)

# df.set_index - 데이터프레임에서 'time' 컬럼을 인덱스로 설정
df = df.set_index(['time'])

# pd.to_datetime() - pandas의 Timestamp 타입으로 변환
df.index = pd.to_datetime(df.index)

# 날짜 순서대로 정렬되어 있지 않은 경우, 날짜 순서대로 정렬
if not df.index.is_monotonic:
    df = df.sort_index()
    
plot_dataset(df, title='Prediction of flow rate difference by reservoir to identify demand for water supply by region')

# generate_time_lags - 시간 지연(lag) 값 생성 (시계열 데이터에서 입력값으로 사용)
# n_lags - 지연 값의 수 
# df의 value 열을 n_lags 만큼 지연(shift) -> 새로운 열 생성

def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n

# input_dim - 입력 차원으로 사용할 지연 값
# 이전 24개의 데이터 값을 이용하여 현재 값을 예측

input_dim = 24

df_generated = generate_time_lags(df, input_dim)

# 확인
print(df_generated)

# 휴일이 영향을 주는지
import holidays

kr_holidays = holidays.KR()

# 해당 날짜가 한국의 공휴일이면 1 반환, 공휴일이 아니면 0 반환
def is_holiday(date):
    date = date.replace(hour = 0)
    return 1 if (date in kr_holidays) else 0

# 결과를 새로운 열 'is_holiday'으로 추가
def add_holiday_col(df, holidays):
    return df.assign(is_holiday = df.index.to_series().apply(is_holiday))


df_features = add_holiday_col(df_generated, kr_holidays)

# 확인
print(df_features)

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'value', 0.4)

scaler = MinMaxScaler()

X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)

batch_size = 64

train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
val_features = torch.Tensor(X_val_arr)
val_targets = torch.Tensor(y_val_arr)
test_features = torch.Tensor(X_test_arr)
test_targets = torch.Tensor(y_test_arr)

train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

device = torch.device('GeForce MX250' if torch.cuda.is_available() else 'cpu')

print(f'{device} is available')

# 모델 생성 및 최적화 객체 초기화

# 입력 데이터의 열 수 
input_dim = X_train.shape[1] #24 + 2
# 출력 데이터의 열 수
output_dim = 1
# LSTM 레이어의 은닉 상태 크기 (노드 수)
hidden_dim = 64
# LSTM 레이어의 층 수
layer_dim = 4
# 미니배치 크기
batch_size = 64
# Dropout을 적용할 확률
dropout = 0.4
# 전체 데이터셋을 통해 진행할 에포크 수
n_epochs = 100
# 학습률
learning_rate = 1e-3
# 가중치 감쇠 (L2 정규화) 계수
weight_decay = 1e-6

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout)

loss_fn = nn.L1Loss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 학습 루프
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
loss_fn.to(device)

train_losses = []
val_losses = []
epoch_losses = []

for epoch in range(1, n_epochs + 1):
    batch_losses = []
    model.train()
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        yhat = model(x_batch)
        loss = loss_fn(yhat, y_batch)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)
    
    with torch.no_grad():
        val_batch_losses = []
        model.eval()
        
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            yhat = model(x_val)
            val_loss = loss_fn(yhat, y_val).item()
            val_batch_losses.append(val_loss)
            
        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)
    
    epoch_losses.append((train_loss, val_loss))
    
    if (epoch <= 10) or (epoch % 50 == 0):
        print(f"[{epoch}/{n_epochs}] Training loss: {train_loss:.4f}\t Validation loss: {val_loss:.4f}")

# 학습된 모델 저장
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
#model_path = os.path.join(models_dir, f"model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth")
torch.save(model.state_dict(), './model.pth')

# 손실 함수 그래프 출력
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()