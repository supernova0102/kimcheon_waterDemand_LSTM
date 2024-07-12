import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from lstm_model import LSTMModel
import random

# 모델 파라미터 정의
input_dim = 26
hidden_dim = 64
layer_dim = 4
output_dim = 1
dropout = 0.4

# 학습된 모델 로드
model_path = "./model.pth"  # 모델 경로 확인에 주의
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout)
model.load_state_dict(torch.load(model_path))
model.eval()

# 평가 데이터셋으로 예측 수행
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 평가 데이터셋 로드
df = pd.read_csv("./data_ts_city.csv")  # 평가 데이터셋 경로 수정 필요
df = df.rename(columns={"datetime": "time", '구미 혁신도시배수지 유출유량 적산차': 'value'})
time = df['time']
value = df['value']
df['weekday'] = 0
weekday = df['weekday']

# 인풋데이터에서 요일 피쳐 도출
def isoweekday(t, wkday):
    for i in range(1, len(wkday)):
        d = datetime.date(int(t[i][:4]), int(t[i][5:7]), int(t[i][8:10]))
        wkday[i] = d.isoweekday()

isoweekday(time, weekday)  # 요일 데이터 계산 => 7: 일 1: 월 2: 화 3: 수 4: 목 5: 금 6: 토

# df.set_index - 데이터프레임에서 'time' 컬럼을 인덱스로 설정
df = df.set_index(['time'])
# pd.to_datetime() - pandas의 Timestamp 타입으로 변환
df.index = pd.to_datetime(df.index)
# 날짜 순서대로 정렬되어 있지 않은 경우, 날짜 순서대로 정렬
if not df.index.is_monotonic:
    df = df.sort_index()

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
# 이전 100개의 데이터 값을 이용하여 현재 값을 예측
input_dim = 24
df_generated = generate_time_lags(df, input_dim)

# 휴일이 영향을 주는지
import holidays
kr_holidays = holidays.KR()

# 해당 날짜가 한국의 공휴일이면 1 반환, 공휴일이 아니면 0 반환
def is_holiday(date):
    date = date.replace(hour=0)
    return 1 if (date in kr_holidays) else 0

# 결과를 새로운 열 'is_holiday'으로 추가
def add_holiday_col(df, holidays):
    return df.assign(is_holiday=df.index.to_series().apply(is_holiday))

df_features = add_holiday_col(df_generated, kr_holidays)

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

scaler = MinMaxScaler()

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'value', 0.4)

scaler = MinMaxScaler()

## 학습할 데이터셋과 평가할 데이터셋 분리
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

test = TensorDataset(test_features, test_targets)

test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

with torch.no_grad():
    predictions = []
    values = []

    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        yhat = model(x_test)
        predictions.append(yhat.cpu().numpy())
        values.append(y_test.cpu().numpy())

    predictions = np.concatenate(predictions)
    values = np.concatenate(values)

# 2차원 배열을 스케일링된 형태에서 원래의 값으로 변환
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
values = scaler.inverse_transform(values.reshape(-1, 1))

# DataFrame 생성
df_result = pd.DataFrame({'prediction': predictions.flatten(), 'value': values.flatten()})


# 연속된 24개의 데이터 선택
start_index = 0  # 시작 인덱스
start_index = random.randrange(0,len(df_result)-24) #무작위 선택
window_size = 24  # 윈도우 크기




# 선택한 데이터 슬라이스
selected_data = df_result.iloc[start_index : start_index + window_size]

# 값을 평면화하고 배율을 축소하여 오차 지표 계산
def calculate_metrics(df):
    return {
        'mae': mean_absolute_error(df['value'], df['prediction']),
        'rmse': mean_squared_error(df['value'], df['prediction']) ** 0.5,
        'r2': r2_score(df['value'], df['prediction'])
    }

result_metrics = calculate_metrics(df_result)

#print("Evaluation Metrics:")
#for metric, value in result_metrics.items():
#    print(f"{metric}: {value:.4f}")

# 기준선 예측 생성
from sklearn.linear_model import LinearRegression

def build_baseline_model(df, test_ratio, target_col):
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result["prediction"] = prediction
    result = result.sort_index()

    return result

df_baseline = build_baseline_model(df_result, 0.2, 'value')
baseline_metrics = calculate_metrics(df_baseline)

#print("\nBaseline Metrics:")
#for metric, value in baseline_metrics.items():
#    print(f"{metric}: {value:.4f}")

# 실제값과 예측값 추출
actual_values = selected_data['value']
predicted_values = selected_data['prediction']

#mae값 계산 출력
diff = np.abs(actual_values - predicted_values)
mae = np.mean(diff)

print("MAE : ", mae)


# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label='Actual')
plt.plot(predicted_values, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()