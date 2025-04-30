import time
import torch
import numpy as np
import pickle
from model import tasks       # model/tasks.py
from model import dynamic_vae # model/dynamic_vae.py
import raspSensor
import requests
# 모델 불러오기
model = torch.load('quantized_model.torch', weights_only=False, map_location='cpu')
model.eval()

# Task 구성
columns = ["soc", "current", "min_temp", "max_single_volt", "min_single_volt", "volt", "max_temp"]
task = tasks.Task(columns=columns, task_name="batterybrandb")

# Normalizer 불러오기
with open('norm.pkl', 'rb') as f:
    normalizer = pickle.load(f)

# Threshold 설정 (추후 수정)
threshold = 0.0074031753465533

# raw dataset 컬럼 순서 (timestamp 포함)
raw_columns = ["volt", "current", "soc", "max_single_volt", "min_single_volt", "max_temp", "min_temp", "timestamp"]

mse = torch.nn.MSELoss(reduction='mean')

while True:
   #센서 데이터 읽어오기
    # sensor_data = {
    #     "volt": 3.7,                 # 팩 전체 전압
    #     "current": 0.0,              # 배터리 전류
    #     "soc": 78.0,                 # state of charge
    #     "max_single_volt": 4.2,      # 최고 셀 전압
    #     "min_single_volt": 4.1,      # 최저 셀 전압
    #     "max_temp": 37.0,            # 최고 셀 온도
    #     "min_temp": 36.0,            # 최저 셀 온도
    #     "timestamp": 0.0             # dummy timestamp
    # }
    sensor_data = raspSensor.getData()

    # 센서 데이터를 raw column 순서로 배열
    sensor_input = np.array([[sensor_data[col] for col in raw_columns]], dtype=np.float32)

    # 정규화 및 timestamp 제거
    normalized_input = normalizer.norm_func(sensor_input)[0]  # (8,) -> 1차원
    normalized_input = normalized_input[:-1]  # timestamp 제거

    # encoder feature 순서로 재배열
    encoder_input = np.array([[normalized_input[raw_columns.index(feat)] for feat in columns]], dtype=np.float32)

    # tensor 변환
    input_tensor = torch.tensor(encoder_input).unsqueeze(1)  # (batch=1, seq=1, feature_dim)

    # 모델 추론
    with torch.no_grad():
        log_p, mean, log_v, z, mean_pred = model(input_tensor,
                                                 encoder_filter=task.encoder_filter,
                                                 decoder_filter=task.decoder_filter,
                                                 seq_lengths=[1],
                                                 noise_scale=1.0)

        target = task.target_filter(input_tensor)
        rec_error = float(mse(log_p, target))

    print(target)
    raspSensor.printData(sensor_data)
    
    # 이상 여부 판단
    if rec_error > threshold:
        print(f"이상 발생 감지! Reconstruction Error: {rec_error:.6f}")
        predict = 1
    else:
        print(f"정상 상태. Reconstruction Error: {rec_error:.6f}")
        predict = 0

    postData = {
        "deviceId": "raspberrypi01",
        "predict": predict,
        "error": rec_error,
        "threshold": threshold
    }
    res = requests.post("http://3.25.67.35:8082/api/monitoring/data", json=postData)
    print(res)



    # 다음 데이터 읽기 전 잠시 대기
    time.sleep(2)
