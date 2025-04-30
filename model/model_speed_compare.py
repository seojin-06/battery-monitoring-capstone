import time
import torch
import numpy as np
import pickle
from model import tasks
from model import dynamic_vae

# 모델 불러오기
model_full = torch.load('model.torch', weights_only=False, map_location='cpu')     # 비경량화 모델
model_full.eval()

model_light = torch.load('quantized_model.torch', weights_only=False, map_location='cpu')  # 경량화 모델
model_light.eval()

# Task 구성
columns = ["soc", "current", "min_temp", "max_single_volt", "min_single_volt", "volt", "max_temp"]
task = tasks.Task(columns=columns, task_name="batterybrandb")

# Normalizer 불러오기
with open('norm.pkl', 'rb') as f:
    normalizer = pickle.load(f)

# raw dataset 컬럼 순서 (timestamp 포함)
raw_columns = ["volt", "current", "soc", "max_single_volt", "min_single_volt", "max_temp", "min_temp", "timestamp"]

while True:
    sensor_data = {
        "volt": 3.7,                 # 팩 전체 전압
        "current": 0.0,              # 배터리 전류
        "soc": 78.0,                 # state of charge
        "max_single_volt": 4.2,      # 최고 셀 전압
        "min_single_volt": 4.1,      # 최저 셀 전압
        "max_temp": 37.0,            # 최고 셀 온도
        "min_temp": 36.0,            # 최저 셀 온도
        "timestamp": 0.0             # dummy timestamp
    }

    # 센서 데이터를 raw column 순서로 배열
    sensor_input = np.array([[sensor_data[col] for col in raw_columns]], dtype=np.float32)

    normalized_input = normalizer.norm_func(sensor_input)[0][:-1]  # (8,) → (7,)

    # encoder feature 순서로 재배열
    encoder_input = np.array([[normalized_input[raw_columns.index(feat)] for feat in columns]], dtype=np.float32)

    # tensor 변환
    input_tensor = torch.tensor(encoder_input).unsqueeze(1)

    # 경량화 모델 추론 시간 측정
    start_time = time.time()
    with torch.no_grad():
        _ = model_light(input_tensor,
                        encoder_filter=task.encoder_filter,
                        decoder_filter=task.decoder_filter,
                        seq_lengths=[1],
                        noise_scale=1.0)
    elapsed_light = time.time() - start_time
    print(f"경량화 모델 추론 시간: {elapsed_light:.6f}초")

    # 비경량화 모델 추론 시간 측정
    start_time = time.time()
    with torch.no_grad():
        _ = model_full(input_tensor,
                       encoder_filter=task.encoder_filter,
                       decoder_filter=task.decoder_filter,
                       seq_lengths=[1],
                       noise_scale=1.0)
    elapsed_full = time.time() - start_time
    print(f"비경량화 모델 추론 시간: {elapsed_full:.6f}초")

    # 다음 데이터 읽기 전 대기
    time.sleep(1)
