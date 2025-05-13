import time
import torch
import numpy as np
import pickle
from collections import deque
from model import tasks
from model import dynamic_vae
import raspSensor
import requests


class SequenceDataCollector:
    def __init__(self, max_seq_length=128, timestamp_increment=10):
        self.max_seq_length = max_seq_length
        self.timestamp_increment = timestamp_increment
        self.data_queue = deque(maxlen=max_seq_length)
        self.current_timestamp = 0
    
    def add_sensor_data(self, sensor_data):
        # 센서 데이터에 타임스탬프 할당
        sensor_data_copy = sensor_data.copy()
        sensor_data_copy["timestamp"] = self.current_timestamp
        self.data_queue.append(sensor_data_copy)
        self.current_timestamp += self.timestamp_increment
        
        # 타임스탬프가 최대값(1270)을 초과하면 다시 0으로 리셋
        # 기존 dataset timestamp: 0-1270 (간격 10)
        if self.current_timestamp > 1270:
            self.current_timestamp = 0
    
    def is_ready(self):
        return len(self.data_queue) == self.max_seq_length
    
    def get_sequence_data(self, raw_columns):
        if not self.is_ready():
            return None
        
        # raw_columns 순서로 센서 데이터 배열 생성
        sequence_data = []
        for data in self.data_queue:
            row = [data[col] for col in raw_columns]
            sequence_data.append(row)
        
        return np.array(sequence_data, dtype=np.float32)



model = torch.load('quantized_model.torch', weights_only=False, map_location='cpu')
model.eval()

columns = ["soc", "current", "min_temp", "max_single_volt", "min_single_volt", "volt", "max_temp"]
task = tasks.Task(columns=columns, task_name="batterybrandb")

with open('norm.pkl', 'rb') as f:
    normalizer = pickle.load(f)

threshold = 0.018541733038268546

# raw dataset 컬럼 순서 (timestamp 포함)
raw_columns = ["volt", "current", "soc", "max_single_volt", "min_single_volt", "max_temp", "min_temp", "timestamp"]

sequence_collector = SequenceDataCollector(max_seq_length=128, timestamp_increment=10)

mse = torch.nn.MSELoss(reduction='mean')

print("시퀀스 데이터 수집을 시작합니다. 128개의 데이터 포인트를 수집할 때까지 기다립니다...")

while True:
    try:
        sensor_data = raspSensor.getData()
        sequence_collector.add_sensor_data(sensor_data)
        
        raspSensor.printData(sensor_data)
        print(f"현재 수집된 데이터 포인트: {len(sequence_collector.data_queue)}/128")
        
        # 충분한 데이터가 모이면 이상 탐지 수행 - 약 20분
        if sequence_collector.is_ready():
            print("시퀀스 데이터가 준비되었습니다. 이상 탐지를 수행합니다...")
            
            sequence_data = sequence_collector.get_sequence_data(raw_columns)
            normalized_sequence = normalizer.norm_func(sequence_data)
            
            # encoder feature 순서로 재배열
            encoder_input = np.zeros((normalized_sequence.shape[0], len(columns)), dtype=np.float32)
            for i in range(normalized_sequence.shape[0]):  # 각 timestamp에 대해
                for j, col in enumerate(columns):  # 각 feature에 대해
                    col_idx = raw_columns.index(col)
                    encoder_input[i, j] = normalized_sequence[i, col_idx]
            
            input_tensor = torch.tensor(encoder_input).unsqueeze(0)  # (batch=1, seq=128, feature_dim)
            
            # 모델 추론
            with torch.no_grad():
                log_p, mean, log_v, z, mean_pred = model(input_tensor,
                                                        encoder_filter=task.encoder_filter,
                                                        decoder_filter=task.decoder_filter,
                                                        seq_lengths=[128],
                                                        noise_scale=1.0)
                
                target = task.target_filter(input_tensor)
                rec_error = float(mse(log_p, target))
       
            if rec_error > threshold:
                print(f"이상 발생 감지! Reconstruction Error: {rec_error:.6f}")
                predict = 1
            else:
                print(f"정상 상태. Reconstruction Error: {rec_error:.6f}")
                predict = 0
            
            # 웹서버에 데이터 POST
            postData = {
                "deviceId": "raspberrypi01",
                "predict": predict,
                "error": rec_error,
                "threshold": threshold
            }
            
            # res = requests.post("http://3.25.67.35:8082/api/monitoring/data", json=postData)
            # print(res)
        
        # 다음 데이터 읽기 전 잠시 대기 (10s)
        time.sleep(10)
    
    except Exception as e:
        print(f"오류 발생: {e}")
        time.sleep(1)
