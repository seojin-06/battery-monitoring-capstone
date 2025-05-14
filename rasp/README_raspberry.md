# 필요 모듈
- raspberry pi 5 (8gb ram)
- INA219
- ADS1115
- DS18B20
- I2C LCD modeul (4 lines disaply)
- 18650 battery * 2(with battery holder)
- 18650 battery charger module
- 5v 0.18A fan
- ETC(breadboard, jumper cables)

# 하드웨어 회로도
![image](https://github.com/user-attachments/assets/950061c3-3af0-4ccb-801c-c9dbdef986f8)


# 전체 흐름
![image](https://github.com/user-attachments/assets/ca57f9e3-009f-45e2-b124-6e7eb161cfe3)

- 각 배터리 셀에 대하여 각각 보호회로 내장 충전 모듈과 보호회로가 없는 충전 모듈을 연결하여 충전 유무에 따른 이상 탐지 여부도 수행할 수 있습니다.
  
**모델 전달 데이터 예시**
```python
sensor_data = {
        "volt": volt,                 # 팩 전체 전압
        "current": (-1)*curr,              # 배터리 전류
        "soc": soc,                 # state of charge
        "max_single_volt": max_volt,      # 최고 셀 전압
        "min_single_volt": min_volt,      # 최저 셀 전압
        "max_temp": max_temp,            # 최고 셀 온도
        "min_temp": min_temp,            # 최저 셀 온도
        "timestamp": 0.0             # dummy timestamp
    }
```
