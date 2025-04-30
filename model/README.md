## 1. 모델 경량화
- `dynamic_quantization.ipynb` 파일 실행
     - `model.torch`를 dynamic quantization 처리
     - 결과로 `quantized_model.torch` 파일 생성
- 기존 모델: `model.torch`
- 양자화된 모델: `quantized_model.torch`

## 2. Raspberry Pi
아래 파일들을 Raspberry Pi로 복사
- `real_time_inference.py`: 실시간 이상 탐지 코드
- `model_speed_compare.py`: 경량화 vs 비경량화 모델 추론 시간 비교 코드
- `model/dynamic_vae.py`: VAE 모델 정의 (폴더 구조 유지)
- `model/tasks.py` (폴더 구조 유지)
- `utils.py`
- `norm.pkl`: 데이터 정규화 scaler 파일
- `model.torch`
- `quantized_model.torch`

패키지 설치
- pytorch 설치 (라즈베리파이용)
- 추가로 필요한 Python 패키지: torch, numpy

추론 시간만 비교할 경우, run

```bash
python model_speed_compare.py
```

실시간 이상 탐지, run

```bash
python real_time_inference.py
```

## 3. Dataset 구조
- volt: 팩 전체 전압
- current: 배터리 전류
- soc: state or charge
- max_single_volt: 최고 셀 전압
- min_single_volt: 최저 셀 전압
- max_temp: 최고 셀 온도
- min_temp: 최저 셀 온도

## 4. Code Reference
```bash
https://github.com/962086838/Battery_fault_detection_NC_github
