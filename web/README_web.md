# 모니터링 웹사이트
## 1. 주요 기능
- 기기 ID 선택: 다수의 라즈베리파이 장치 중 모니터링할 기기를 선택
- 에러 수치 시각화: Chart.js를 활용하여 에러 수치 그래프를 실시간으로 표시
- 이상 여부 확인: predict 값을 기준으로 이상 상태 여부를 확인
- 최근 로그 테이블: 최신 10건의 로그 데이터를 테이블 형태로 확인
- 요약 정보 제공: 평균 에러, 임계치, 이상 탐지 횟수 표시
## 2. 기술 스택
- Frontend: HTML, JavaScript, Chart.js
- Backend: Spring Boot (Java)
- Database: AWS RDS (MariaDB)
- Server: AWS EC2 (Ubuntu)
## 3. 시퀀스
![시퀀스 drawio](https://github.com/user-attachments/assets/83fc004b-0a25-436c-89a7-00ff7850028f)
- 라즈베리파이가 이상 탐지 결과(deviceId, error, predict, threshold)를 EC2 서버로 POST 전송
- EC2에서 Spring Boot 서버가 데이터를 수신하고 RDS에 저장
- 웹사이트에서 기기 선택 시 해당 기기의 데이터를 서버 API를 통해 요청
- 수신된 데이터를 그래프로 시각화하고 테이블에 표시
