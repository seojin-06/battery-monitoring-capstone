import os
import RPi_I2C_driver #GPL License
import time
#ADS1115
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

#ina219 전압 측정 모듈
import board
from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219 #BSD License

#LCD 모듈 초기화
mylcd = RPi_I2C_driver.lcd()

# ds18b20 온도데이터값을 아래 temp_sensor 경로에 저장하기 위한 명령어
os.system('modprobe w1-gpio')
os.system('modprobe w1-therm')

# 라즈베리파이가 센서데이터를 받는 경로를 설정(경로는 라즈베리마다 다름))
temp_sensor='/sys/bus/w1/devices/28-0000003860f9/w1_slave'

'''
온도 관련 코드
'''
# 온도 데이터 파일의 내용을 읽어오는 함수
def temp_raw():
    f = open(temp_sensor,'r')
    lines = f.readlines()
    f.close()
    return lines
 
# 읽어온 파일의 구문을 분석해 온도부분만 반환하는 함수
def read_temp():
    lines = temp_raw()
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = temp_raw()
 
 
    temp_output = lines[1].find('t=')
 
    if temp_output != -1:
        temp_string = lines[1].strip()[temp_output+2:]
        temp_c = float(temp_string) / 1000.0
        temp_f = temp_c * 9.0/5.0 + 32.0
        return temp_c, temp_f
 
'''
전압 측정 코드
'''
# 배터리 전압으로 SOC 초기값 설정 
# TODO : 정확도를 높이기 위한 방법 고민 필요 (배터리 방전 곡선 데이터 활용 등...(데이터 시트를 아직 못 찾음))
def voltage_to_soc(voltage):
    if voltage >= 4.20:
        return 100
    elif voltage >= 4.10:
        return 90
    elif voltage >= 4.00:
        return 80
    elif voltage >= 3.90:
        return 70
    elif voltage >= 3.80:
        return 60
    elif voltage >= 3.70:
        return 50
    elif voltage >= 3.60:
        return 30
    elif voltage >= 3.50:
        return 20
    elif voltage >= 3.40:
        return 10
    else:
        return 0

i2c_bus = board.I2C()  # uses board.SCL and board.SDA 0x40
ina219 = INA219(i2c_bus)

# 정확도 및 범위 설정
ina219.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina219.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina219.bus_voltage_range = BusVoltageRange.RANGE_16V

def getBatteryStatus():
    bus_voltage = ina219.bus_voltage  # voltage on V- (load side)
    shunt_voltage = ina219.shunt_voltage  # voltage between V+ and V- across the shunt
    current = ina219.current  # current in mA
    power = ina219.power  # power in watts

    internal_resistance = 0.1  # 내부저항 값 (Ω), 실제 측정값으로 정밀화 필요. 0.1Ω은 추정 값.

    # OCV 근사값 계산
    # 실제 측정값
    bus_voltage = ina219.bus_voltage
    shunt_voltage = ina219.shunt_voltage
    measured_voltage = bus_voltage + shunt_voltage
    current_A = ina219.current / 1000.0

    # OCV 근사값 계산
    ocv_voltage = measured_voltage + (current_A * internal_resistance)
    
    # SOC 업데이트
    soc = voltage_to_soc(ocv_voltage)
    '''
    END
    '''
    # INA219 measure bus voltage on the load side. So PSU voltage = bus_voltage + shunt_voltage
    # print("Voltage (VIN+) : {:6.3f}   V".format(bus_voltage + shunt_voltage))
    # print("Voltage (VIN-) : {:6.3f}   V".format(bus_voltage))
    # print("Shunt Voltage  : {:8.5f} V".format(shunt_voltage))
    # print("Shunt Current  : {:7.4f}  A".format(current / 1000))
    #print("Power Calc.    : {:8.5f} W".format(bus_voltage * (current / 1000)))
    # print("Power Register : {:6.3f}   W".format(power))
    #print("ocv_voltage    : {:6.3f}   V".format(ocv_voltage))
    #print("SOC           : {:3d} %".format(soc))
    # print("")

    # Check internal calculations haven't overflowed (doesn't detect ADC overflows)
    if ina219.overflow:
        print("Internal Math Overflow Detected!")
        print("")

    voltage = ocv_voltage
    current = current/1000
    return voltage, current, soc

# ADS1115를 이용한 1번 셀 전압 측정
ads = ADS.ADS1115(i2c_bus)
def getSingleCellVoltage():
    channel = AnalogIn(ads, ADS.P0)
    return channel.voltage


def getData():
    global min_volt, max_volt, min_temp, max_temp
    # 센서 데이터 읽어오기
    volt, curr, soc = getBatteryStatus()
    firstCellVol = getSingleCellVoltage()
    secondCellVol = volt-firstCellVol
    soc = voltage_to_soc(volt/2)
    
    min_volt = min(firstCellVol, secondCellVol)
    max_volt = max(firstCellVol, secondCellVol)
    temp_c, _ = read_temp()
    # 온도 센서가 하나라 우선 하나의 온도만 측정
    min_temp = temp_c
    max_temp = temp_c
    sensor_data = {
        "volt": volt/2,                 # 팩 전체 전압
        "current": curr,              # 배터리 전류
        "soc": soc,                 # state of charge
        "max_single_volt": max_volt,      # 최고 셀 전압
        "min_single_volt": min_volt,      # 최저 셀 전압
        "max_temp": max_temp,            # 최고 셀 온도
        "min_temp": min_temp,            # 최저 셀 온도
        "timestamp": 0.0             # dummy timestamp
    }
    return sensor_data

def printData(data):
    print(f"voltage: {data['volt']:.3f} V")
    print(f"current: {data['current']:.3f} A")
    print(f"soc: {data['soc']:3d} %")
    print(f"max_single_volt: {data['max_single_volt']:.3f} V")
    print(f"min_single_volt: {data['min_single_volt']:.3f} V")
    print(f"max_temp: {data['max_temp']:.3f} C")
    print(f"min_temp: {data['min_temp']:.3f} C")
    print(f"timestamp: {data['timestamp']:.3f} s")
    print("")

'''
측정 값을 실시간으로 LCD에 출력하는 코드 (모든 코드 통합)
TODO: 웹서버에 데이터 POST 필요
'''
if __name__ == "__main__":
    while True:
        #print(read_temp())
        data = getData()
        printData(data)
        vol, curr, soc = getBatteryStatus()
        
        now = time.localtime()
        current_time = time.strftime("%H:%M:%S", now)

        mylcd.lcd_display_string(f"Time: {current_time} Soc: {soc:3d}%", 1)
        mylcd.lcd_display_string(f"Temp: {read_temp()[0]:.3f} C", 2)
        mylcd.lcd_display_string(f"vol: {vol:.3f}v ", 3)
        mylcd.lcd_display_string(f"curr: {curr:.3f}", 4)
        time.sleep(3)
        mylcd.lcd_clear()

