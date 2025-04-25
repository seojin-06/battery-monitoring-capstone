package com.battery.monitoring.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter

public class MonitorDTO {
    private int deviceId;
    private int predict;
    private float error;
    private float threshold;
}
