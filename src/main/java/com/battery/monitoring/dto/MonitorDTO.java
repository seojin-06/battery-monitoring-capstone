package com.battery.monitoring.dto;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Data
public class MonitorDTO {
    private String deviceId;
    private int predict;
    private float error;
    private float threshold;
}
