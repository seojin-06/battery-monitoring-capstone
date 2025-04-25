package com.battery.monitoring.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "monitor_log")
@Getter
@Setter
public class MonitorLog {

    @Id @GeneratedValue
    private Long id;

    @Column(name = "device_id")
    private int deviceId;

    private int predict;

    private float error;

    private float threshold;

    private LocalDateTime timestamp;
}
