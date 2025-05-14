package com.battery.monitoring.controller;

import com.battery.monitoring.domain.MonitorLog;
import com.battery.monitoring.dto.MonitorDTO;
import com.battery.monitoring.repository.MonitorLogRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;

import java.time.LocalDateTime;

@Controller
@RequiredArgsConstructor
@RequestMapping("/api/monitoring")
public class DataController {
    private final MonitorLogRepository monitorLogRepository;

    @PostMapping("/data")
    public ResponseEntity<String> receiveData(@RequestBody MonitorDTO monitorDTO) {
        MonitorLog monitorLog = new MonitorLog();
        monitorLog.setDeviceId(monitorDTO.getDeviceId());
        monitorLog.setPredict(monitorDTO.getPredict());
        monitorLog.setError(monitorDTO.getError());
        monitorLog.setThreshold(monitorDTO.getThreshold());
        monitorLog.setTimestamp(LocalDateTime.now());

        monitorLogRepository.save(monitorLog);
        return ResponseEntity.ok("데이터 저장 완료");
    }
}
