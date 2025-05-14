package com.battery.monitoring.controller;

import com.battery.monitoring.domain.MonitorLog;
import com.battery.monitoring.repository.MonitorLogRepository;
import lombok.AllArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequiredArgsConstructor
@Slf4j
public class MonitorController {
    private final MonitorLogRepository repos;

    @GetMapping("/data")
    public List<MonitorLog> getMonitorDataByDevice(String deviceId) {
        return repos.findTop100ByDeviceIdOrderByTimestampDesc(deviceId);
    }

    @GetMapping("/devices")
    public List<String> getAllDeviceIds() {
        List<String> ids = repos.findDistinctDeviceId();
        log.info("조회된 deviceIds = " + ids);
        return ids;
    }
}
