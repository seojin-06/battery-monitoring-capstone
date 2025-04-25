package com.battery.monitoring.repository;

import com.battery.monitoring.domain.MonitorLog;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface MonitorLogRepository extends JpaRepository<MonitorLog, Long> {
    List<MonitorLog> findTop100ByDeviceIdOrderByTimestampDesc(int deviceId);

    @Query("SELECT DISTINCT m.deviceId FROM MonitorLog m ORDER BY m.deviceId")
    List<Integer> findDistinctDeviceId();
}
