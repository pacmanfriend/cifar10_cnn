use crate::api::types::{SystemCpuInfo, SystemGpuInfo};
use axum::Json;
use std::process::Command;
use sysinfo::System;

pub async fn cpu_info() -> Json<SystemCpuInfo> {
    let info = tokio::task::spawn_blocking(|| {
        let mut system = System::new_all();
        system.refresh_cpu();
        let cpu = system.global_cpu_info();
        SystemCpuInfo {
            name: cpu.brand().to_string(),
            physical_cores: system.physical_core_count(),
            logical_cores: system.cpus().len(),
            usage_percent: cpu.cpu_usage(),
        }
    })
    .await
    .expect("cpu_info task panicked");
    Json(info)
}

pub async fn gpu_info() -> Json<SystemGpuInfo> {
    let info = tokio::task::spawn_blocking(|| {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output();

        match output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let first = stdout.lines().next().unwrap_or_default();
                let parts = first.split(',').map(str::trim).collect::<Vec<_>>();
                SystemGpuInfo {
                    available: parts.len() >= 3,
                    name: parts.first().map(|value| (*value).to_string()),
                    vram_mb: parts.get(1).and_then(|value| value.parse().ok()),
                    driver_version: parts.get(2).map(|value| (*value).to_string()),
                    error: None,
                }
            }
            Ok(output) => SystemGpuInfo {
                available: false,
                name: None,
                vram_mb: None,
                driver_version: None,
                error: Some(String::from_utf8_lossy(&output.stderr).trim().to_string()),
            },
            Err(err) => SystemGpuInfo {
                available: false,
                name: None,
                vram_mb: None,
                driver_version: None,
                error: Some(err.to_string()),
            },
        }
    })
    .await
    .expect("gpu_info task panicked");
    Json(info)
}
