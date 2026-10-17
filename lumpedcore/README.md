# Lumped HBV92

`lumpedcore` 是与 `disthybridcore` 并列的集总式日尺度 HBV92 模型实现。它复用
`data_repository` 中的流域数据，但只读取以下信息：

- `forcing/grid/` 中的逐日网格降水、气温和潜在蒸散；
- `hydro_grid/basin_fraction.nc`，仅用于计算流域面积权重和流域面积；
- `discharge/` 中唯一标记为出口站（`is_outlet=True`）的实测流量。

DEM、河网、子流域拓扑等分布式模型输入不会进入 HBV92 计算。

## 数据目录

```text
data_repository/
└── basin_<name>/
    ├── forcing/
    │   ├── grid/
    │   │   ├── basin_<name>_era5_t_*.nc
    │   │   ├── basin_<name>_gleam_pet_*.nc
    │   │   ├── basin_<name>_mswep_prec_*.nc
    │   │   └── normalization/
    │   └── basin_mean/
    │       ├── basin_<name>_daily_basin_mean_*.nc
    │       └── manifest.json
    ├── hydro_grid/
    └── discharge/
```

`basin_mean/` 中的文件由 `prepare` 阶段确定性生成。源文件的路径、大小和修改
时间会写入 `manifest.json`；签名一致时直接复用缓存，源数据发生变化时重新聚合。

## 运行

在仓库根目录执行：

命令行只保留 `prepare` 和 `all` 两种模式；`all` 会在同一次运行中完成率定和验证，
不提供另行加载参数文件的独立验证模式。

```powershell
python -m lumpedcore.experiments.run --mode prepare --basin_name ybbs
```

完成预处理、SCE-UA 率定和验证：

```powershell
python -m lumpedcore.experiments.run `
  --mode all `
  --basin_name ybbs `
  --calibration_start 2008-01-01 `
  --calibration_end 2014-12-31 `
  --validation_start 2015-01-01 `
  --validation_end 2017-12-31 `
  --spinup_days 365 `
  --routing gamma `
  --repetitions 5000 `
  --ngs 5 `
  --seed 42
```

## 模型与输出

- HBV92 使用 12 个参数：`beta, fc, k0, k1, k2, lp, perc, uzl, tt, cfmax,
  cfr, cwh`；
- 默认启用 15 日 Gamma 单位线，可通过 `--routing none` 关闭；
- 率定目标为出口站流量的精确 `1 - NSE`，不做中间舍入；
- 模型从预热期连续运行到率定或验证期结束，中间不重置状态；
- 流量换算为 `Q = runoff(mm/day) * area(km²) / 86.4`。

每次运行写入 `outputs/lumped/<basin>_hbv92_<timestamp>/`，其中包含配置、最佳
参数、完整率定历史、率定/验证逐日结果、指标、水量平衡诊断和可选图件。

