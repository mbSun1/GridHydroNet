# dlcore

`dlcore` 使用流域平均降水、气温和潜在蒸散预测唯一出口站的日流量。它不读取
DEM、地形属性、河网、子流域、非出口站流量、HBV 状态或历史流量输入。

## 模型

当前包含 15 个模型：DLinear、iTransformer、LSTM、Informer、Reformer、
Transformer、ETSformer、Pyraformer、Crossformer、CARDformer、BiLSTM、GRU、
BiGRU、RNN 和 TCN。

TCN 使用左侧填充的残差因果膨胀卷积。默认包含 7 个残差块，膨胀率为
`1, 2, 4, 8, 16, 32, 64`，每个块含两层卷积核为 3 的一维卷积，时间感受野
为 509 天，可覆盖默认 365 天输入窗口。模型读取最后一个时间位置的因果特征，
输出目标日出口径流。

前 10 个模型依据论文 [From RNNs to Transformers: benchmarking deep learning
architectures for hydrologic prediction](https://doi.org/10.5194/hess-29-6811-2025)
随附工程的回归批处理模型范围进行日尺度单流域适配；参考工程按 CC BY-NC 4.0
许可发布。这里删除了 CAMELS、多流域静态属性和原训练器耦合，并统一接口为：

```python
prediction = model(forcing_window, time_features=None)
```

输入形状为 `[batch, 365, 3]`。窗口包含目标日及此前364天的降水、气温和
潜在蒸散，不读取历史径流；模型估算窗口最后一天的出口径流，输出形状为
`[batch, 1]`。这是已知当天气象条件下的同日径流模拟，不表述为提前预报。

## 数据

直接使用：

```text
data_repository/basin_<name>/forcing/basin_mean/*.nc
data_repository/basin_<name>/forcing/basin_mean/manifest.json
data_repository/basin_<name>/discharge/processed/*.nc
```

归一化统计仅由训练期计算，验证期和测试期目标日期互不重叠。出口流量缺测不会插值，
对应目标窗口会被排除。

## 运行

命令行只保留 `prepare` 和 `all` 两种模式。`all` 会在同一次运行中完成训练、验证和
测试，不提供另行加载检查点的独立测试模式。

命令行仅暴露流域与模型选择、随机种子、训练/验证/测试时期、输入长度、
训练轮次、批量大小、学习率、weight decay、dropout、早停、DataLoader 进程数、
运行设备和输出行为。模型维度、层数、注意力头数、patch、Reformer 和梯度裁剪均为对应函数中的内部默认参数；实际采用
的模型结构会写入 `model_metadata.json`，内部训练设置会写入 `run_config.yaml`。

校验数据：

```powershell
python -m dlcore.experiments.run --mode prepare --basin_name ybbs
```

训练、验证和测试一个模型：

```powershell
python -m dlcore.experiments.run --mode all --basin_name ybbs --model transformer
```

使用 TCN：

```powershell
python -m dlcore.experiments.run --mode all --basin_name pascagoula --model tcn
```

例如，使用包含目标日在内的365天气象估算目标日径流：

```powershell
python -m dlcore.experiments.run --mode all --basin_name ybbs --model transformer --lookback 365
```

结果写入 `outputs/dl/<basin>_<model>_<timestamp>/`。根目录中的
`period_metrics_comparison.csv` 按训练、验证和测试时期汇总最终性能指标。
默认不截断负流量，并在指标中记录负值比例；只有显式传入
`--clip_negative_discharge` 才会截断。
