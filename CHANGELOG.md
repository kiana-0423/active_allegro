# Changelog

All notable changes to this project will be documented in this file.

---

## [v0.2] - 2026-04-21

### 项目简介

`GMD_active_learning` 是连接 `GMD`（分子动力学核心）与 `GMD_se3gnn`（机器学习势训练）的工程化主动学习控制层。项目不侵入 MD 核心或训练核心，通过 adapter 模式实现模块解耦，使主动学习策略可以独立演化与替换。

**核心特色：**

- **在线可靠性监控**：集成 LJ 投影监控器、集成偏差监控、物理性检查等多种 OOD 预警手段
- **LJ 投影预警**：将 MLIP 原子力投影到元素对 Lennard-Jones 参数空间，通过拟合残差、参数物理性与历史跳变量触发主动学习，仅作为 OOD 检测触发器，不替代 MLIP
- **风险聚合**：多监控器输出加权聚合为统一风险分，通过 YAML 配置权重与阈值
- **候选管理**：候选构型队列化管理，支持 pair distance histogram 去重（可扩展至 SOAP/ACSF）
- **DFT 标注**：生成 VASP / CP2K 输入文件与作业脚本
- **重训练编排**：通过 adapter 调用 `GMD_se3gnn` 完成训练、导出与模型注册
- **全配置驱动**：所有阈值、目录和模块开关均通过 YAML 配置文件控制，支持 dry-run 模式

**安装：**

```bash
pip install -e ".[dev]"
```

**快速开始：**

```bash
gmd-al monitor-example
python examples/minimal_lj_monitor_example.py
python examples/minimal_active_learning_loop.py
```

**配置文件：**

| 文件 | 用途 |
|------|------|
| `configs/active_learning.yaml` | 总流程、目录、dry-run 开关 |
| `configs/monitor.yaml` | 监控器开关、阈值、LJ 参数、风险权重 |
| `configs/dft_labeling.yaml` | DFT 标注任务模板与作业脚本 |
| `configs/retraining.yaml` | `GMD_se3gnn` 调用方式与导出命令 |

**运行测试：**

```bash
pytest
```

---

## [Unreleased]
