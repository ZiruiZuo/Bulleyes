# Cardiac Bullseye

将心脏 MR 转换为短轴堆栈，并生成患者特异性的 AHA-17 供血区牛眼图、冠脉投影和冠脉距离牛眼图。

项目支持三种用法：

1. 只生成短轴；
2. 使用已有短轴数据生成牛眼图；
3. 一次完成“原始 MR → 短轴 → 牛眼图”。

## 目录结构

```text
Bulleyes/
├── src/cardiac_bullseye/
│   ├── short_axis/       # 短轴规划、重采样和 geometry.json
│   ├── territory/        # 供血区处理和 AHA-17 映射
│   ├── coronary/         # 冠脉中心线提取与投影
│   ├── distance/         # 冠脉距离体积和距离牛眼图
│   └── workflows/        # 独立流程及完整流程
├── tests/                # 自动化测试
├── config.example.toml   # 完整配置示例
├── pyproject.toml        # 安装信息和命令行入口
└── README.md
```

## 环境要求

- Python 3.10 或更高版本；
- Windows、Linux 或 macOS；
- 能够显示 Matplotlib/PyQt5 图形窗口的桌面环境。

短轴规划和 AIP/IIP 标记是交互操作，不适合在无图形界面的远程终端中运行。

## 安装

在 PowerShell 中进入项目目录并执行：

```powershell
Set-Location E:\ZZR\Code\Tools\Bulleyes
python -m pip install -e .
```

安装后会提供三个命令：

```text
cardiac-short-axis   只生成短轴
cardiac-bullseye     只生成牛眼图
cardiac-pipeline     运行完整流程
```

可以用以下命令确认安装是否成功：

```powershell
cardiac-short-axis --help
cardiac-bullseye --help
cardiac-pipeline --help
```

## 准备配置文件

复制示例配置：

```powershell
Copy-Item .\config.example.toml .\case.toml
```

然后编辑 `case.toml`。所有相对路径均以 `case.toml` 所在目录为基准。Windows 路径建议使用 TOML 单引号和正斜杠，例如：

```toml
[case]
id = 'patient_043'
output_directory = 'outputs'

[short_axis]
source_mr = 'inputs/patient_043_mr.nii.gz'
source_mask = 'inputs/patient_043_territory.nii.gz'

[bullseye]
coronary_mask = 'inputs/patient_043_coronary.nii.gz'
world_myocardium_mask = 'inputs/patient_043_myocardium.nii.gz'
```

建议始终填写非空的 `case.id`，它会成为所有输出文件的前缀。完整参数及注释见 [`config.example.toml`](config.example.toml)。

### 三种模式所需输入

| 模式 | 必需配置 |
| --- | --- |
| 只生成短轴 | `short_axis.source_mr`；需要同步生成 mask 时再填写 `short_axis.source_mask` |
| 只生成牛眼图 | `bullseye.short_axis_mr`、`short_axis_mask`、`geometry_json`、`coronary_mask`、`world_myocardium_mask` |
| 完整流程 | `short_axis.source_mr`、`source_mask`，以及 `bullseye.coronary_mask`、`world_myocardium_mask` |

`cardiac-pipeline` 会直接使用本次刚生成的短轴文件，因此不会读取配置中的 `bullseye.short_axis_mr`、`short_axis_mask` 和 `geometry_json`。

## 用法一：只生成短轴

```powershell
cardiac-short-axis --config .\case.toml
```

交互窗口分为三步：

1. 在轴位图上依次选择 Apex 和二尖瓣中心；
2. 在重格式化视图中再次确认 Apex 和二尖瓣中心；
3. 拖动 Apex/Base 边界线确定覆盖范围，然后点击 `Save`。

主要输出：

```text
<case.id>_short_axis.nii.gz
<case.id>_short_axis_mask.nii.gz   # 配置 source_mask 时生成
<case.id>_geometry.json
```

## 用法二：只生成牛眼图

先在 `[bullseye]` 中填写已有短轴 MR、短轴供血区 mask 和同次规划生成的 `geometry.json`，然后运行：

```powershell
cardiac-bullseye --config .\case.toml
```

程序会逐层要求选择 AIP 和 IIP。左键选择点，右键撤销；`Accept` 接受当前层，`Redo` 重选，`Skip` 跳过当前层。

主要输出：

```text
<case.id>_bullseye.png
<case.id>_distance_bullseye.png
<case.id>_distance_world.nii.gz
<case.id>_distance_short_axis.nii.gz
```

## 用法三：运行完整流程

```powershell
cardiac-pipeline --config .\case.toml
```

该命令先执行短轴规划，再将新生成的短轴 MR、mask 和几何信息直接传入牛眼图流程。最终输出包含上述 3 个短轴文件和 4 个牛眼图/距离文件。

## Python 调用

三个命令均有对应的 Python API：

```python
from cardiac_bullseye.config import load_config
from cardiac_bullseye.workflows import (
    generate_bullseye,
    generate_short_axis,
    run_complete_pipeline,
)

short_axis_config = load_config("case.toml", "short-axis")
short_axis_outputs = generate_short_axis(short_axis_config)

bullseye_config = load_config("case.toml", "bullseye")
bullseye_outputs = generate_bullseye(bullseye_config)

pipeline_config = load_config("case.toml", "pipeline")
pipeline_outputs = run_complete_pipeline(pipeline_config)
```

## 数据约定

- 所有 NIfTI 数据必须带有正确的 affine，并处于一致的世界坐标系。
- `territory_label_map` 将原始供血区标签映射为 `LAD=1`、`LCX=2`、`RCA=3`。
- `geometry.json`、短轴 MR 和短轴 mask 必须来自同一次、同一病例的短轴规划。
- 如果冠脉 mask 已经是一体素宽的中心线，将 `coronary_mask_is_centerline` 设置为 `true`；否则程序会先进行三维骨架化。
- 若不希望生成结果后弹出图窗，将 `show_final_figure` 设置为 `false`。

## 常见问题

### 找不到 `cardiac-*` 命令

确认当前 Python 环境正确，然后在项目目录重新执行：

```powershell
python -m pip install -e .
```

### 配置路径找不到

相对路径是相对于 `case.toml`，不是相对于当前 PowerShell 目录。也可以直接使用绝对路径。

### 图形窗口无法打开

请在本机桌面会话运行，并确认 PyQt5 已安装。远程无界面服务器通常不能完成交互标记。

## 运行测试

```powershell
Set-Location E:\ZZR\Code\Tools\Bulleyes
$env:PYTHONPATH = "$PWD/src"
python -m pytest tests -v
```
