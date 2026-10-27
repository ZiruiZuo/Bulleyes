# `cardiac-lge-bullseye` 使用教程

本教程用于指导第一次接触本项目的同学，从短轴 LGE 图像和心肌 mask 生成：

- 原始 LGE 强度牛眼图；
- 缺血性心肌梗死 scar mask；
- AHA 1–16 节段 scar 比例表；
- scar 比例热力牛眼图。

示例命令：

```powershell
cardiac-lge-bullseye --config .\toml\case_032602_lge.toml
```

> 注意：仓库中的实际配置文件名是 `case_032602_lge.toml`，不要在下划线前额外加入反斜杠。

## 1. 方法概览

程序按以下顺序工作：

1. 读取短轴 LGE 和心肌 mask，并检查二者是否位于完全相同的物理网格；
2. 裁掉心肌 mask 最前端和最后端的连续空白层；
3. 由用户逐层标记 AIP 和 IIP，确定 AHA 节段的角度方向；
4. 由用户在每个保留层使用画刷标记正常远端心肌 ROI；
5. 在每层正常心肌 ROI 内计算均值和样本标准差；
6. 以该层的 `mean + N × sample std` 作为 scar 阈值，默认 `N = 5`；
7. 将心肌体素划分至 AHA 1–16 节段，计算各节段 scar 体素比例；
8. 保存数值结果、scar mask 和牛眼图。

本程序采用逐层阈值：第 1 层使用第 1 层手绘 ROI 的统计量，第 2 层使用第 2 层 ROI 的统计量，以此类推。不同层之间不共用正常心肌均值和标准差。

对第 `i` 层：

```text
threshold_i = mean_i + scar_sd_multiplier × sample_std_i

scar_i = myocardium_i AND finite(LGE_i) AND (LGE_i > threshold_i)
```

其中样本标准差使用 `ddof = 1`。阈值比较是严格大于 `>`，等于阈值的体素不计为 scar。

## 2. 使用范围与重要说明

该流程针对缺血性心肌梗死的 LGE scar 半自动定量。SCMR 2020 后处理共识将正常远端心肌 ROI 加 5 SD 作为梗死半自动阈值的一个起始方法，同时强调定量前仍需进行视觉评估，并检查伪影、心肌抑制不佳和微血管阻塞等情况。

该工具目前不提供阈值分割后的逐体素手工修正，因此必须认真检查正常心肌 ROI 和最终 scar mask。如果病例不存在可信的正常远端心肌，例如疑似弥漫性病变、多支血管广泛受累或图像存在明显强度不均，`n-SD` 结果可能不可靠。

本工具适合科研分析和方法验证，不能替代有资质医师的临床诊断。

## 3. 所需数据

每个病例至少需要两个三维文件。

### 3.1 短轴 LGE 图像

要求：

- 三维单分量标量图像；
- 推荐格式为 `.nii` 或 `.nii.gz`；
- 必须已经是心脏短轴层面；
- 数组层序必须为 Apex → Base；
- 应保留原始 LGE 信号强度，本程序不做强度归一化。

### 3.2 左心室心肌 mask

要求：

- 与 LGE 图像逐体素对齐；
- 推荐格式为 `.nii.gz`、`.nrrd` 或 `.seg.nrrd`；
- 所有非零值都会被视为心肌；
- 应覆盖完整左心室心肌，包括正常心肌和 scar，而不是只标记正常心肌；
- 不应包含左室血池、右室心肌、心包脂肪或明显的心外组织。

程序严格检查 LGE 与 mask 的以下信息：

| 项目 | 要求 |
| --- | --- |
| Size | 完全一致 |
| Spacing | 一致，允许极小数值误差 |
| Origin | 一致，允许极小数值误差 |
| Direction | 一致，允许极小数值误差 |

如果二者不同，程序会直接报错，不会自动重采样 mask。应先在 3D Slicer 或其他医学图像软件中把 mask 重采样到 LGE 网格，并再次目视检查轮廓是否贴合心肌。

### 3.3 数据质控建议

运行前至少检查：

- LGE 与 mask 的层数和层序一致；
- 第一个有效心肌层位于心尖侧，最后一个有效层位于基底侧；
- 心内膜和心外膜边界合理；
- mask 没有明显漏标、越界或错层；
- LGE 中正常心肌得到适当抑制；
- 没有把血池、心外脂肪或伪影误标为心肌。

程序允许 mask 在心肌堆栈前后存在空白层，这些首尾空白层会自动裁掉。中间空白层不会用于 ROI 和 scar 计算，但仍建议先检查其是否属于标注错误。

## 4. 创建 Python 环境并安装

以下示例适用于 Windows PowerShell 和 Anaconda/Miniconda。推荐使用 Python 3.10 或 3.11。

### 4.1 创建独立环境

```powershell
conda create -n cardiac-bullseye python=3.10 -y
conda activate cardiac-bullseye
```

确认当前 Python 来自新环境：

```powershell
python --version
where.exe python
```

### 4.2 安装项目及依赖

进入 `Bulleyes` 项目根目录：

```powershell
Set-Location E:\ZZR\Code\Tools\Bulleyes
python -m pip install --upgrade pip
python -m pip install -e .
```

`pip install -e .` 会根据 `pyproject.toml` 自动安装主要依赖：

- NumPy；
- SciPy；
- Matplotlib；
- NiBabel；
- scikit-image；
- PyQt5；
- SimpleITK；
- Python 3.10 使用的 TOML 兼容包 `tomli`。

`-e` 表示可编辑安装。项目代码更新后通常不需要重复安装；如果更换了 Python 环境，则需要在新环境中重新执行安装命令。

### 4.3 验证安装

```powershell
cardiac-lge-bullseye --help
Get-Command cardiac-lge-bullseye
```

也可以检查关键依赖：

```powershell
python -c "import numpy, scipy, matplotlib, nibabel, skimage, SimpleITK, PyQt5; print('Dependencies OK')"
```

如果提示找不到 `cardiac-lge-bullseye`，先确认已经激活正确的 Conda 环境，再重新执行：

```powershell
python -m pip install -e E:\ZZR\Code\Tools\Bulleyes
```

## 5. 准备 TOML 配置文件

当前 `032602` 病例配置位于：

```text
E:\ZZR\Code\Tools\Bulleyes\toml\case_032602_lge.toml
```

内容结构如下：

```toml
[case]
id = '032602'
output_directory = 'E:/ZZR/Code/Tools/Bulleyes/outputs/032602_lge'

[lge_bullseye]
lge = 'E:/ZZR/Data/HospitalData/ZhongNan/032602_test/lge_sax.nii.gz'
mask = 'E:/ZZR/Data/HospitalData/ZhongNan/032602_test/lge_mask_032602.seg.nrrd'

angle_count = 360
radial_count = 300
angular_smoothing_sigma_deg = 1.5
radial_smoothing_sigma = 1.5
six_segment_warp = true

scar_sd_multiplier = 5.0
normal_roi_min_voxels = 20
```

### 5.1 配置项解释

| 配置项 | 含义 |
| --- | --- |
| `case.id` | 病例编号，也是所有输出文件的前缀 |
| `case.output_directory` | 输出目录，不存在时自动创建 |
| `lge_bullseye.lge` | 短轴 LGE 图像路径 |
| `lge_bullseye.mask` | 同网格左室心肌 mask 路径 |
| `angle_count` | 连续强度牛眼图的角向采样数 |
| `radial_count` | 连续强度牛眼图的径向采样数 |
| `angular_smoothing_sigma_deg` | 强度牛眼图的角向平滑参数，单位为度 |
| `radial_smoothing_sigma` | 强度牛眼图的径向平滑参数 |
| `six_segment_warp` | 是否依据 AIP/IIP 对六节段角度进行归一化 |
| `scar_sd_multiplier` | scar 阈值中的 SD 倍数，默认 5 |
| `normal_roi_min_voxels` | 每层正常心肌 ROI 的最少有效体素数，默认 20 |

平滑参数只用于连续 LGE 强度牛眼图。scar mask、scar 体素计数和各 AHA 节段 scar 比例均直接基于原始短轴体素计算，不对二值 scar 进行平滑或插值。

### 5.2 为新病例创建配置

复制现有配置后修改病例编号、输入路径和输出目录：

```powershell
Copy-Item .\toml\case_032602_lge.toml .\toml\case_NEW_lge.toml
notepad .\toml\case_NEW_lge.toml
```

Windows 路径推荐使用 TOML 单引号和正斜杠：

```toml
lge = 'D:/data/case_NEW/lge_sax.nii.gz'
```

相对路径以 TOML 配置文件所在目录为基准，而不是以 PowerShell 当前目录为基准。为了减少教学时的路径错误，初次使用推荐填写绝对路径。

## 6. 运行程序

激活环境并进入项目根目录：

```powershell
conda activate cardiac-bullseye
Set-Location E:\ZZR\Code\Tools\Bulleyes
```

运行 `032602` 示例：

```powershell
cardiac-lge-bullseye --config .\toml\case_032602_lge.toml
```

程序会在终端依次显示 `1/7` 至 `7/7` 的处理阶段。运行过程中不要关闭 PowerShell 窗口。

## 7. 第一步交互：逐层标记 AIP 和 IIP

### 7.1 标记对象

- AIP：右心室与左心室交界处的前侧插入点，显示为红色；
- IIP：右心室与左心室交界处的下侧插入点，显示为青色。

每层必须按 AIP → IIP 的顺序点击。绿色 `+` 是程序根据心肌 mask 估计的左室中心，仅用于辅助观察，不是需要点击的点。

### 7.2 鼠标和按钮

| 操作 | 功能 |
| --- | --- |
| 左键点击 | 依次选择 AIP、IIP |
| 右键点击 | 撤销最后一个点 |
| 鼠标滚轮 | 以指针所在位置为中心缩放 |
| 中键拖动 | 平移图像 |
| `Fit` | 恢复完整视野 |
| `Redo` | 清空当前层的两个点并重新选择 |
| `Accept` | 接受当前层，进入下一层 |
| `Skip` | 当前层不手工标点，后续由相邻有效层插值 |
| `Delete` | 删除基底端不应参与分析的层 |
| `Save && Quit` | 保存当前标注状态并退出本标注窗口 |

### 7.3 `Skip` 与 `Delete` 的区别

`Skip` 用于心肌仍应参与分析、但当前层难以可靠识别 AIP/IIP 的情况。程序会根据其他有效层插值该层的 AIP/IIP；该层仍然需要手绘正常心肌 ROI，并参与 scar 计算。

`Delete` 用于基底端不应纳入左室节段分析的截面。所有删除层必须构成连续的基底端尾部，例如最后 2 层或最后 3 层。不能删除中间单独一层，也不能删除全部层，否则后续会报错。

建议尽量直接标记清晰可见的插入点，只在确实无法判断时使用 `Skip`。

### 7.4 默认重新标记；可选复用 AIP/IIP

默认 `reuse_saved_points = false`。每次执行 `cardiac-lge-bullseye` 时，程序都会忽略已有的 `<case.id>_lge_landmarks.json`，从空白状态重新逐层标记。

如果只是更换 label/mask，而 LGE 层数和层内尺寸没有变化，可以在 `[lge_bullseye]` 中设置：

```toml
reuse_saved_points = true
```

程序将验证并直接读取同一输出目录中的 `<case.id>_lge_landmarks.json`，完全跳过 AIP/IIP 标注窗口。`Skip` 与 `Delete` 状态也会复用；正常心肌 ROI 不会复用，仍需逐层重新涂抹。

当前操作仍会实时写入同名 JSON，供本次后续步骤使用。重新运行命令时，新标注会覆盖旧结果。

`Save && Quit` 不是“下次继续”的功能。正式分析时应完成全部所需层的标记；如果中途退出后重新运行，仍会从头开始。

## 8. 第二步交互：逐层涂抹正常心肌 ROI

AIP/IIP 阶段结束后，程序会打开 `Normal myocardium ROI annotation` 窗口。每个未删除且包含心肌的层都需要单独选择正常远端心肌 ROI。

### 8.1 正常心肌 ROI 的选择原则

优先选择：

- 视觉上无高信号强化的远端心肌；
- 与疑似梗死供血区相对远离的心肌；
- 信号相对均匀、没有明显伪影的区域；
- 位于心肌 mask 内部、与心内膜和心外膜边界保持适当距离的区域；
- 面积不小于配置的 `normal_roi_min_voxels`。

应避免：

- 可疑 LGE 高信号或其边缘；
- 梗死区域内的低信号微血管阻塞区；
- 左室血池和乳头肌；
- 心外膜脂肪和心外组织；
- 心内膜、心外膜附近明显的部分容积体素；
- 呼吸、运动、卷褶、鬼影及线圈强度不均区域；
- 只挑选最暗的少数体素。这样可能压低均值和标准差，造成 scar 高估。

ROI 应代表该层正常远端心肌，而不是单纯追求最低信号。若整层没有可信的正常心肌参考区，不应随意选择；应记录该病例的局限性，并由有经验的影像医师判断是否适合使用该方法。

### 8.2 画刷操作

| 操作 | 功能 |
| --- | --- |
| 左键拖动 | 涂抹 ROI |
| 右键拖动 | 擦除 ROI |
| `Brush radius` | 调整画刷半径，范围 1–20 像素 |
| 鼠标滚轮 | 以指针所在位置为中心缩放 |
| 中键拖动 | 平移图像 |
| `Fit` | 恢复完整视野 |
| `Undo` | 撤销上一笔涂抹或擦除 |
| `Clear` | 清空当前层 ROI |
| `Accept` | 接受当前层 ROI，进入下一层 |
| `Save && Quit` | 保存当前状态并退出 ROI 窗口 |

蓝色半透明区域表示当前 ROI。ROI 会自动与心肌 mask 和有限 LGE 体素求交，因此画到 mask 外的部分不会计入统计。

当有效体素数达到最低要求后，窗口标题会显示：

```text
voxels=体素数, mean=均值, std=样本标准差, 5-SD=阈值
```

红色半透明区域是按照当前 ROI 预览得到的候选 scar。接受前应检查：

- 红色区域是否主要位于预期的缺血性强化范围；
- 是否把血池边缘或心外伪影误判为 scar；
- scar 范围是否因 ROI 混入高信号而异常缩小；
- scar 范围是否因只选择极暗像素而异常扩大。

如果预览明显不合理，使用 `Clear` 或右键擦除后重新选择，不要直接接受。

### 8.3 ROI 不会跨运行复用

每次运行都会忽略已有的 `<case.id>_lge_normal_rois.json`，所有有效层均从空白画布重新涂抹。当前运行会保存新 ROI 和最终统计信息，但下次运行不会加载它们继续编辑。

## 9. AHA 节段和 scar 比例

程序按照保留层的 Apex → Base 位置划分长轴区域：

- 心尖部：AHA 13–16；
- 中部：AHA 7–12；
- 基底部：AHA 1–6。

AIP/IIP 用于确定节段的角度方向。每个节段的 scar 比例直接按原始体素计数：

```text
scar_percent = scar_voxels / myocardial_voxels × 100%
```

第 17 节段不参与当前 scar 计算，scar 牛眼图中心显示 `N/A`。

scar 热力图颜色固定为：

- 0%：白色；
- 比例升高：逐渐过渡为浅黄、橙色和红橙色；
- 色标范围：0–100%。

## 10. 输出文件

以病例 `032602` 为例，输出目录中会生成：

| 文件 | 内容 |
| --- | --- |
| `032602_lge_landmarks.json` | 本次逐层 AIP/IIP、Skip 和 Delete 状态 |
| `032602_lge_normal_rois.json` | 本次逐层正常心肌 ROI，以及最终 mean、std、threshold |
| `032602_lge_scar_mask.nii.gz` | 二值 scar mask，保持原 LGE 的完整网格和物理信息 |
| `032602_lge_scar_segments.csv` | AHA 1–16 的心肌体素数、scar 体素数和 scar 百分比 |
| `032602_lge_scar_bullseye.png` | AHA 1–16 scar 比例热力牛眼图 |
| `032602_lge_intensity_bullseye.npz` | 连续 LGE 强度牛眼图的数值数组 |
| `032602_lge_intensity_bullseye.png` | 连续原始 LGE 强度牛眼图 |

NPZ 中主要包含：

- `intensity`：径向 × 角向的连续 LGE 强度；
- `radial_positions`：径向位置；
- `canonical_angles`：标准化角度；
- `apex_intensity`：心尖中心强度。

同一病例再次运行时，上述同名结果会被新结果覆盖。打开 PNG 的软件如果持续占用文件，Windows 可能阻止覆盖；遇到提示时先关闭图片查看器、PowerPoint 或其他占用该文件的程序，再重新运行。

## 11. 结果质控

完成后不要只看牛眼图，建议按以下顺序检查：

1. 在 3D Slicer 中把 `*_lge_scar_mask.nii.gz` 叠加到原 LGE；
2. 检查 scar 是否严格位于输入心肌 mask 内；
3. 检查高信号血池边缘、心外脂肪和伪影是否被误分；
4. 对照 `*_lge_normal_rois.json` 检查每层 ROI 体素数和阈值；
5. 对照 `*_lge_scar_segments.csv` 检查每个节段的分母是否合理；
6. 确认牛眼图节段方向与 AIP/IIP 标记一致；
7. 结合原始 LGE、cine 和其他可用序列进行视觉复核。

建议保存操作记录，包括软件版本、配置文件、操作者、ROI 选择日期以及需要人工说明的图像质量问题。

## 12. 常见问题

### 12.1 找不到命令

错误示例：

```text
cardiac-lge-bullseye : The term ... is not recognized
```

处理：

```powershell
conda activate cardiac-bullseye
python -m pip install -e E:\ZZR\Code\Tools\Bulleyes
Get-Command cardiac-lge-bullseye
```

### 12.2 找不到配置文件

确认当前位于项目根目录，并使用实际文件名：

```powershell
Set-Location E:\ZZR\Code\Tools\Bulleyes
Test-Path .\toml\case_032602_lge.toml
```

返回 `True` 后再运行命令。

### 12.3 LGE 与 mask 几何信息不同

典型错误包含：

```text
LGE and myocardium mask size/spacing/origin/direction differ
```

原因是二者不在同一物理网格。不要只修改文件头来绕过检查；应把 mask 按最近邻插值真正重采样到 LGE 网格，并检查重采样结果。

### 12.4 图形窗口无法打开

程序需要交互式 Matplotlib 后端。若 Qt 窗口无法出现：

```powershell
python -m pip install --upgrade matplotlib PyQt5
```

在 Spyder 中运行时，建议把图形后端设置为 Qt，并重启控制台。不要使用只在 Plots 面板中显示静态图的 Inline 后端。

### 12.5 ROI 不能接受

如果提示：

```text
ROI has ... valid myocardial voxels; at least 20 are required
```

说明与心肌 mask 相交后的有效 ROI 少于 `normal_roi_min_voxels`。应扩大正常心肌 ROI，并确认没有把画刷涂在 mask 外。不要仅为了通过检查随意降低最少体素数。

### 12.6 删除层报错

如果提示删除层必须构成连续基底端尾部，说明在中间层使用了 `Delete`，或者删除序列不连续。重新运行并只对最靠近 Base 的连续若干层使用 `Delete`；难以标记但仍属于左室的中间层应使用 `Skip`。

### 12.7 输出 PNG 无法覆盖

关闭正在打开旧 PNG 的图片查看器或文档软件后重试。程序会先在输出目录完整写入临时 PNG，再原子替换同名结果，并对 Windows 的短暂文件占用自动重试。

## 13. 一页式操作检查清单

### 运行前

- [ ] 已激活正确 Conda 环境；
- [ ] `cardiac-lge-bullseye --help` 可以运行；
- [ ] LGE 是三维短轴标量图像；
- [ ] 层序为 Apex → Base；
- [ ] mask 覆盖完整左室心肌；
- [ ] LGE 与 mask 的 size、spacing、origin、direction 一致；
- [ ] TOML 中病例编号、输入路径和输出目录正确。

### AIP/IIP 标记

- [ ] 每层按 AIP → IIP 顺序点击；
- [ ] 放大后确认点位位于两个 RV 插入点；
- [ ] 只有无法可靠识别点位时才使用 `Skip`；
- [ ] `Delete` 只用于连续的基底端尾部；
- [ ] 知道默认会从第 1 层重新标记；仅在图像兼容时才启用 `reuse_saved_points = true`。

### 正常心肌 ROI

- [ ] 每个有效层都重新涂抹 ROI；
- [ ] ROI 位于无 LGE 的远端正常心肌；
- [ ] 避开血池、边界、脂肪、伪影和微血管阻塞区；
- [ ] ROI 不少于配置要求的有效体素数；
- [ ] 接受前检查红色 scar 预览是否合理。

### 运行后

- [ ] 七个输出文件均已生成；
- [ ] scar mask 已叠加回原 LGE 检查；
- [ ] CSV 的心肌体素数和 scar 比例合理；
- [ ] 牛眼图节段方向正确；
- [ ] 对异常病例和图像质量问题进行了记录。

## 14. 参考资料

- Schulz-Menger J, et al. [Standardized image interpretation and post-processing in cardiovascular magnetic resonance — 2020 update](https://pmc.ncbi.nlm.nih.gov/articles/PMC7066763/). *Journal of Cardiovascular Magnetic Resonance*. 2020;22:19.
- Heiberg E, et al. [Infarct quantification with cardiovascular magnetic resonance using “standard deviation from remote” is unreliable: validation in multi-centre multi-vendor data](https://pubmed.ncbi.nlm.nih.gov/36336693/). 该研究提示远端 ROI 位置、面积和 SD 阈值会影响结果，因此跨病例或跨中心研究必须统一操作规范并进行视觉质控。
