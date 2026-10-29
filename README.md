# Cardiac Bullseye

将心脏 MR 转换为短轴堆栈，并生成患者特异性的 AHA-17 供血区牛眼图、冠脉投影和冠脉距离牛眼图。

项目提供六个独立工作流/命令：只生成短轴、使用已有短轴数据生成供血区牛眼图、生成
连续强度牛眼图、一次完成“原始 MR → 短轴 → 牛眼图”、完整 LGE 强度与 scar 牛眼图，
以及使用已有 scar mask 的外部 scar 牛眼图。后两种 LGE 模式分别对应
`cardiac-lge-bullseye` 和 `cardiac-lge-mask-bullseye`。

## 目录结构

```text
Bulleyes/
├── src/cardiac_bullseye/
│   ├── short_axis/       # 短轴规划、重采样和 geometry.json
│   ├── territory/        # 供血区处理和 AHA-17 映射
│   ├── coronary/         # 冠脉中心线提取与投影
│   ├── distance/         # 冠脉距离体积和距离牛眼图
│   ├── intensity/        # 任意短轴图像的连续强度牛眼图
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
Set-Location .\Bulleyes
python -m pip install -e .
```

安装后会提供六个命令：

```text
cardiac-short-axis   只生成短轴
cardiac-bullseye     只生成牛眼图
cardiac-intensity-bullseye 由短轴 image 和心肌 mask 生成连续强度牛眼图
cardiac-pipeline     运行完整流程
cardiac-lge-bullseye 由短轴 LGE 和同网格心肌 mask 生成强度与 scar 牛眼图
cardiac-lge-mask-bullseye --config <file> 由已有 scar mask 生成 LGE 牛眼图
```

可以用以下命令确认安装是否成功：

```powershell
cardiac-short-axis --help
cardiac-bullseye --help
cardiac-intensity-bullseye --help
cardiac-pipeline --help
cardiac-lge-bullseye --help
cardiac-lge-mask-bullseye --help
```

## 配置文件通用规则

建议每条命令使用一份独立的 TOML 配置文件。配置文件只需要包含当前命令用到的
section，不需要把六种流程的配置全部写在同一个文件里。

- 每份配置都必须有 `[case]`；`case.id` 是输出文件名前缀。
- `case.output_directory` 是结果保存目录，不存在时会自动创建。
- 相对路径以 TOML 文件所在目录为基准，而不是以 PowerShell 当前目录为基准。
- Windows 路径建议使用 TOML 单引号和正斜杠，例如
  `'inputs/patient_043_lge_sax.nii.gz'`。
- 下列示例假设配置文件位于项目根目录，数据放在 `inputs/` 下；实际使用时请替换为
  自己的数据路径。
- 所有可选参数和默认值见 [`config.example.toml`](config.example.toml)。

### 跨命令、跨输出目录共享 AIP/IIP

`[bullseye]`、`[lge_bullseye]` 和 `[lge_mask_bullseye]` 都支持可选的
`landmarks_path`。例如在三份独立 TOML 的对应 section 中都设置：

```toml
landmarks_path = 'shared/patient_043_landmarks.json'
reuse_saved_points = true
```

该路径同样相对于各自 TOML 所在目录解析；确保解析后指向同一文件，即可让不同
`output_directory` 共享一个 JSON。省略时，`[bullseye]` 使用输出目录中的
`<case>_landmarks.json`，两个 LGE 命令使用 `<case>_lge_landmarks.json`。
`reuse_saved_points = true` 且文件存在时复用 AIP/IIP、Skip/Delete 和手动质心，
不打开标注窗口，也不改写 JSON；文件不存在时自动打开所选模式的标注窗口。
设为 `false` 时从空白状态重新标注，成功保存时覆盖配置路径中的旧数据。

共享只适用于严格相同的短轴网格：层数、Apex → Base 顺序、像素坐标、层内方向、
物理位置和心肌覆盖范围都必须对应；LGE 会先裁掉首尾没有心肌的空层，因此共享
JSON 必须对应裁剪后的层索引。相同 shape 本身不能证明配准；请自行确认物理网格。
程序检查保存的图像 shape、层条目、点坐标和版本，不会自动配准或重采样。
`Delete` 必须从基底端连续删除；至少保留两个不同位置的心肌层，否则映射前报错。

新 JSON 写入 `coordinate_version = 1`，采用共同的 `(slice, x, y)` 图像顺序和
对应的层内像素坐标。无版本的旧 `*_landmarks.json`（不包括 `*_lge_landmarks.json`）按已知
共同格式读取；只有未设置 `landmarks_path` 时，默认的旧
`<case>_lge_landmarks.json` 才会在内存中迁移旧 LGE 的转置坐标（含手动质心）。
迁移不改写原文件。显式共享路径即使名为 `*_lge_landmarks.json` 也不会启用该
旧 LGE 推断；其他来源不明的无版本文件和不支持的版本会被拒绝，需重新标注保存。

## 用法一：只生成短轴

命令：

```powershell
cardiac-short-axis --config .\case_short_axis.toml
```

需要的数据：

- `source_mr`：需要重格式化的原始 MR，必需。
- `source_mask`：与原始 MR 同网格的供血区或其他标签 mask，可选；填写后才会同步生成短轴 mask。
- `orientation_reference`：参考 LGE 短轴，可选；用于确定层内旋转方向和参考长轴方向。

`case_short_axis.toml`：

```toml
[case]
id = 'patient_043'
output_directory = 'outputs/patient_043_short_axis'

[short_axis]
source_mr = 'inputs/patient_043_mr.nii.gz'
source_mask = 'inputs/patient_043_territory.nii.gz'

# 以下参数可省略。
# reuse_saved_points = false
# orientation_reference = 'inputs/patient_043_lge_sax.nii.gz'
# slice_spacing_mm = 8.0
# pixel_spacing_mm = 1.25
# fov_mm = [192.0, 192.0]
```

交互窗口依次完成 Apex/二尖瓣中心选择、长轴确认和 Apex–Base 覆盖范围确认。

输出：

```text
patient_043_short_axis.nii.gz
patient_043_short_axis_mask.nii.gz   # 仅在配置 source_mask 时生成
patient_043_geometry.json
patient_043_short_axis_points.json
```

程序每次成功保存短轴时都会写出 `short_axis_points.json`。默认
`reuse_saved_points = false`，仍会重新打开规划窗口；改为 `true` 后直接读取该文件，
跳过 Apex/二尖瓣中心和 Apex–Base 范围选择，并用相同短轴几何重采样新的 label mask。
源 MR 的 shape 和 affine 必须与保存关键点时一致，否则程序会拒绝复用。

## 用法二：使用已有短轴数据生成血管灌注区域牛眼图和距离图

命令：

```powershell
cardiac-bullseye --config .\case_bullseye.toml --mode per-slice
```

`--mode` 控制 AIP/IIP 标注方式：

```powershell
# 现有方式：在每个保留层分别选择 AIP/IIP，并保持原有 r=1–4 投影。
cardiac-bullseye --config .\case_bullseye.toml --mode per-slice --projection 1-4

# 单参考层方式，并把真实心尖范围纳入 r=0–4。
cardiac-bullseye --config .\case_bullseye.toml --mode single-reference --projection 0-4
```

省略 `--mode` 时默认使用 `per-slice`，因此原有命令和脚本保持兼容。
省略 `--projection` 时默认使用 `1-4`。

需要的数据：

- 已有短轴 MR、短轴供血区 mask，以及同次短轴规划生成的 `geometry.json`。
- 世界坐标系下的冠脉 mask 和心肌 mask。
- `short_axis_mr`、`short_axis_mask` 和 `geometry_json` 必须来自同一病例、同一次短轴规划。

`geometry.json` 保存短轴体素坐标与原始世界坐标之间的关系，程序需要它把世界空间
的冠脉和心肌数据投影到短轴牛眼图中，因此不能用其他病例或其他重采样产生的文件代替。

`case_bullseye.toml`：

```toml
[case]
id = 'patient_043'
output_directory = 'outputs/patient_043_bullseye'

[bullseye]
short_axis_mr = 'outputs/patient_043_short_axis/patient_043_short_axis.nii.gz'
short_axis_mask = 'outputs/patient_043_short_axis/patient_043_short_axis_mask.nii.gz'
geometry_json = 'outputs/patient_043_short_axis/patient_043_geometry.json'
coronary_mask = 'inputs/patient_043_coronary.nii.gz'
world_myocardium_mask = 'inputs/patient_043_myocardium.nii.gz'

# 可选：与其他命令共享；省略时使用输出目录中的 <case>_landmarks.json。
# landmarks_path = 'shared/patient_043_landmarks.json'
# 默认 false；true 时优先读取上述路径，缺失时打开标注窗口。
reuse_saved_points = false

# 下面是常用可选参数；省略时使用程序默认值。
# 输入Territory mask 原始值（可改） = 程序内部的标准标签
territory_label_map = { 2 = 1, 3 = 2, 4 = 3 }
coronary_mask_is_centerline = false

# 从 coronary_mask 中提取标签值为 2、3、4 （可改）的血管，计算每个心肌体素到这些冠脉的最短三维距离。
distance_vessel_labels = [2, 3, 4]
show_final_figure = true
```

程序逐层重新选择 AIP 和 IIP。滚轮缩放，中键拖动平移，`Fit` 恢复视野；
`Accept` 接受本层，`Redo` 重选，`Skip` 跳过本层，其层会由相邻有关键点标注的层插值。
`Delete` 仅用于从基底端连续移除截面（用于手动对齐LGE）。
设置`reuse_saved_points = true` 后会保留 Skip/Delete 状态并完全跳过标注窗口。
保存文件的层数、图像尺寸和点坐标必须与当前短轴图兼容。

在 `single-reference` 模式中，窗口默认打开中间的有效心肌层。使用 `Previous`
和 `Next` 浏览层面，在最清晰的一层选择 AIP、IIP，然后点击
`Accept Reference`。程序不会把同一绝对像素坐标复制到所有层，而是保存参考层
中“质心到 AIP/IIP”的两个方向，并围绕每层自己的质心生成角度参考点。这样每层
仍使用自己的供血区和距离数据，同时 Territory、距离图和冠脉投影使用一致的角度
标准。`Delete`/`Restore` 可在接受参考层前排除或恢复层面。生成的 landmark JSON
会记录 `annotation_mode`、零基参考层索引和便于查看的一基参考层编号。

每层首先按自动规则显示绿色 `+` 质心。若位置不合适，可用鼠标左键直接拖动
绿色 `+` 修正；在其他位置左键单击仍用于依次选择 AIP、IIP。手动质心作为覆盖值
写入 landmark JSON。即使随后点击 `Skip`，该层的手动质心仍会保留；`Redo` 会
同时清空本层 AIP/IIP 并恢复自动质心，`Delete` 则丢弃本层手动质心。
`reuse_saved_points = true` 会同时复用 AIP/IIP、Skip 状态和手动质心。旧版不含
质心字段的 landmark JSON 仍兼容，并继续使用自动质心。

几何中心按整套短轴逐层计算：闭合心肌层使用其包围的 LV 心腔质心；C 形等不闭合
心肌层不再使用心肌自身质心，而是复制层索引最近的闭合心肌层中心。若两侧距离
相同，优先采用索引较大（更靠基底）的闭合层。空 mask 层不参与匹配；如果整套
非空心肌层都没有闭合心腔，程序会在标注/计算前报错。该中心同时用于界面中的
绿色十字、角度映射和后续牛眼图计算。

`--projection` 控制心肌 mask 沿长轴映射到牛眼图的范围：

- `1-4`：兼容原有行为。第一张含心肌 mask 的层映射到 `r=1`，最后一个未
  `Delete` 的基底层映射到 `r=4`。中心是推断值：第17节段供血区由13–16节段
  的平均概率决定，距离值取13–16四个节段均值；冠脉中落在 `r<1` 的点不显示。
- `0-4`：第一张含心肌 mask 的层映射到 `r=0`，基底层仍映射到 `r=4`。
  Territory 与距离图在 `r<1` 显示真实层间插值结果，不再用统一圆形覆盖；第17
  节段归属由整个 `r<1` 区域的平均概率多数决定。该模式忽略配置中的固定
  `apex_territory`。冠脉使用相同的 `0–4` 坐标并保留所有 `r>=0` 的投影点。

两种模式下 Territory、距离图和冠脉叠加始终共用同一套径向坐标。

如果要与未配准的 LGE scar 牛眼图比较，必须人工保证两套短轴图的第一张有 mask
层以及最后一个未 `Delete` 基底层分别对应。这是标准化 AHA 坐标对应，不是体素级配准。

输出：

```text
patient_043_landmarks.json #手动选择的关键点，方便下次直接复用
patient_043_bullseye.png
patient_043_distance_bullseye.png
patient_043_distance_world.nii.gz
patient_043_distance_short_axis.nii.gz
```

## 用法三：由短轴图像生成连续强度牛眼图

该命令与 `cardiac-bullseye` 使用相同的短轴视图、AIP/IIP 标注界面和
`<case>_landmarks.json`。默认不读取供血区标签、冠脉或 `geometry.json`；也可以通过
TOML 开关把注册后的冠脉投影到强度牛眼图。它在每层、每个角度扇区内计算
`short_axis_mask > 0` 区域的原始 image 平均值，然后插值为连续牛眼图。

命令：

```powershell
# 每个保留层分别标注 AIP/IIP，保持原有 r=1–4 投影。
cardiac-intensity-bullseye --config .\case_intensity.toml --mode per-slice --projection 1-4

# 只选一个参考层，并将最靠心尖有效层映射到 r=0。
cardiac-intensity-bullseye --config .\case_intensity.toml --mode single-reference --projection 0-4
```

最小配置 `case_intensity.toml`：

```toml
[case]
id = 'patient_043'
output_directory = 'outputs/patient_043_intensity'

[bullseye]
short_axis_mr = 'outputs/patient_043_short_axis/patient_043_short_axis.nii.gz'
short_axis_mask = 'outputs/patient_043_short_axis/patient_043_short_axis_mask.nii.gz'

# false：每次重新标注并覆盖共享 landmark；true：优先读取共享 landmark。
reuse_saved_points = false

# 以下参数可省略。
angle_count = 360
radial_count = 300
angular_smoothing_sigma_deg = 1.5
radial_smoothing_sigma = 1.5
show_final_figure = true
```

可选冠脉叠加配置：

```toml
[bullseye]
# 上面的 short_axis_mr、short_axis_mask 等配置保持不变。
intensity_coronary_overlay = true
geometry_json = 'outputs/patient_043_short_axis/patient_043_geometry.json'
coronary_mask = 'inputs/patient_043_coronary.nii.gz'

# 以下冠脉提取和显示参数与 cardiac-bullseye 共用，均可省略。
coronary_mask_is_centerline = false
coronary_min_path_length_mm = 2.0
coronary_resample_step_mm = 0.75
coronary_smoothing_sigma_voxels = 0.0
coronary_label_colors = { 1 = '#d62728', 2 = '#ff7f0e', 3 = '#9467bd' }
coronary_label_names = { 1 = 'Coronary LAD', 2 = 'Coronary LCX', 3 = 'Coronary RCA' }
coronary_linewidth = 1.8
```

开启后，`geometry_json` 必须与 `short_axis_mr` 的 shape 和 affine 完全匹配，
`coronary_mask` 必须处于该 geometry 所描述的世界坐标系。该功能不需要
`world_myocardium_mask`，因为它只画冠脉，不计算距离图。冠脉直接叠加到原来的
`<case>_intensity_bullseye.png`，不会生成第二张图。

`short_axis_mr` 与 `short_axis_mask` 必须 shape 和 affine 一致；mask 中所有非零标签
都视为心肌。`Skip` 层不直接采样，其数值由相邻有效层插值得到；`Delete` 层完全
排除。`--projection 1-4` 保持原逻辑：最靠心尖有效层位于 `r=1`，中心 `r<1`
使用该层全部有效心肌体素的原始平均值统一填充。`--projection 0-4` 则把最靠心尖
有效层映射到 `r=0`，并在整个 `r=0–4` 范围显示按角度和层面插值得到的真实数值，
不再额外覆盖中心。为避免所有角度在极坐标原点汇聚形成放射状伪影，强度命令会在
`r<1` 内使用 smoothstep 中心融合：`r=0` 收敛为该半径的环向平均值，并平滑恢复
角度差异，到 `r=1` 时完全保留原始插值结果。该融合仅用于
`cardiac-intensity-bullseye` 和两个 LGE 命令。两种模式都把最后一个未 `Delete`
的基底层映射到 `r=4`；省略 `--projection` 时默认使用 `1-4`。

开启冠脉叠加时，冠脉使用同一个径向框架：`--projection 1-4` 不绘制 `r<1` 的
冠脉点，`--projection 0-4` 则允许冠脉一直投影到中心。中心感知融合只修改强度
背景，不修改冠脉线的位置或形状。

当 `reuse_saved_points = true` 且输出目录中已有 `<case>_landmarks.json` 时，程序直接
复用 AIP/IIP、Skip/Delete 和手动质心。若文件不存在，则自动打开标注界面；设为
`false` 时始终重新标注，并覆盖同名共享 landmark。因此，只要两条命令的输入短轴
栈完全一致，就可以与 `cardiac-bullseye` 共用参考点。

输出：

```text
patient_043_landmarks.json
patient_043_intensity_bullseye.png
```

图像采用 Matplotlib 的 `CMRmap` 热力图，colorbar 保留输入 image 的原始数值单位。

## 用法四：从原始 MR 一次完成短轴和牛眼图

命令：

```powershell
cardiac-pipeline --config .\case_pipeline.toml
```

需要的数据：

- 原始 MR 和与其同网格的供血区 mask。
- 世界坐标系下的冠脉 mask 和心肌 mask。
- 不需要填写 `bullseye.short_axis_mr`、`short_axis_mask` 或 `geometry_json`；程序会直接使用本次生成的短轴结果。

`case_pipeline.toml`：

```toml
[case]
id = 'patient_043'
output_directory = 'outputs/patient_043_pipeline'

[short_axis]
source_mr = 'inputs/patient_043_mr.nii.gz'
source_mask = 'inputs/patient_043_territory.nii.gz'
# orientation_reference = 'inputs/patient_043_lge_sax.nii.gz'

[bullseye]
coronary_mask = 'inputs/patient_043_coronary.nii.gz'
world_myocardium_mask = 'inputs/patient_043_myocardium.nii.gz'
territory_label_map = { 2 = 1, 3 = 2, 4 = 3 }
coronary_mask_is_centerline = false
distance_vessel_labels = [2, 3, 4]
show_final_figure = true
```

该命令先执行用法一的短轴规划，再直接执行用法二的 AIP/IIP 标注和牛眼图计算。
输出：

```text
patient_043_short_axis.nii.gz
patient_043_short_axis_mask.nii.gz
patient_043_geometry.json
patient_043_landmarks.json
patient_043_bullseye.png
patient_043_distance_bullseye.png
patient_043_distance_world.nii.gz
patient_043_distance_short_axis.nii.gz
```

## 用法五：完整 LGE 流程——手动画正常心肌 ROI 并计算 scar

命令：

```powershell
cardiac-lge-bullseye --config .\case_lge.toml
# 逐层标注，最靠心尖心肌层从 r=1 开始。
cardiac-lge-bullseye --config .\case_lge.toml --mode per-slice --projection 1-4
# 单参考层标注，包含真实的 r<1 心尖范围。
cardiac-lge-bullseye --config .\case_lge.toml --mode single-reference --projection 0-4
```

需要的数据：

- Apex → Base 排列的短轴 LGE。
- 直接在该 LGE 网格上标注的心肌 mask；支持标量 NIfTI/NRRD 和 vector `.seg.nrrd`。
- LGE 与心肌 mask 的 size、spacing、origin 和 direction 必须一致；程序不会自动重采样。

`case_lge.toml`：

```toml
[case]
id = 'patient_043'
output_directory = 'outputs/patient_043_lge'

[lge_bullseye]
lge = 'inputs/patient_043_lge_sax.nii.gz'
mask = 'inputs/patient_043_lge_myocardium.seg.nrrd'
# 可选：跨输出目录共享 AIP/IIP。
# landmarks_path = 'shared/patient_043_landmarks.json'

# 以下均为可选参数，右侧是默认值。
angle_count = 360
radial_count = 300
angular_smoothing_sigma_deg = 1.5
radial_smoothing_sigma = 1.5
six_segment_warp = true
scar_sd_multiplier = 5.0
normal_roi_min_voxels = 20
reuse_saved_points = false
```

流程分为两步：

1. 根据 `--mode` 逐层选择 AIP/IIP，或选择一个参考层后按各层质心传播方向。
   可缩放和平移，也可使用 `Skip` 或从基底端连续 `Delete`。
   绿色 `+` 为自动质心，可直接拖动修正；在 `Skip` 前完成的质心修正仍会保存。
2. 在每个保留层用画刷选择远端正常心肌 ROI。ROI 自动与心肌 mask 求交；少于
   `normal_roi_min_voxels` 个有效体素时会提示。程序逐层使用
   `mean + scar_sd_multiplier × sample SD` 计算 scar 阈值（样本标准差，`ddof=1`），
   只有严格大于阈值的心肌体素才算 scar，等于阈值不计入。

默认每次运行都会忽略以前的 landmark 和 ROI 标注，从空白状态重新选择。
设置 `reuse_saved_points = true` 后，按上述共享路径规则优先复用 landmark；
正常心肌 ROI 仍然每次重新选择，旧的 ROI JSON 缓存不会复用。
LGE 强度图和 scar 分段采用与 Territory 相同的逐层几何中心和共享径向框架。

输出：

```text
patient_043_lge_landmarks.json
patient_043_lge_intensity_bullseye.npz
patient_043_lge_intensity_bullseye.png
patient_043_lge_normal_rois.json
patient_043_lge_scar_mask.nii.gz
patient_043_lge_scar_segments.csv
patient_043_lge_scar_bullseye.png
```

scar 牛眼图使用原始体素比例；0% 为白色，比例越高越趋向橙红色。
PNG 使用安全覆盖逻辑，scar NIfTI 保留原 LGE 的完整物理网格。

两个 LGE 命令都接受 `--mode per-slice|single-reference` 与
`--projection 1-4|0-4`，可任意组合；省略时分别为 `per-slice` 和 `1-4`，
原来仅传 `--config` 的调用仍有效。它们以裁剪后第一张含心肌 mask 的层为心尖锚点，
以最后一个未 Delete 基底层为基底锚点；强度插值与原生 scar AHA 分段使用完全相同
的源层半径。Skip 不移动锚点：强度通过相邻有效层插值，scar 仍统计该层原生体素，
其角度参考由相邻有效 AIP/IIP 插值得到。

- `1-4`：心尖锚点为 `r=1`、基底为 `r=4`，测量 AHA 1–16；完整 LGE 的
  强度中心使用心尖层心肌平均值，scar 中心为 `N/A`，CSV 只有16行。
  外部 scar 命令中心保持空白，不绘制或标注第17节段。
- `0-4`：心尖锚点为 `r=0`、基底为 `r=4`。准确边界为
  `0 <= r < 1` 属第17节段，`1 <= r < 2` 属13–16，
  `2 <= r < 3` 属7–12，`3 <= r <= 4` 属1–6。
  第17节段 scar 百分比为该范围内汇总的原生 scar 体素数 / 原生心肌体素数 × 100，
  不是各层百分比的平均值，也不是插值图像的像素比例；其中心按测量值着色和标注，
  无心肌体素时显示 `N/A`。完整 LGE 的 CSV 包含17行。
  连续强度绘制到原点，在 `r<1` 内按 smoothstep 融合到每个半径的环向均值：
  原点无角度差异，到 `r=1` 完全恢复插值强度；不覆盖额外的统一中心圆。

## 用法六：使用已有 scar mask 生成 LGE 牛眼图

这条命令适用于 scar mask 已由独立 `lgeMask` 工具或其他方法生成的情况。它不会
再次选择正常心肌 ROI，也不会重新计算 scar。

命令：

```powershell
cardiac-lge-mask-bullseye --config .\case_lge_mask.toml
cardiac-lge-mask-bullseye --config .\case_lge_mask.toml --mode per-slice --projection 1-4
cardiac-lge-mask-bullseye --config .\case_lge_mask.toml --mode single-reference --projection 0-4
```

需要的数据：

- Apex → Base 排列的短轴 LGE。
- 心肌 mask。
- 已有的二值 scar mask。
- 三份数据的 size、spacing、origin 和 direction 必须完全一致；程序不会重采样。
- scar 必须完全位于心肌内，否则程序在打开标注窗口前报错。

心肌 mask 支持标量 NIfTI、标量 NRRD 和 vector/component `.seg.nrrd`；对于
`.seg.nrrd`，任一 component 非零即视为心肌。

`case_lge_mask.toml`：

```toml
[case]
id = 'patient_043'
output_directory = 'outputs/patient_043_lge_mask_bullseye'

[lge_mask_bullseye]
lge = 'inputs/patient_043_lge_sax.nii.gz'
mask = 'inputs/patient_043_lge_myocardium.seg.nrrd'
lge_scar_mask = 'inputs/patient_043_lge_scar_mask.nii.gz'
# 可选：与完整 LGE、Territory 或强度命令共享同网格 landmark。
# landmarks_path = 'shared/patient_043_landmarks.json'

# 以下均为可选参数，右侧是默认值。
angle_count = 360
radial_count = 300
angular_smoothing_sigma_deg = 1.5
radial_smoothing_sigma = 1.5
six_segment_warp = true
reuse_saved_points = false
```

程序只需要准备 AIP/IIP；不会进入正常心肌 ROI 界面。默认 `1-4` 计算并显示
AHA 1–16，中心空白；`0-4` 还测量、着色并标注第17节段，强度连续绘制到原点。
选择 AIP/IIP 时同样会先显示自动质心；可拖动绿色 `+` 修正，并可在修正后点击
`Skip` 保留该层手动质心。
设置 `reuse_saved_points = true` 后按共享路径规则优先读取已保存标注；缺失时仍会
打开所选模式的窗口。本模式同样对不闭合心肌层使用最近闭合层的 LV 心腔质心。

输出恰好为：

```text
patient_043_lge_landmarks.json
patient_043_lge_intensity_bullseye.png
patient_043_lge_scar_bullseye.png
```

该命令不会生成 NPZ、CSV、ROI JSON，也不会复制输入 scar mask。

## Python 调用

五个命令均有对应的 Python API：

```python
from cardiac_bullseye.config import load_config
from cardiac_bullseye.workflows import (
    generate_bullseye,
    generate_lge_bullseye,
    generate_lge_mask_bullseye,
    generate_short_axis,
    run_complete_pipeline,
)

short_axis_config = load_config("case.toml", "short-axis")
short_axis_outputs = generate_short_axis(short_axis_config)

bullseye_config = load_config("case.toml", "bullseye")
bullseye_outputs = generate_bullseye(bullseye_config)

pipeline_config = load_config("case.toml", "pipeline")
pipeline_outputs = run_complete_pipeline(pipeline_config)

lge_config = load_config("case_lge.toml", "lge-bullseye")
lge_outputs = generate_lge_bullseye(lge_config)

lge_mask_config = load_config("case_lge_mask.toml", "lge-mask-bullseye")
lge_mask_outputs = generate_lge_mask_bullseye(
    lge_mask_config, landmark_mode="single-reference", projection="0-4"
)
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
Set-Location .\Bulleyes
$env:PYTHONPATH = "$PWD/src"
python -m pytest tests -v
```
