# Par_A1 方案二修改记录

日期：2026-03-24

## 1. 修改目标

按照 `Par_A1.md` 的方案二，将 YOLOE 的 visual prompt 推理从“旧类集合被提示类完全覆盖”改为“20 个已有缺陷类 + 新提示缺陷类”在同一个检测头、同一个分数空间内联合推理。

核心目标有两点：

1. 保留原有 20 个旧类名称列表 `base_names`，并提取对应文本原型作为基类原型 `base_pe`。
2. 提取新缺陷视觉提示原型 `vpe` 与名称列表 `vp_names` 后，执行统一拼接：
   `all_names = base_names + vp_names`
   `all_pe = torch.cat([base_pe, vpe], dim=1)`

同时，将推理时原本被强制打开的 `agnostic_nms=True` 改为可配置项，混合旧类/新类时推荐使用 class-aware NMS，也就是 `agnostic_nms=False`。

## 2. 实际修改文件

### 2.1 `ultralytics/models/yolo/model.py`

对 `YOLOE.predict()` 做了以下修改：

1. 新增旧类原型装配逻辑
   - 新增 `_get_base_prompt_bank()`，用于从当前模型类别名中提取基类名称，并生成：
     - `base_names`
     - `base_tpe`：原始文本特征
     - `base_pe`：经过 head `get_tpe()` 处理后的基类原型
   - 默认取前 `20` 个类别作为旧类，可通过 `base_nc` 覆盖。
   - 也支持通过 `base_names=[...]` 显式传入旧类名列表。

2. 新增 visual prompt 规范化逻辑
   - 新增 `_normalize_visual_prompts()`，将提示类别重映射为连续 ID。
   - 单图推理时保留提示名称列表 `vp_names`。

3. 修改 visual prompt 分支的推理装配
   - 不再执行“只保留提示类”的覆盖逻辑。
   - 改为保留旧类 `base_names`，再与提示类 `vp_names` 拼接为 `all_names`。

4. 两种 visual prompt 推理路径都支持联合原型库
   - `refer_image is not None`：
     - 从参考图提取 `vpe`
     - 执行 `all_pe = torch.cat([base_pe, vpe], dim=1)`
     - 调用 `self.model.set_classes(all_names, all_pe)`
   - `refer_image is None`：
     - 通过 predictor 在前向时同时传入 `base_tpe` 和 visual prompt
     - 让模型内部按 `tpe + vpe` 做统一分类

5. 修改 NMS 默认行为
   - 删除了原本强制写死的 `self.overrides["agnostic_nms"] = True`
   - 改为：
     - `kwargs.setdefault("agnostic_nms", self.overrides.get("agnostic_nms", False))`
   - 这样调用时可显式控制，默认回到 class-aware NMS。

### 2.2 `ultralytics/models/yolo/yoloe/predict.py`

为了支持 `refer_image is None` 的“旧类文本原型 + 新类视觉提示”联合推理，补了 predictor 联动：

1. 新增 `set_base_text_prompts(self, base_tpe)`
   - 缓存旧类原始文本特征。

2. 调整 `set_prompts()`
   - 每次设置新 prompt 时会清空旧的 `base_tpe` 缓存，避免串状态。

3. 调整 `inference()`
   - 前向时同时传入：
     - `tpe=self.base_tpe`
     - `vpe=self.prompts`
   - 让 YOLOE 内部直接使用统一 prototype bank 做分类。

## 3. 修改后的推理使用方法

## 3.1 推荐用法：参考图模式

适合先在参考图上框出新缺陷，再对目标图片/目录做联合推理。

```python
import numpy as np
from ultralytics import YOLOE

model = YOLOE("./runs/detect/VP/weights/best.pt")

visual_prompts = dict(
    bboxes=np.array([[383, 620, 469, 744]]),
    cls=["novel_defect_a"],
)

results = model.predict(
    source="./testImg",
    refer_image="./testImg/1.jpg",
    visual_prompts=visual_prompts,
    conf=0.2,
    agnostic_nms=False,
    base_nc=20,
    save=True,
)
```

说明：

- `base_nc=20` 表示保留前 20 个旧类。
- `agnostic_nms=False` 表示启用 class-aware NMS，混合旧类/新类时推荐这样用。
- 若 `cls` 传字符串，则新类名称会直接体现在结果标签中。

## 3.2 直接 visual prompt 模式

如果不提供 `refer_image`，则会在当前推理流程中同时送入：

- 旧类文本特征 `base_tpe`
- 新类 visual prompt 特征

示例：

```python
results = model.predict(
    source="./testImg/2.jpg",
    visual_prompts=dict(
        bboxes=np.array([[383, 620, 469, 744]]),
        cls=["novel_defect_a"],
    ),
    conf=0.2,
    agnostic_nms=False,
    base_nc=20,
)
```

## 3.3 如果要显式指定旧类名称

当当前权重中的 `model.names` 不可靠，或你不想简单取前 20 类时，可以直接传 `base_names`：

```python
base_names = [
    "old_defect_01",
    "old_defect_02",
    "...",
    "old_defect_20",
]

results = model.predict(
    source="./testImg",
    refer_image="./testImg/1.jpg",
    visual_prompts=visual_prompts,
    base_names=base_names,
    agnostic_nms=False,
)
```

当前仓库 `ultralytics/cfg/datasets/ybj20260303-yoloe.yaml` 的前 20 个名称会被默认视为旧类；如果业务定义和这里不一致，建议总是显式传 `base_names`。

## 4. 推理快速验证

推荐按下面的顺序做快速检查：

1. 标签检查
   - 运行一次带 `visual_prompts` 的推理。
   - 确认结果标签里同时出现旧类名和新 prompt 类名，而不是只剩 `object0/object1/...`。

2. 类别数检查
   - 对参考图模式，推理前后检查 `len(model.names)` 或结果里的 `names`。
   - 期望值应为 `20 + 新提示类数`。

3. NMS 行为检查
   - 分别对比：
     - `agnostic_nms=True`
     - `agnostic_nms=False`
   - 对相邻且外观相似的新旧缺陷，通常 `agnostic_nms=False` 更不容易相互错误抑制。

4. 单图可视化检查
   - 在同一张图中同时观察：
     - 已有旧类缺陷是否还能被正常检出
     - 新提示缺陷是否能被检出
     - 两者是否共享同一套结果输出与排序

本次在当前环境下完成了语法级静态校验，但由于环境中缺少完整运行依赖，未执行真实 GPU 推理验证。

## 5. 训练方法说明

这次修改是“推理级联合原型库”改造，没有改 loss、trainer 和数据流，因此训练仍建议沿用现有两阶段方式：

1. 旧类闭集训练
   - 使用 `trainE.py`
   - 训练已有 20 类缺陷的基础检测能力

2. visual prompt 能力微调
   - 使用 `trainE_VP.py`
   - 从旧类模型继续训练 `savpe` 相关能力

3. 联合推理落地
   - 推理时通过本次改造后的 `model.predict(..., visual_prompts=..., ...)`
   - 动态拼接 `base_pe + vpe`
   - 实现旧类 + 新类单模型联合推理

也就是说，当前版本不要求你先完成“联合训练”才能用方案二。现阶段可以先落地：

- 旧类正常训练
- VP 能力正常训练
- 推理时统一 prototype bank 融合

如果后续要继续做“训练级统一”，则需要再改：

- `YOLOEModel.loss()`
- hybrid trainer
- validator
- 训练数据组织与采样策略

这些不在本次修改范围内。

## 6. 注意事项

1. 默认旧类来源
   - 当前默认旧类为 `model.names` 的前 `20` 项。
   - 如果你的权重类别顺序与业务定义不一致，请显式传 `base_names`。

2. 新类名称来源
   - 单图 prompt 时，`visual_prompts["cls"]` 传字符串最稳妥。
   - 如果传数值类 ID，新类名会退化为 `object0/object1/...`。

3. NMS 建议
   - 混合旧类 + 新类时，推荐：
     - `agnostic_nms=False`
   - 也就是使用 class-aware NMS。

4. 当前改造范围
   - 已实现单模型联合推理。
   - 未实现训练阶段的旧类/新类联合损失。

## 7. 本次结论

本次修改已经将 `Par_A1.md` 的方案二落到代码：

- 保留旧类 20 类原型
- 提取新类 visual prompt 原型
- 在 YOLOE 中统一拼接 prototype bank
- 在同一个检测头和共享分数空间内完成联合分类
- 将 NMS 从强制 class-agnostic 改为可配置，并默认回到更适合混合模式的 class-aware NMS
