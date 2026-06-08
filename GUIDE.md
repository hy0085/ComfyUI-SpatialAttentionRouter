# ComfyUI-ColorRegion（颜色分区）

用颜色蒙版控制 SDXL 每个区域的 prompt——在内置画板上涂色，红色画龙、绿色画灯、蓝色画背景，互不干扰。

---

## 解决的问题

传统 ComfyUI 工作流中，prompt 全局生效。写 "dragon" 和 "desk lamp"，模型自行决定龙和台灯的位置，极易导致语义泄漏（semantic bleeding）——龙鳞纹理跑到台灯上、房间光线和火焰特效混在一起。

社区分区方案普遍采用硬隔离：

| 方案 | 机制 | 缺陷 |
|---|---|---|
| **Regional Prompter** | 断开区域外 token 的注意力连接 | 非自然边界、构图崩塌 |
| **Attention Couple** | 在 UNet 中切分区域计算注意力 | 缺乏羽化过渡 |

**ColorRegion（颜色分区）** 采用 Output Blending + Region Logit Bias 混合架构——KV 物理隔离切断跨区 token 污染，pre-softmax 空间导向锁定位置，mask 加权混合保留羽化过渡。

---

## 参考技术来源

| 来源 | 类型 | 借鉴 | 改进 |
|---|---|---|---|
| **DenseDiffusion** (NAVER AI, CVPR 2024) | 学术论文 | Cross-Attention Hook 注入模式、SD 版本自动检测 | 二值 mask → Output Blending + Logit Bias |
| **Attention Couple** (laksjdjf) | 社区项目 | Output Blending 算法框架、`model_patcher` 接口 | KV 隔离 + Logit Bias 双重锁定 |
| **Regional Prompter** | 社区项目 | 颜色蒙版 → 区域分区交互直觉 | 硬阻断 → 物理隔离 + 羽化过渡 |

三者的关系：

```
Regional Prompter 的交互直觉（颜色蒙版分区）
    +
Attention Couple 的计算框架（Output Blending）
    +
DenseDiffusion 的注入机制（Cross-Attention Hook）
    =
ColorRegion / 颜色分区（KV 隔离 + Logit Bias + Output Blending 三重锁定）
```

---

## 核心技术原理（v2.1-beta）

### 架构：Output Blending + Region Logit Bias

```
v1.0 (Bias Injection):                     v2.1 (Output Blending):
[全局, 区域1, 区域2] × bias → softmax      attn(全局 + 区域1) × mask₁
                                           + attn(全局 + 区域2) × mask₂
❌ 全局 token bias=0 在所有位置"合法逃课"    + attn(全局) × mask_bg
❌ CFG uncond 也被污染                      ────────────────────────
                                            ✅ 算台灯时 K/V 里不存在 1girl
                                            ✅ uncond 走全注意力保 CFG 结构
                                            ✅ pre-softmax Logit Bias 强化空间锁定
```

### KV 隔离（核心机制）

对每个区域独立拼接 `[base_tokens + region_tokens]` 计算注意力，再乘 mask 加权混合。区域 B 的 K 和 V 矩阵里**物理上不存在**区域 A 的 token。

```
传统 Cross-Attention（所有 token 全局可见）：
  Attention = softmax(Q · [K_global | K_region1 | K_region2]ᵀ / √d) · V
  → 区域1 的 token 可以影响区域2 的空间位置

ColorRegion（KV 物理隔离）：
  对每个区域 i 独立计算：
    K_i = [K_global | K_region_i]
    V_i = [V_global | V_region_i]
    output_i = softmax(Q · K_iᵀ / √d) · V_i

  最终输出 = Σ output_i × mask_i + output_bg × mask_bg
            ─────────────────────────────────────────────
                       Σ mask_i + mask_bg
```

### Region Logit Bias

在 KV 隔离的基础上，对区域专属 token 施加 pre-softmax 空间偏置：

```
steering_bias = -strength × (1 - mask)²  （二次方非线性衰减）
```

选用二次方而非线性衰减的原因：线性 `-(1-m)` 在羽化过渡区施加的惩罚过大，导致视觉接缝；二次方 `-(1-m)²` 让羽化区几乎不受影响，仅在远离核心区时强力介入。

| mask 值 | 空间位置 | (1 - mask)² | bias（strength=15） | 效果 |
|---|---|---|---|---|
| 1.0 | 区域核心 | 0 | 0 | 注意力不受影响 |
| 0.8 | 羽化过渡区 | 0.04 | -0.6 | 极轻微引导，保护柔边 |
| 0.0 | 区域外 | 1.0 | -15.0 | 最大隔离 |

### CFG 安全隔离

Classifier-Free Guidance 需要同时计算 cond 和 uncond 的注意力。为确保 CFG 基准结构不被破坏：

- **uncond（负面 prompt）**：走标准全注意力，不受路由影响
- **cond（正面 prompt）**：走 KV 隔离 + Logit Bias 路由

通过 `extra_options["cond_or_uncond"]` 判断 batch 成员身份，逐 batch 分支处理。

---

## 项目结构

```
ComfyUI-ColorRegion/
├── __init__.py                  # 插件入口 + WEB_DIRECTORY
├── nodes.py                     # Layer 5: ComfyUI 节点（3 个）
├── region_parser.py             # Layer 1: 颜色蒙版 → {color: mask[H,W]}
├── affinity_parser.py           # Layer 2: region prompt CLIP 编码 + token 映射
├── cross_attention.py           # Layer 3: Output Blending + Logit Bias 核心
├── attention_router.py          # Layer 4: SD 版本检测 + Hook 注入
├── web/js/spatial_router_ui.js  # Frontend: Canvas 画板 + 颜色选择 UI
├── test_core.py                 # 独立单元测试（10 项）
└── requirements.txt
```

| 层 | 文件 | 职责 |
|---|---|---|
| Layer 1 | `region_parser.py` | 从 RGB 颜色蒙版提取唯一颜色 → 逐色生成二值 mask → 高斯羽化 |
| Layer 2 | `affinity_parser.py` | 每区域独立 CLIP 编码 → token 位置映射 → 按 [global \| region_1 \| region_2 \| ...] 拼接 |
| Layer 3 | `cross_attention.py` | KV 隔离 + Logit Bias + Output Blending 前向传播；Mask Pyramid Cache |
| Layer 4 | `attention_router.py` | SD 版本检测 → UNet Transformer Block 枚举 → `set_model_attn2_replace` 注入 |
| Layer 5 | `nodes.py` | 3 个 ComfyUI 节点封装；内置 Canvas 画板 |

---

## 安装

```bash
cp -r ComfyUI-ColorRegion /path/to/ComfyUI/custom_nodes/
# 重启 ComfyUI
```

兼容性：SDXL（主要） / SD1.x / SD2.x / 🔮 Flux（未来）

---

## 使用方式

### 方式一：内置画板（推荐，无需外部图片）

直接在节点上的黑色画板中涂色——选中一行区域（radio），画笔自动切到该颜色：

```
1. 拖出 ColorRegion 节点
2. 在画板上用不同颜色涂出各区域
3. 每行填好 prompt 描述
4. 连线：model + clip + conditioning → KSampler
```

工作流连线：

```
Load Checkpoint ── MODEL ──┐
                  ── CLIP ──┤
                            ├── ColorRegion ── MODEL ──┐
                            │       ↑                  │
                            │  内置 Canvas 画板         ├── KSampler ── VAE Decode
                            │  (直接涂色，无需外部图)    │
                            │       conditioning ──────┘
```

### 方式二：外部蒙版图片（兼容旧工作流）

连接 `mask_image`——画板为空时自动回退到外部图片。

### Prompt 规范

**全局提示词（Global Prompt）：只放画质和风格**——严禁写任何具体物体或场景。

```
✅ global_prompt: masterpiece, best quality, ultra detailed, anime style
❌ global_prompt: dragon, cyberpunk street, room interior
```

**区域提示词（Region）：承接所有具体描述**。

```
#ff0000: giant red dragon, detailed scales, fire breath
#00ff00: tiny antique desk lamp, brass, warm glow
#0000ff: dimly lit room, large window, night sky
```

铁律：Base = 画质/风格词，Region = 场景/物体/光线。两者职责分离，绝不重叠。

**BREAK 语法**：在区域 prompt 中用 `BREAK` 创建 CLIP 编码边界，降低同一区域内属性间的 token 污染：

```
#ff0000: 1girl BREAK red dress BREAK white socks BREAK blue gloves
```

每个 BREAK 段落独立 CLIP 编码后拼接，`1girl` / `red dress` / `white socks` 在 embedding 空间互不可见。

### 参数

| 参数 | 作用 | 推荐值 |
|---|---|---|
| `strength` | Logit Bias 倍增因子（二次衰减公式） | **12~15** 日常，**18~25** 极限隔离 |
| `feather_px` | 羽化半径（像素） | **20~40**（默认 30） |

调节思路：初始设 strength=15, CFG=5 → 串色则升 strength(18→20)、降 CFG(4.5) → 爆色块则降 strength 或 CFG(4.0) → 接缝生硬则升 feather_px(40)。

### 节点

| 节点 | 功能 |
|---|---|
| **ColorRegion** | 全能节点：内置画板 + mask 解析 + prompt 编码 + 模型注入 |
| **ColorRegionAdvanced** | 接收已编码 conditioning，不包含画板 UI |
| **ColorMaskPreview** | 调试用：可视化检测到的颜色区域 |

---

## 调试

### 日志输出

插件通过 `print()` 和 `_debug_router.log` 两种渠道输出诊断信息。

**终端输出**（`print`，跳过日志级别，始终可见）：

| 标记 | 内容 |
|---|---|
| `[AFFINITY DEBUG] ROUTER HIT CHECK` | 每区域 token 范围 + mask 覆盖像素数 |
| `[ROUTER] region=#ff0000 tokens=77:154 mask_pixels=16384` | 确认路由注册成功 |
| `[AFFINITY DEBUG] FINAL OUTPUT` | 最终 concat_cond + pooled 的 shape/dtype |
| `[AFFINITY DEBUG] FALLBACK RETURN CHECK` | 走 fallback 时打印 base_conditioning 详情 |

**文件日志**（`_debug_router.log`）：

| 标记 | 内容 |
|---|---|
| `[AFFINITY] BREAK prompt` / `merged_cond shape` / `pooled shape` | BREAK 编码全链路 |
| `SpatialAttentionBias INIT` | 区域颜色、token 范围、mask 形状 |
| `INIT V2.1-BETA DIAGNOSTICS` | CFG Array、Batch Size、Extra Options Keys |

### 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| 某个区域内容消失 | 全局 prompt 与区域描述语义冲突 | 删除全局中 `simple`、`minimal`、`dark` 等方向性形容词 |
| 主体位置不对 | SDXL 居中构图先验 | 拉高 strength 到 15-20 |
| 颜色匹配不上 | 截图取色导致 `#ff0000` → `#fa0205` | 模糊匹配已内置（曼哈顿距离 ≤45），检查控制台 `Fuzzy matched` 日志 |
| 图片崩了 | strength 太高 | 降低 strength，增大 feather_px |
| 节点 UI 异常 | 旧节点缓存了脏数据 | Ctrl+F5 刷新 → 删除旧节点 → 拖新节点 |

---

## 运行测试

```bash
python test_core.py   # 10 项测试，不依赖 ComfyUI
```

---

## 许可证

MIT License

---

## 致谢

- [DenseDiffusion](https://github.com/naver-ai/DenseDiffusion) (CVPR 2024) — Attention Modulation 机制
- [Attention Couple](https://github.com/laksjdjf/Attention-Couple) — Output Blending 算法框架
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — 模块化 AI 图像生成框架
