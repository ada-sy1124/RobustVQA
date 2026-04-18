# 基于 GRPO 的复合奖励机制多模态大模型在电商退货退款场景中的推理对齐优化

## 1. 项目概述

### 1.1 项目目标与任务定义

RobustVQA 旨在通过强化学习（RL）训练一个在多模态问答任务中具备更强准确性、鲁棒性与可解释性的 VQA 系统。项目核心不是只优化最终答案，而是同时优化：
- 最终答案正确性
- 推理过程（CoT）的逻辑自洽性（Self-Consistency）
- 推理过程可验证性（Verifiability）

任务要求模型输入图像与多选题后，严格输出结构化结果：

```text
<think>
...
</think>
<answer>
...
</answer>
```

若输出不符合结构规范，将被格式奖励惩罚。

### 1.2 现状挑战

当前多模态模型在复杂科学问答（如 ScienceQA）上常见问题：
- CoT 可能是事后合理化，并非驱动答案的真实因果链
- 仅用二元准确率奖励（答对 1 / 答错 0）信号过于稀疏
- 难以同时优化答对、逻辑可靠、输出规范

### 1.3 技术路线与创新点

基于 `Qwen2.5-VL-7B-Instruct`，使用 RL 优化如下复合奖励：

$$
R_{Total}=0.7\times R_{Acc}+0.3\times R_{Consistency}+R_{Length}+R_{Format}
$$

关键创新：
- 两阶段一致性校验奖励 `R_Consistency`
- 条件长度激励 `R_Length`（只在答错时激活）

---

## 2. 强化学习奖励机制（重点）

### 2.1 总奖励架构

$$
R_{Total}=0.7\times R_{Acc}+0.3\times R_{Consistency}+R_{Length}+R_{Format}
$$

- `R_Acc`：答案正确性主导项
- `R_Consistency`：逻辑链可验证性约束项
- `R_Length`：深度思考激励项（带豁免）
- `R_Format`：结构输出硬约束项

### 2.2 Acc Reward

- 预测选项等于标准答案：`R_Acc = 1.0`
- 否则：`R_Acc = 0.0`

### 2.3 Consistency Reward（两阶段逻辑校验）

#### 阶段一：初始推理
模型基于完整输入（图像+问题+选项）生成：
- 思维链 `C1`
- 初始答案 `A1`

#### 阶段二：逻辑审计
移除图像输入，仅给 `C1` 与打乱顺序后的选项，要求模型作为逻辑校验器推导答案 `A2`。

目标：检验 `C1` 是否包含足以独立支撑结论的证据链。

#### `R_Consistency` 打分表

| 情景 | A1（多模态） | A2（仅CoT） | 逻辑状态 | R_Consistency |
|---|---|---|---|---|
| I 理想 CoT | 正确 | 正确 | 逻辑自证完整 | 1.0 |
| II 正确但 CoT 缺陷 | 正确 | 错误 | 逻辑不自洽 | 0.5 |
| III 错误但意外自洽 | 错误 | 正确 | 逻辑不自洽 | 0.5 |
| IV 逻辑稳定但错误 | 错误 | 错误且 A1=A2 | 逻辑一致但结论错 | 0.1 |
| V 逻辑混乱且错误 | 错误 | 错误且 A1≠A2 | 逻辑混乱 | 0.0 |

### 2.4 Length Reward（条件激励）

长度阈值：
- `l_min = 50`
- `l_opt = 100`
- `l_max = 220`

$$
R_{Length}(L)=
\begin{cases}
-1 & L<50 \\
-1+\frac{L-50}{100-50} & 50\le L<100 \\
0 & 100\le L\le 220 \\
-1 & L>220
\end{cases}
$$

豁免机制：
- 若 `R_Acc=1.0`，强制 `R_Length=0`
- 只在答错样本上激活长度激励，鼓励更深入探索，100-220的长度是对模型的正确CoT做四分位数分箱统计出来的

### 2.5 Format Reward（硬约束）

- 严格匹配 `<think>...</think><answer>...</answer>`：`R_Format=0.0`
- 否则：`R_Format=-1.0`

实现细节上做两层校验：
- 正则 pattern 校验
- `think` / `answer` 标签唯一性校验（防止多重 answer 漏洞）

---

## 3. 数据集构建与预处理

### 3.1 核心数据源

基于 `derek-thomas/ScienceQA` 进行训练和评估。

核心字段包括：
- `image`
- `question`
- `choices`
- `answer`

### 3.2 筛选规则

- 图像有效性过滤：`image` 非空
- 选项数量过滤：`len(choices) >= 2`

经过筛选后得到高质量可训练样本，用于 RL 的 train/val 划分。

### 3.3 数据脚本

- 数据处理：`data_process/get_data_sqa.py`（兼容入口）
- 主逻辑脚本：`data_process/get_data_spa.py`
- 样本检查：`data_process/查看数据.py`

---

## 4. 训练与部署细节

### 4.1 环境依赖（Python 3.10）

```bash
pip install vllm==0.8.5.post1
pip install qwen-vl-utils
pip install flash-attn==2.7.4.post1
pip install transformers==4.52.4
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0

cd /hy-tmp/RobustVQA/verl
pip install -e .
```

### 4.2 Self-Verifier 服务

```bash
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
  --model /hy-tmp/modelscope_cache/models/Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --trust-remote-code
```

### 4.3 RL 训练（GRPO）

训练脚本：`verl/scripts/run_grpo.sh`

关键配置（示例）：
- 模型：`Qwen3-VL-8B-Instruct`
- `trainer.total_epochs=2`
- `actor_rollout_ref.rollout.n=8`
- `data.train_batch_size=64`
- `actor_rollout_ref.actor.optim.lr=1e-6`
- 多卡训练：`trainer.n_gpus_per_node=4`

实际运行前需按本机路径修改 `TRAIN_FILES` / `VAL_FILES` / `MODEL_PATH` / `OUTPUT_PATH`。

---

## 5. 测评与性能

测评脚本：`eval/run.sh`

测评前先启动 vLLM 服务，再设置：
- `MODEL_SERVER`
- `MODEL_PATH`
- `dataset`

当前流程默认评估 `jsonl` 数据集并计算准确率。

### 5.1 结果示例

| 模型 | 测试集 Acc | 提升 |
|---|---:|---:|
| Qwen3-VL-8B-Instruct | 60.91% | N/A |
| RL-step-194 | 68.03% | +8.12pp |

---

## 6. 仓库结构

| 路径 | 说明 |
|---|---|
| `data_process/` | 数据构造、预处理、样本检查 |
| `eval/` | 推理拉取与指标统计 |
| `verl/` | RL 训练框架与扩展奖励逻辑 |

---

## 7. 致谢与引用

训练基础设施来自 ByteDance Seed 团队开源的 `verl / HybridFlow`。

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```
