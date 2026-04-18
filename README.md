# 电商退换货场景下的多模态推理对齐

本仓库围绕**电商客服退换货**场景，用强化学习对齐多模态大模型（MLLM）的推理行为：抑制**迎合用户情绪、脱离事实的「迎合性伪推理」**，缓解奖励稀疏，并提升逻辑自洽与可解释性。基座模型为 **Qwen3-VL-8B-Instruct**；训练侧基于开源 RL 框架 **[verl](https://github.com/volcengine/verl)**（HybridFlow）扩展。

---

## 项目背景

- **问题**：在退换货、退款等高风险对话里，多模态大模型容易被用户情绪带偏，产生缺乏事实与视觉依据的伪推理。
- **目标**：构建面向该场景的 RL 推理对齐机制，降低奖励稀疏性，并让模型在跨模态信息整合上更稳健、可追溯。

---

## 核心贡献

### 数据管线

- 以 **Amazon Review Data** 为基础，利用大模型合成**对抗性客诉**样本。
- 设计**自动化退款相关标注**流程，沉淀可复用的电商风控向多模态数据构造管线。

### 奖励设计

- 设计**复合奖励**，将准确率、逻辑稳健性、格式遵循等多维目标解耦，减轻奖励稀疏。
- 构建**两阶段逻辑校验**框架，拦截缺乏图像证据支撑的伪推理，为策略提供更**稠密**的过程奖励信号。

### 策略与训练设计

- 引入**条件化长度激励**与**惩罚豁免**等策略，平衡探索效率。
- 引导 VLM 进行更长的链式推理与跨模态信息融合，以提升推理结论的可靠性。

---

## 仓库结构（与本项目直接相关部分）

| 路径 | 说明 |
|------|------|
| `data_process/` | 数据构造与预处理脚本（多模态样本、提示模板等） |
| `eval/` | 模型回复拉取与测评相关脚本 |
| `verl/` | RL 训练库本体；与本课题相关的奖励等逻辑见 `verl/verl/workers/reward_manager/utils/rewards.py` 等扩展点 |

更完整的安装、分布式训练与算法说明请参考上游文档：[verl 文档](https://verl.readthedocs.io/en/latest/index.html)。

---

## 致谢与引用

强化学习训练基础设施来自 ByteDance Seed 团队开源的 **verl** / **HybridFlow**。若使用 verl 进行研究，建议引用：

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

论文链接：[HybridFlow](https://arxiv.org/abs/2409.19256v2)。
