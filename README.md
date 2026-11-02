# Deep Knowledge Tracing (DKT) 跑通与复现

本项目基于 `pyKT-toolkit` 框架，完整复现了经典的知识追踪模型 **DKT (Deep Knowledge Tracing)**。项目记录了从零开始的数据工程、环境适配以及真实的基准评估。

## 项目亮点 (Highlights)

*   **真实数据源**：采用官方包含完整 `question_id` 与 `skill_id` 的 ASSISTments2009 原始数据集，杜绝了数据泄漏。
*   **严谨的时间线切分**：使用 pyKT 官方预处理脚本，严格按照时间线（前 80% 训练，后 20% 测试）切分数据，真实反映模型的预测能力。
*   **纯 CPU 环境适配**：魔改了框架源码中的设备指定逻辑，彻底解决无 GPU 环境下的 `CUDA Tensor` 报错。
*   **全流程自动化**：包含日志记录、正则解析与训练曲线可视化脚本。

## 实验结果 (Results)

在官方完整的 ASSISTments2009 数据集上，模型无法再通过“背知识点”作弊，跑出了符合学术界基准的真实水平：

*   **Best Epoch**: 13
*   **Valid AUC**: 0.8955
*   **Valid ACC**: 0.8301

### 训练曲线
![Training Curves](./assets/training_curves.png)

## 📁 目录结构

```text
DKT/
├── assets/                 # 存放 README 图片
├── pykt-toolkit/           # 核心框架代码 (包含 CPU 适配修改)
│   ├── configs/            # 模型与数据维度配置文件
│   ├── data/assist2009/    # 预处理后的训练与测试序列
│   ├── examples/           # 运行脚本 (含 plot_results.py)
│   └── pykt/               # 框架底层源码
├── .gitignore              # 忽略大体积数据与缓存
└── README.md