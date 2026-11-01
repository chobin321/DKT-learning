# DKT Learning

本项目记录了基于 `pyKT-toolkit` 复现 Deep Knowledge Tracing (DKT) 模型的完整过程。

## 运行方式
1. 准备数据：`python convert_data.py`
2. 启动训练：
   ```bash
   cd pykt-toolkit/examples
   python wandb_dkt_train.py --use_wandb=0 --dataset_name=assist2009
