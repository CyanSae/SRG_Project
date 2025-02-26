# Tips
- Ubuntu 24.04
- ```pip install -r requirements.txt ```
- dgl和torch是CPU版本，用GPU的话可以另外安装
- 代码重新整理过，比较仓促，有些地方（比如路径）可能比较乱
- 100M以上的文件（比如pkl）不要直接传GitHub
# RGCN
RGCN/shuffled_dataset/creation_1346_shuffled.csv 目前的数据集
RGCN/my_rgcn/process_all.py 生成用于训练的数据集pkl
RGCN/my_rgcn/rgcn.py RGCN模型训练
# SRG
SOG/src/sog/data_process.py 批量生成SRG
SOG/src/sog/sog_builder.py SRG构建主要代码
# DATA
DATA/SOG_SET 生成的SRGs
