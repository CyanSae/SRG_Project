# 恶意合约检测原型系统

该原型系统位于 `Prototype` 目录下，满足以下流程：

1. 用户输入合约地址或创建字节码。
2. 若输入的是地址，后端通过 Etherscan API 抓取创建交易与创建字节码。
3. 后端调用 `SOG/src/sog/sog_builder_connect.py` 与 `SOG/src/sog/data_process_connection.py` 生成 SRG JSON。
4. 后端调用训练好的层次化 RGCN 模型 `RGCN/model/trained_model/hie/best_model_hie_rgcn_2426_393-400.pt` 进行推理。
5. 前端展示二分类、大类分类、具体类型分类结果，并支持下载 SRG JSON。
6. 系统支持对生成的 SRG JSON 进行图谱可视化展示。

## 功能说明

- 支持单条检测与批量检测。
- 支持两种输入来源：合约地址、创建字节码。
- 单个合约检测模式下不显示检测粒度选择，并动态隐藏全部批量输入框。
- 单个合约检测时，若选择“直接输入字节码”，页面会动态隐藏单个合约地址输入栏。
- 批量合约检测模式下显示检测粒度选择，并动态隐藏单个合约地址输入框和单个合约字节码输入框。
- 批量模式下可直接在文本框中逐行输入，也可通过文件逐行导入。
- 批量模式支持上传 `csv` 文件，表头格式为 `contract_address,bytecode`。
- 单条与批量结果均支持打开 SRG 可视化页面查看图结构。

## 启动方式

建议在项目根目录执行：

```bash
python Prototype/run.py
```

启动后访问：

```text
http://127.0.0.1:8000
```

## 环境变量

- `ETHERSCAN_API_KEY`：地址抓取模式所需的 Etherscan API Key。当前原型已内置默认 Key，也可用环境变量覆盖。
- `PROTOTYPE_HOST`：服务监听地址，默认 `127.0.0.1`。
- `PROTOTYPE_PORT`：服务端口，默认 `8000`。
- `SOG_PYTHON`：用于执行 SRG 构建脚本的 Python，默认 `python`。
- `RGCN_PYTHON`：用于执行模型推理脚本的 Python，默认 `/home/sandra/anaconda3/envs/rgcn/bin/python`。
- `RGCN_MODEL_PATH`：模型权重路径。

## 目录说明

- `backend/`：后端 API、配置与服务封装。
- `frontend/`：原生前端页面与静态资源。
- `scripts/`：SOG 构建与 RGCN 推理独立脚本。
- `runtime/`：运行时任务输出目录。
