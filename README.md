# 图像增强对多模态模型物体识别的影响

本项目把一个图片或图片集分别送入 25 个实验条件：预处理原图基线，以及 8 种增强方法各 3 档强度。每次模型调用只接收同一条件下的图片集，绝不混合不同方法或强度。模型在 `config/config.yaml` 中三选一：Qwen3.5、LLaVA、或同类 Hugging Face 多模态模型。

## 架构

```text
JSONL 样本清单
  → EXIF 方向修正、可选旋转、等比例缩放与补边、可选锐化
  → 原图 + 8 种方法 × Low/Med/High → img/enhance/<run-id>/
  → 按实验条件组装图片集 + 问题 + 系统提示词
  → Hugging Face Processor 编码 → GPU 模型生成 token → 解码、JSON 校验
  → runs/<run-id>/predictions.jsonl → results/<run-id>_metrics.json
```

核心代码：`mama/imaging.py`（预处理）、`mama/enhancements.py`（增强）、`mama/model.py`（提示词与推理）、`mama/pipeline.py`（编排）、`mama/evaluation.py`（指标）。

增强方法为 `gamma`、`dcp`、`zero_dce`、`retinex`、`clahe`、`white_balance`、`waternet`、`reserved_identity`。`zero_dce` 是每张图即时优化的七层零参考曲线网络，不依赖额外权重；它是实验性实现，增加迭代次数可改善拟合但会增加耗时。`waternet` 使用仓库内已有权重；Low/Med/High 通过原图与完整增强结果按固定比例混合。`reserved_identity` 是有明确标签的空槽位，三档与预处理原图相同，用于后续替换，不应当当成有效增强。白平衡采用灰度世界算法，可校正整体偏 R/G/B；对于局部偏色效果有限。

## 数据格式

准备 JSONL 文件，每行一个样本。图片路径相对清单文件所在目录；每个样本的所有图片针对同一个目标和问题。`truth` 可省略；省略时只生成预测，不计算该样本的指标。

```json
{"id":"sample-001","images":["images/a.jpg","images/b.jpg"],"target":"救生圈","question":"每张图中是否有救生圈？分别指出位置和数量。","truth":[{"count":1},{"count":0}]}
```

`truth` 按输入图片顺序给出实例数量；大于零视为存在。每个条件的输出形如 `{"images":[{"image_index":0,"present":true,"count":1,"locations":["左上"]}]}`。解析器检查顺序、类型、数量与位置个数的一致性。无效回答记为 `invalid`，计入错误率和准确率分母；混淆矩阵的 `tn/fp/fn/tp` 只包含有效回答。准确率、错误率、精确率、召回率、F1、有效回答率、计数完全正确率及计数 MAE 均按条件单独统计。位置是语言描述，目前没有位置真值指标。

## 运行
1. 请在项目中完成以下指令：
    ```bash
    mkdir ./enhance
    cd enhance
    git clone https://github.com/tnwei/waternet.git --depth 1 
    ```
2. 激活环境：`source ~/miniconda3/etc/profile.d/conda.sh && conda activate mama`。
3. 在 `config/config.yaml` 配置 `model.repo_id` 与 `model.path`，运行 `python -m mama download` 将完整模型仓库下载到 `models/`，包括权重、config、tokenizer 和 processor 文件。也可使用 Hugging Face CLI：

   ```bash
   hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
   # 或：hf download llava-hf/llava-1.5-7b-hf --local-dir models/llava-1.5-7b-hf
   ```

4. 在 `config/config.yaml` 选择 `model.type` 和对应 `model.path`。未知模型选 `custom`，它必须兼容 `AutoProcessor` 和 `AutoModelForImageTextToText` 的聊天模板与图像输入。默认使用 CUDA 和 BF16；显卡不支持 BF16 时选 `float16`。CPU 仅用于小规模调试，设 `device: cpu`、`dtype: float32`。
5. 运行：

   ```bash
   python -m mama all --manifest dataset/samples.jsonl
   # 可拆成阶段：
   python -m mama prepare --manifest dataset/samples.jsonl --run-id trial1
   python -m mama infer --prepared runs/trial1/prepared.jsonl --run-id trial1
   python -m mama evaluate --predictions runs/trial1/predictions.jsonl --run-id trial1
   ```

`prepare` 无需下载大模型。每张输入图保存一份预处理原图与 24 份增强图，文件名包含 UUID、方法和强度。经典图像方法并行执行；WaterNet 和零参考网络顺序执行以控制显存。推理逐条件运行，显存占用主要由所选模型与单个图片集决定；较大的图片集可能超出上下文长度或显存，请缩小预处理尺寸或拆分样本。只比较同一批样本、同一问题和同一模型下的条件指标。

## 注意

- 模型生成的回答是 token 张量经解码后的文本，不是直接输出 JSON 张量；原始文本保存在预测文件中，格式错误会被记录。
- `ImageOps.pad` 统一画布尺寸并保留完整内容，补边可能影响识别；固定锐化参数可统一处理流程，但无法保证严重失焦图像具有相同实际清晰度。不要把某一方法的图片偷偷换入另一条件。
- `zero_dce` 即时拟合的结果存在随机性；正式对比可固定随机种子或提供训练好的 Zero-DCE 权重后替换该实现。
- `models/`、`img/enhance/`、`runs/` 和 `results/` 可能很大，应按实验需要管理磁盘空间。

模型接口参考：[Qwen3.5 Transformers 文档](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)、[LLaVA Transformers 文档](https://huggingface.co/docs/transformers/model_doc/llava)。
