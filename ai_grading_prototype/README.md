# AI Grading Prototype

这是一个“类似 `llm_assisted_grading` 的最小原型骨架”。

目标不是一次做成完整系统，而是先把链路拆清楚：

1. 读题目与标准答案
2. 读取学生答题卡/手写图像
3. 生成评分请求
4. 汇总评分结果
5. 输出人工复核 PDF

## 这个原型最少需要什么

### 必需输入

- `questions.xlsx`：题目、标准答案、评分要点
- `raw/`：学生答题卡扫描图
- `config.py`：题号、ROI、输出目录、模型参数
- `OPENAI_API_KEY`：如果要接大模型评分

### 必需模块

- `preprocess`：裁切、对齐、识别学生信息
- `question_bank`：读取题库
- `batch_builder`：把图片和题目拼成评分请求
- `result_aggregator`：合并多次评分结果
- `pdf_report`：生成人工检查 PDF
- `manifest`：记录每次运行产物

### 先不做也能跑的部分

- 自动 OMR 识别做得很完美
- 批量并发优化
- 复杂的网页前端
- 多学科通用化

## 推荐目录

```text
ai_grading_prototype/
  README.md
  requirements.txt
  .env.example
  src/
    config.py
    question_bank.py
    preprocess.py
    batch_builder.py
    result_aggregator.py
    pdf_report.py
    pipeline.py
  data/
    raw/
    processed/
    batches/
    results/
    manifests/
```

## MVP 流程

1. 手工放入一小批答题卡图片
2. 从 Excel 读取 1~2 道题的标准答案
3. 对每张图裁切出答题区域
4. 调用模型评分
5. 导出 CSV + PDF

## 你现在真正要准备的东西

- 一份题库表
- 一批扫描图片
- 统一的题目编号规则
- 评分标准
- OpenAI API Key

## 后续可扩展

- 自动识别学生编号
- 自动识别题号/版本
- 批量异步评分
- 评分波动分析
- 复核标记
- 评语生成

