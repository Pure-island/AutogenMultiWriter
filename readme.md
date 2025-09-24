AutoGen 大模型教程/书籍写作
===================

概述
--

一个基于 AutoGen v0.4 异步 API 的多智能体写作流水线脚本，使用多个 AI 智能体协作生成结构化技术文档，包括生成目录、撰写章节内容以及进行内容评审优化。

## 功能特点

----

* **异步并行处理**​：使用 asyncio 实现高效并行处理，可同时生成多个章节内容

* **多智能体协作**​：使用三个专门的智能体分别负责目录生成、内容撰写和内容评审

* **迭代改进机制**​：每个生成的内容都会经过评审和改进循环，确保质量

* **断点续写功能**​：自动生成的目录和章节内容会保存到文件，避免重复生成

* **重试机制**​：在内容生成和评审过程中加入重试逻辑，提高稳定性

* **支持从文件读取主题**​：可以指定文本文件作为输入主题

环境要求
----

### Python 版本

* Python 3.7+

### 依赖包

运行前请安装以下依赖：
    pip install -U "autogen-agentchat" "autogen-ext[openai]"

配置说明
----

### API 密钥设置

脚本默认使用 [SiliconFlow 平台](https://api.siliconflow.cn/v1)提供的模型服务，可按需求更改url使用其他平台的服务。

需要设置 API 密钥：

1. 直接在脚本中修改(45~70行)：
   
   ```python
         llm_client_writer = OpenAIChatCompletionClient(
   
          model='deepseek-ai/DeepSeek-V3.1', 
          api_key='你的API密钥', 
          # ...
   
          )
   ```

2. 推荐使用环境变量方式（更安全）：
      export SILICONFLOW_API_KEY="你的API密钥"
   然后在脚本中使用：
      api_key=os.environ.get("SILICONFLOW_API_KEY")

### 模型配置

脚本默认使用以下模型：

* 目录生成和内容撰写：DeepSeek-V3.1

* 内容评审：Kimi-K2-Instruct

如需更换模型，请修改相应的 `OpenAIChatCompletionClient`配置。

----

## 基本用法

    python autogen_multi_writer_parallel.py --topic "你的主题" --audience "目标读者"

### 参数说明

* `--topic`: 文章主题（默认："如何用 AutoGen 搭建多智能体写作流水线（v0.4 异步版）"）

* `--audience`: 目标读者（默认："熟悉 Python 的工程师"）

* `--concurrency`: 并行工作者数量（默认：32，根据机器性能和 API 限制调整）

### 示例

    # 生成关于机器学习的教程
    python autogen_multi_writer_parallel.py --topic "机器学习入门" --audience "初学者"
    
    # 使用更高的并发度
    python autogen_multi_writer_parallel.py --topic "Python高级编程" --audience "有经验的开发者" --concurrency 64

## 高级使用示例

### 从文件读取主题

可以将复杂主题写入文本文件，然后传递给脚本：

    echo "深入理解深度学习中的注意力机制及其在自然语言处理中的应用" > topic.txt
    python autogen_multi_writer_parallel.py --topic topic.txt --audience "AI研究人员"

### 调整并发数量

根据你的系统性能和API限制调整并发数：

    python autogen_multi_writer_parallel.py --topic "Web开发最佳实践" --audience "前端工程师" --concurrency 8

### 断点续写功能

脚本会自动保存已生成的内容，如果中断后重新运行相同主题，会自动跳过已存在的章节：

    python autogen_multi_writer_parallel.py --topic "数据科学实战指南" --audience "数据分析师"

## 输出文件格式

脚本运行后，生成的内容将保存在 `output_async` 目录中：

### 目录文件
    00_toc_{主题slug}.json - 教程目录结构

### 章节文件
    {章节编号}_{小节编号}_{章节标题slug}_{小节标题slug}.md

例如：
    01_02_introduction_what-is-autogen.md

### 文件结构示例
