# 多智能体并行写作流水线

这个仓库围绕同一个核心目标展开：为一个主题自动生成成体系的教程或书籍内容。

核心流程是：

1. 先规划目录（TOC）
2. 再按 chapter / section 写正文
3. 再通过 reviewer 迭代改稿
4. 最终落盘为结构化 Markdown

当前仓库包含三条路线：

1. `langgraph_multi_writer.py`
基于 LangChain + LangGraph 的当前主改造路线。

2. `autogen_multi_writer_parallel.py`
基于 AutoGen 的旧实现，仍保留用于对照和回退。

3. `skills/write-course-book/`
开发中的 skill 方案，目前还不好用，不建议当成稳定入口。

## 环境要求

- Python 3.9+
- 推荐使用独立虚拟环境或 conda 环境

## 路线一：LangChain / LangGraph

文件：`langgraph_multi_writer.py`

### 功能

- 使用 `StateGraph` 编排 TOC 和 section 写作流程
- writer 使用 LangChain 调用 OpenAI-compatible 接口
- section 级别支持 `asyncio` 并发
- 支持流式接收
- 流式中断后会基于已收到的 partial text 尝试续写
- TOC JSON 兼容两种返回格式：
  - 纯 JSON
  - fenced JSON，例如 ```` ```json ... ``` ````
- 支持读取已有 TOC，跳过目录生成后直接写作
- 支持复用已生成 section 文件

### 依赖安装

```bash
pip install -U langchain-openai langgraph
```

### 配置方式

使用 OpenAI-compatible 配置即可，不要求绑定某个具体服务商。

LangGraph 路线和 AutoGen 路线现在都按同一套 `WRITER_* / REVIEWER_*` 环境变量结构读取配置。

常用环境变量：

- `WRITER_BASE_URL`
- `WRITER_API_KEY`
- `WRITER_MODEL`
- `REVIEWER_BASE_URL`
- `REVIEWER_API_KEY`
- `REVIEWER_MODEL`

也可以在项目根目录放 `.env` 文件。

示例：

```dotenv
WRITER_BASE_URL=https://your-openai-compatible-base-url/v1
WRITER_API_KEY=your-writer-key
WRITER_MODEL=your-writer-model

REVIEWER_BASE_URL=https://your-openai-compatible-base-url/v1
REVIEWER_API_KEY=your-reviewer-key
REVIEWER_MODEL=your-reviewer-model
```

### 用法

基础运行：

```bash
python langgraph_multi_writer.py --topic "你的主题" --audience "目标读者"
```

常用参数：

- `--topic`：写作主题，或 `.txt` 文件路径
- `--audience`：目标读者
- `--concurrency`：并发 worker 数
- `--max-toc-iter`：目录迭代次数
- `--max-section-iter`：小节迭代次数
- `--notes`：对整篇内容的补充要求
- `--toc-file`：读取已有 TOC JSON，跳过目录生成，直接进入正文写作

示例：

```bash
# 直接指定主题
python langgraph_multi_writer.py \
  --topic "机器学习入门" \
  --audience "初学者"

# 从文件读取主题
python langgraph_multi_writer.py \
  --topic topic.txt \
  --audience "AI 研究人员"

# 控制并发与迭代
python langgraph_multi_writer.py \
  --topic "Python 高级编程" \
  --audience "有经验的开发者" \
  --concurrency 4 \
  --max-toc-iter 3 \
  --max-section-iter 2 \
  --notes "强调工程实践与性能优化"

# 使用已有 TOC，跳过目录生成
python langgraph_multi_writer.py \
  --topic "Python 高级编程" \
  --audience "有经验的开发者" \
  --toc-file "output_async/00_toc_python-高级编程.json"
```

### 当前状态

- 这是当前主改造路线
- 已经能跑基础流程
- 仍建议先从小主题、小并发验证 provider 稳定性
- 当前 section prompt 倾向生成更长内容

## 路线二：AutoGen

文件：`autogen_multi_writer_parallel.py`

### 功能

- 基于 AutoGen v0.4 的多 agent 写作流程
- 目录生成、正文写作、内容评审分工明确
- 使用 `asyncio` 并发写 section
- 支持目录和章节文件落盘后复用

### 依赖安装

```bash
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

### 用法

```bash
python autogen_multi_writer_parallel.py --topic "你的主题" --audience "目标读者"
```

示例：

```bash
python autogen_multi_writer_parallel.py \
  --topic "Python 高级编程" \
  --audience "有经验的开发者"
```

### 当前状态

- 这是旧实现
- 仍可作为对照和回退路线
- 如果你在比较调用方式或回归旧逻辑，可以继续使用它

## 路线三：Skill（开发中）

目录：`skills/write-course-book/`

### 目标

把“生成整本教程”抽象成一套 skill：

- 先提问确认需求
- 再规划 TOC
- 再驱动 section 写作与改稿
- 最终形成可续跑的工作流

### 当前状态

- 还在开发中
- 目前还不好用
- 不建议把它当成稳定生产入口
- 更适合继续迭代设计，而不是直接依赖

## 输出结构

生成结果默认写入 `output_async/`：

- `00_toc_{topic_slug}.json`：目录 JSON
- `{chapter_idx}_{chapter_slug}/{chapter_idx}_{section_idx}_{section_slug}.md`：章节小节 Markdown

示例：

```text
output_async/
  00_toc_机器学习入门.json
  01_基础概念/
    01_01_什么是机器学习.md
    01_02_监督学习与无监督学习.md
```

当使用 `--toc-file` 时，外部 TOC 也会复制保存到当前 `output_async/00_toc_*.json`。

## 注意事项

- 模型调用成本与并发度、章节数量、迭代次数强相关
- 并发建议从小值开始，观察 provider 稳定性后再提高
- 若中断后重跑同一主题，可复用已生成文件，减少重复开销
- 如果 provider 对长文本或流式请求不稳定，优先先做最小主题验证
