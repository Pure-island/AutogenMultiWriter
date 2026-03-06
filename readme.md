# AutoGen 多智能体并行写作流水线

基于 AutoGen v0.4 异步 API 的教程/书籍生成脚本。通过「目录规划 + 章节写作 + 内容评审」三类智能体协作，自动生成结构化 Markdown 文档，并支持并发写作与中断续写。

## 功能概览

- 异步并行生成：使用 `asyncio` 并发处理多个小节
- 多智能体协作：TOC Agent 生成目录，Writer Agent 写正文，Reviewer Agent 给改进建议
- 迭代改进：小节支持评审后再生成，提升可读性与完整度
- 断点续写：目录与章节文件落盘后可复用，避免重复生成
- 重试机制：调用模型时带重试，降低临时失败影响
- 主题输入灵活：支持直接传 `--topic` 字符串或 `.txt` 文件

## 环境要求

- Python 3.9+
- 依赖安装：

```bash
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

## 配置说明

脚本默认通过 SiliconFlow 兼容 OpenAI 接口调用模型（`https://api.siliconflow.cn/v1`）。

建议将 API Key 放到环境变量中，不要硬编码在脚本里。

PowerShell:

```powershell
$env:SILICONFLOW_API_KEY="你的API密钥"
```

macOS/Linux:

```bash
export SILICONFLOW_API_KEY="你的API密钥"
```

然后在代码中读取：

```python
api_key = os.environ.get("SILICONFLOW_API_KEY")
```

当前默认模型（以代码为准）：

- Writer: `deepseek-ai/DeepSeek-V3.2`
- Reviewer: `deepseek-ai/DeepSeek-V3.2`

## 快速开始

```bash
python autogen_multi_writer_parallel.py --topic "你的主题" --audience "目标读者"
```

### 常用参数

- `--topic`：写作主题，或 `.txt` 文件路径
- `--audience`：目标读者
- `--concurrency`：并发 worker 数（默认 8）
- `--max-toc-iter`：目录迭代次数（默认 2）
- `--max-section-iter`：小节迭代次数（默认 1）
- `--notes`：对整篇内容的补充要求（会影响目录与写作）

### 示例

```bash
# 直接指定主题
python autogen_multi_writer_parallel.py \
  --topic "机器学习入门" \
  --audience "初学者"

# 从文件读取主题
python autogen_multi_writer_parallel.py \
  --topic topic.txt \
  --audience "AI 研究人员"

# 控制并发与迭代
python autogen_multi_writer_parallel.py \
  --topic "Python 高级编程" \
  --audience "有经验的开发者" \
  --concurrency 4 \
  --max-toc-iter 3 \
  --max-section-iter 2 \
  --notes "强调工程实践与性能优化"
```

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

## 注意事项

- 模型调用成本与并发度、章节数量、迭代次数强相关
- 并发建议从小值开始，观察 API 限流和稳定性后再提高
- 若中断后重跑同一主题，可复用已生成文件，减少重复开销
