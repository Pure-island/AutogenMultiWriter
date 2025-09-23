# autogen_multi_writer_v0_4_async.py
# AutoGen v0.4 异步 & 并行版本：多智能体写作流水线（TOC、Writer、Reviewer）
# 说明：采用 v0.4 的异步 AgentChat API（AssistantAgent.run 为 async），按章节并发生成小节正文并与 Reviewer 迭代。
# 运行前：安装并使用 v0.4 相关包：
#   pip install -U "autogen-agentchat" "autogen-ext[openai]"
# 并设置环境变量（或在下面直接填入）例如：OPENAI_API_KEY

import asyncio
import json
import os
import re
import pathlib
from typing import Dict, List, Tuple

# AutoGen v0.4 imports
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ---------------- 配置 ----------------
# OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OUT_DIR = pathlib.Path(__file__).parent / "output_async"
OUT_DIR.mkdir(exist_ok=True)
MAX_TOC_ITER = 3
MAX_SECTION_ITER = 3
CONCURRENCY = 32  # 并行工作者数量（根据机器/速率限制调整）

# ---------- 工具函数 ----------

def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s[:120]


def save_section_md(chapter_idx:int, section_idx:int, chapter_title:str, section_title:str, content_md:str) -> str:
    filename = f"{chapter_idx:02d}_{section_idx:02d}_{slugify(chapter_title)}_{slugify(section_title)}.md"
    path = OUT_DIR / filename
    # 同步写入（文件小，影响可忽略）。如需严格异步可改用 aiofiles。
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {chapter_title}\n\n## {section_title}\n\n")
        f.write(content_md)
    return str(path)

# ---------- 创建模型客户端与 agents ----------
llm_client_writer = OpenAIChatCompletionClient(
    model='deepseek-ai/DeepSeek-V3.1', 
    api_key='xxx', 
    base_url='https://api.siliconflow.cn/v1',
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
        "structured_output": True,
        "thinking_budget": 25536,
        "max_tokens": 160000 - 25536
    },
    )
llm_client_viewer = OpenAIChatCompletionClient(
    model='moonshotai/Kimi-K2-Instruct-0905', 
    api_key='xxx', 
    base_url='https://api.siliconflow.cn/v1',
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
        "structured_output": True,
        "thinking_budget": 25536,
        "max_tokens": 160000 - 25536
    },
    )


# system prompts（可按需微调）
toc_sys = (    
    "你是专业的教材大纲与课程设计专家。"
    "你是教材大纲与章节内容设计专家。接收主题和目标读者后，返回一个分章分节的目录。"
    "必须返回严格 JSON，格式如下：{\n  \"title\": \"<整篇标题>\",\n  \"chapters\": [ {\"title\":\"..\", \"sections\": [ {\"title\":\"..\", \"desc\":\"一句话概述\"} ] } ]\n}"
    "不要返回额外说明文本。不需要```json等标记。"
    "根据内容的复杂度，生成 3 ~ 15 章，每章 2 ~ 10 节。"
    "标题应当简明且具有概括性\n"
    "小节描述需准确反映核心内容\n"
    "重点是相关的知识覆盖要全面且有深度、成体系\n"
    "对于给定的主题，请生成一个清晰、逻辑合理且面向目标读者的目录。"
    "你可以对涉及的主题进行适当扩展或细化，确保内容的专业性和准确性、覆盖全面、知识成体系。"
)

writer_sys = (
    "你是技术写作机器人。给定 {chapter_title} 和 {section_title}，输出该小节的完整正文（Markdown 格式）。"
    "正文约 5000 ~ 50000 字，必要时包含代码块或示例。只返回 Markdown 文本。"
    "你是技术写作机器人（Writer Agent）。你会收到下列变量：{chapter_title}、{section_title}、{section_desc}、{audience}。"
    "任务：为该小节产出清晰、面向目标读者的 Markdown 正文。"
    "格式约束："
    " 1) 以一行 10-25 字的 TL;DR 开始（加粗或斜体）。"
    " 2) 正文应包含若干段落、必要时用小标题 (## 或 ###)、步骤列表或代码块。"
    " 3) 正文约 5000 ~ 50000 字（可根据节复杂度调整），应尽量详实。"
    " 4) 结尾处可添加一个“进一步阅读”小节（可选）。"
    " 5) 保证markdown渲染的可读性，如公式，图片，表格，代码块等。例子：内联公式：$ E=mc^2 $，块级公式：$$\nE=mc^2\n$$，代码块：```python\nprint('hello world')\n```"
    " 6) 对于图片，可以插入来自网络的相关图片，使用Markdown语法插入，例如：![描述](图片URL)。如wikipedia等权威网站的图片优先。"
    "必须严格只返回 Markdown 内容（无元信息、无多余解释）。"
    "重点："
    " 1) 逻辑连贯性：是否按教学进度由浅入深。\n"
    " 2) 完整性：是否覆盖理论、实践、案例、扩展。\n"
    " 3) 学习友好性：是否有引导性问题、总结、提示。\n"
    " 4) 技术正确性：代码、公式、术语是否正确。\n"
    " 5) 格式合规性：Markdown 是否规范，可否直接渲染。\n"
)

reviewer_sys = (
    "你是教材审稿专家（Reviewer Agent），负责严格编辑和改进。"
    "输入会包含 'type': 'toc' 或 'content'。"
    "输出格式：必须为 JSON。"
    "若 type=='toc'：返回 { 'type':'toc', 'issues': [...] }\n"
    "若 type=='content'：返回 { 'type':'content', 'issues': [...] }\n"
    "每个 issue 对象包含：\n"
    "  - 'severity': 'critical' | 'major' | 'minor'\n"
    "  - 'location': '章节/节名 或 行号'\n"
    "  - 'message': 问题简述\n"
    "  - 'explanation': 为什么是问题（1–3句）\n"
    "  - 'suggestion': 具体修改方案\n"
    "  - 'example_fix': 可选，修订后的示例片段\n"
    "对于 content，还需附加字段：'markdown_ok': true|false\n"
    "检查重点：\n"
    " 1) 逻辑连贯性：是否按教学进度由浅入深。\n"
    " 2) 完整性：是否覆盖理论、实践、案例、扩展。\n"
    " 3) 学习友好性：是否有引导性问题、总结、提示。\n"
    " 4) 技术正确性：代码、公式、术语是否正确。\n"
    " 5) 格式合规性：Markdown 是否规范，可否直接渲染。\n"
    "禁止返回额外说明文本，仅返回 JSON。"
)

# 创建 agents（构造函数可能因版本差异需要调整）
toc_agent = AssistantAgent(name="toc_agent", system_message=toc_sys, model_client=llm_client_writer)
toc_reviewer_agent = AssistantAgent(name="reviewer_agent", system_message=reviewer_sys, model_client=llm_client_viewer)

# ---------- v0.4 风格的异步辅助函数 ----------

def _extract_text_from_task_result(result) -> str:
    """从 TaskResult 中取最后一个文本消息的内容，兼容 v0.4 返回结构。"""
    # result.messages 是消息列表；取最后一个有 content 的 message
    for msg in reversed(result.messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return ""


async def generate_initial_toc(topic: str, audience: str, max_iter=MAX_TOC_ITER) -> Dict:
    prompt = f"请为主题 `{topic}`（面向 `{audience}`）生成整篇教程目录，注意分章节并列出小节。"
    result = await toc_agent.run(task=prompt)
    raw = _extract_text_from_task_result(result)
    try:
        toc = json.loads(raw)
    except Exception as e:
        raise ValueError(f"解析 TOC JSON 失败: {e}\n原始返回:\n{raw}")

    # reviewer 迭代改进（异步顺序进行）
    for i in range(max_iter):
        review_input = json.dumps({"type":"toc", "toc": toc, "audience": audience, "topic": topic}, ensure_ascii=False)
        r = await toc_reviewer_agent.run(task=review_input)
        raw_r = _extract_text_from_task_result(r)
        try:
            robj = json.loads(raw_r)
        except Exception:
            break
        improve_prompt = "请根据以下建议，返回最终 TOC（JSON）:\n" + json.dumps(robj, ensure_ascii=False)
        improved = await toc_agent.run(task=improve_prompt)
        try:
            toc = json.loads(_extract_text_from_task_result(improved))
        except Exception:
            break

    return toc


async def generate_and_improve_section(chapter_title: str, section_title: str, audience: str, topic: str, toc: Dict, max_iter=MAX_SECTION_ITER) -> str:

    writer_agent = AssistantAgent(name="writer_agent", system_message=writer_sys, model_client=llm_client_writer)
    reviewer_agent = AssistantAgent(name="reviewer_agent", system_message=reviewer_sys, model_client=llm_client_viewer)

    prompt = f"当前写作主题：主题 `{topic}`， 写作整体目录：目录 `{toc}`， 现在请为小节 `{section_title}`（所属章节：{chapter_title}）（面向 `{audience}`）写正文（Markdown）。"
    res = await writer_agent.run(task=prompt)
    content = _extract_text_from_task_result(res)

    for i in range(max_iter):
        review_input = json.dumps({"type": "content", "chapter": chapter_title, "section": section_title, "content": content, "audience": audience}, ensure_ascii=False)
        rv = await reviewer_agent.run(task=review_input)
        raw_r = _extract_text_from_task_result(rv)
        try:
            robj = json.loads(raw_r)
        except Exception:
            break
        # 将 reviewer 建议交给 writer 进一步改进（这里把建议直接当作新的 task）
        improve_prompt = "请根据以下建议，返回最终正文（Markdown）:\n" + json.dumps(robj, ensure_ascii=False)
        res2 = await writer_agent.run(task=improve_prompt)
        content = _extract_text_from_task_result(res2)

    return content


async def run_pipeline_v0_4(topic: str, audience: str, concurrency: int = CONCURRENCY) -> List[str]:
    toc = await generate_initial_toc(topic, audience)
    print(json.dumps(toc, ensure_ascii=False, indent=2))
    chapters = toc.get("chapters", [])

    # 构建所有待写小节的任务元数据
    tasks_meta: List[Tuple[int,int,str,str]] = []
    for ci, ch in enumerate(chapters, start=1):
        for si, sec in enumerate(ch.get("sections", []), start=1):
            tasks_meta.append((ci, si, ch.get("title",""), sec.get("title","")))

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def worker(ci:int, si:int, chapter_title:str, section_title:str):
        async with semaphore:
            try:
                md = await generate_and_improve_section(chapter_title, section_title, audience, topic, toc)
                path = save_section_md(ci, si, chapter_title, section_title, md)
                print(f"Saved: {path}")
                return path
            except Exception as e:
                print(f"Error {chapter_title} - {section_title}: {e}")
                return None

    # 并发提交所有任务
    coros = [worker(ci, si, ch_t, sec_t) for (ci, si, ch_t, sec_t) in tasks_meta]
    completed = await asyncio.gather(*coros)
    results = [p for p in completed if p]

    # 关闭 model client（释放资源）
    await llm_client_writer.close()
    await llm_client_viewer.close()

    print(f"全部任务完成，共保存 {len(results)} 个小节。")
    return results


# ---------- CLI 入口 ----------
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="AutoGen v0.4 异步并行多 agent 写作流水线")
    p.add_argument("--topic", required=False, default="如何用 AutoGen 搭建多智能体写作流水线（v0.4 异步版）")
    p.add_argument("--audience", required=False, default="熟悉 Python 的工程师")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = p.parse_args()

    asyncio.run(run_pipeline_v0_4(args.topic, args.audience, concurrency=args.concurrency))
