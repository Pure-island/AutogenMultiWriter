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
    },
    )


# system prompts（可按需微调）
toc_sys = (
    "你是目录生成专家。接收主题和目标读者后，返回一个分章分节的目录。"
    "必须返回严格 JSON，格式如下：{\n  \"title\": \"<整篇标题>\",\n  \"chapters\": [ {\"title\":\"..\", \"sections\": [ {\"title\":\"..\", \"desc\":\"一句话概述\"} ] } ]\n}"
    "不要返回额外说明文本。不需要```json等标记。"
    
)

writer_sys = (
    "你是技术写作机器人。给定 {chapter_title} 和 {section_title}，输出该小节的完整正文（Markdown 格式）。"
    "正文约 2000 ~ 10000 字，必要时包含代码块或示例。只返回 Markdown 文本。"
    "你是技术写作机器人（Writer Agent）。你会收到下列变量：{chapter_title}、{section_title}、{section_desc}、{audience}。"
    "任务：为该小节产出清晰、面向目标读者的 Markdown 正文。"
    "格式约束："
    " 1) 以一行 10-25 字的 TL;DR 开始（加粗或斜体）。"
    " 2) 正文应包含若干段落、必要时用小标题 (## 或 ###)、步骤列表或代码块。"
    " 3) 正文约 2000 ~ 10000 字（可根据节复杂度调整），应尽量详实。"
    " 4) 结尾处可添加一个“进一步阅读”小节（可选）。"
    " 5) 保证markdown渲染的可读性，如公式需要包裹在markdown的公式标签中。"
    "必须严格只返回 Markdown 内容（无元信息、无多余解释）。"
)

reviewer_sys = (
    "你是严格的编辑/改进者。输入会指定 'type': 'toc' 或 'content'。\n"
    "- 若 type=='toc'，返回 JSON: { 'type':'toc', 'issues': [...] }\n"
    "- 若 type=='content'，返回 JSON: { 'type':'content', 'issues': [...] }\n"
    "请严格只返回 JSON。不需要额外说明文本。不需要```json等标记。"
    "检查重点（但不限于）：逻辑连贯性、面向读者的深度/广度、先决知识遗漏、概念错误、重复内容、可读性与格式、代码示例正确性（如有）。"
    "输出要求：严格 JSON。"
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
        review_input = json.dumps({"type":"toc", "toc": toc, "audience": audience}, ensure_ascii=False)
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


async def generate_and_improve_section(chapter_title: str, section_title: str, audience: str, max_iter=MAX_SECTION_ITER) -> str:

    writer_agent = AssistantAgent(name="writer_agent", system_message=writer_sys, model_client=llm_client_dpskv3_1)
    reviewer_agent = AssistantAgent(name="reviewer_agent", system_message=reviewer_sys, model_client=llm_client_kimik2)

    prompt = f"请为小节 `{section_title}`（所属章节：{chapter_title}）（面向 `{audience}`）写正文（Markdown）。"
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
                md = await generate_and_improve_section(chapter_title, section_title, audience)
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
    await llm_client_dpskv3_1.close()
    await llm_client_dpskv3_1.close()

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
