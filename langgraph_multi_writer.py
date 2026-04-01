import argparse
import asyncio
import json
import logging
import os
import pathlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


logger = logging.getLogger(__name__)

OUT_DIR = pathlib.Path(__file__).parent / "output_async"
OUT_DIR.mkdir(exist_ok=True)
MAX_TOC_ITER = 2
MAX_SECTION_ITER = 1
CONCURRENCY = 8
RETRY_DELAYS_SECONDS = (5, 15, 30)
MAX_STREAM_CONTINUATIONS = 2


def load_env_file(env_path: pathlib.Path) -> None:
    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


load_env_file(pathlib.Path(__file__).parent / ".env")


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s+-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s[:120]


def _chapter_dir_for(chapter_idx: int, chapter_title: str) -> pathlib.Path:
    return OUT_DIR / f"{chapter_idx:02d}_{slugify(chapter_title)}"


def _section_path_for(
    chapter_idx: int,
    section_idx: int,
    chapter_title: str,
    section_title: str,
) -> pathlib.Path:
    chapter_dir = _chapter_dir_for(chapter_idx, chapter_title)
    filename = f"{chapter_idx:02d}_{section_idx:02d}_{slugify(section_title)}.md"
    return chapter_dir / filename


def _load_saved_section_content(section_path: pathlib.Path) -> str:
    with open(section_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    body_start = 0
    while body_start < len(lines) and (
        lines[body_start].startswith("# ") or lines[body_start].startswith("## ")
    ):
        body_start += 1

    while body_start < len(lines) and not lines[body_start].strip():
        body_start += 1

    return "\n".join(lines[body_start:]) if body_start < len(lines) else ""


def save_section_md(
    chapter_idx: int,
    section_idx: int,
    chapter_title: str,
    section_title: str,
    content_md: str,
) -> str:
    chapter_dir = _chapter_dir_for(chapter_idx, chapter_title)
    chapter_dir.mkdir(exist_ok=True)
    path = _section_path_for(chapter_idx, section_idx, chapter_title, section_title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {chapter_title}\n\n## {section_title}\n\n")
        f.write(content_md)
    return str(path)


def _toc_path_for_topic(topic: str) -> pathlib.Path:
    return OUT_DIR / f"00_toc_{slugify(topic)}.json"


def _save_toc_to_file(toc: Dict[str, Any], topic: str) -> pathlib.Path:
    toc_path = _toc_path_for_topic(topic)
    with open(toc_path, "w", encoding="utf-8") as f:
        json.dump(toc, f, ensure_ascii=False, indent=2)
    print(f"TOC saved to {toc_path}")
    return toc_path


def _load_toc_from_file(toc_file: str) -> Dict[str, Any]:
    toc_path = pathlib.Path(toc_file)
    with open(toc_path, "r", encoding="utf-8") as f:
        toc = json.load(f)
    _validate_toc_structure(toc)
    print(f"Loaded TOC from {toc_path}")
    return toc


def _validate_toc_structure(toc: Dict[str, Any]) -> None:
    if not isinstance(toc, dict):
        raise ValueError("TOC 必须是 JSON 对象")
    if "title" not in toc or "chapters" not in toc:
        raise ValueError("TOC 缺少必要字段: title 或 chapters")
    if not isinstance(toc["chapters"], list):
        raise ValueError("TOC 字段 chapters 必须是数组")

    for idx, chapter in enumerate(toc["chapters"], start=1):
        if not isinstance(chapter, dict):
            raise ValueError(f"第 {idx} 章不是对象")
        if "title" not in chapter or "sections" not in chapter:
            raise ValueError(f"第 {idx} 章缺少 title 或 sections")
        if not isinstance(chapter["sections"], list):
            raise ValueError(f"第 {idx} 章的 sections 必须是数组")


def _print_toc(toc: Dict[str, Any]) -> None:
    print("\n========== 当前目录 ==========")
    print(f"标题: {toc.get('title', '')}")
    for ci, ch in enumerate(toc.get("chapters", []), start=1):
        print(f"{ci:02d}. {ch.get('title', '')}")
        for si, sec in enumerate(ch.get("sections", []), start=1):
            desc = sec.get("desc", "")
            line = f"    {ci:02d}.{si:02d} {sec.get('title', '')}"
            if desc:
                line += f" - {desc}"
            print(line)
    print("==============================\n")


def _print_generation_brief(topic: str, audience: str, planning_notes: str) -> None:
    print("\n========== 写作需求确认 ==========")
    print(f"主题: {topic}")
    print(f"目标读者: {audience}")
    print("补充要求:")
    print(planning_notes if planning_notes.strip() else "(无)")
    print("===============================\n")


def _confirm_generation_brief(
    topic: str, audience: str, planning_notes: str = ""
) -> Tuple[bool, bool, str, str, str]:
    changed = False

    while True:
        _print_generation_brief(topic, audience, planning_notes)
        print("开始生成目录前，请先确认需求：")
        print("  [a] 认可当前需求，开始生成目录")
        print("  [t] 修改主题")
        print("  [u] 修改目标读者")
        print("  [n] 添加补充要求")
        print("  [c] 清空补充要求")
        print("  [q] 退出")
        choice = input("请选择操作（默认 a）: ").strip().lower()

        if choice in ("", "a", "accept"):
            return True, changed, topic, audience, planning_notes
        if choice in ("t", "topic"):
            new_topic = input("请输入新的主题: ").strip()
            if new_topic:
                topic = new_topic
                changed = True
            continue
        if choice in ("u", "audience"):
            new_audience = input("请输入新的目标读者: ").strip()
            if new_audience:
                audience = new_audience
                changed = True
            continue
        if choice in ("n", "note", "notes"):
            note = input("请输入补充要求（将追加）: ").strip()
            if note:
                planning_notes = (
                    f"{planning_notes}\n{note}".strip() if planning_notes else note
                )
                changed = True
            continue
        if choice in ("c", "clear"):
            if planning_notes:
                planning_notes = ""
                changed = True
            continue
        if choice in ("q", "quit", "exit"):
            return False, False, topic, audience, planning_notes


toc_sys = (
    "你是专业的教材大纲与课程设计专家。"
    "你是教材大纲与章节内容设计专家。接收主题和目标读者后，返回一个分章分节的目录。"
    '必须返回严格 JSON，格式如下：{\n  "title": "<整篇标题>",\n  "chapters": [ {"title":"..", "sections": [ {"title":"..", "desc":"一句话详述本节内容"} ] } ]\n}'
    "不要返回额外说明文本。不需要```json等标记。"
    "根据内容的复杂度，生成 3 ~ 20 章，每章 2 ~ 20 节。可酌情修改章节数量。"
    "标题应当简明且具有概括性\n"
    "小节描述需准确反映核心内容\n"
    "重点是相关的知识覆盖要全面且有深度、成体系\n"
    "对于给定的主题，请生成一个清晰、逻辑合理且面向目标读者的目录。"
    "你可以对涉及的主题进行适当扩展或细化，确保内容的专业性和准确性、覆盖全面、知识成体系。"
    "注意：现在生成结果可能出现“极”字符污染，输出token可能被随机替换为“极”，请注意纠正。"
)

writer_sys = (
    "你是技术写作机器人。给定 {chapter_title} 和 {section_title}，输出该小节的完整正文（Markdown 格式）。"
    "正文约 5000 ~ 50000 字，必要时包含代码块或示例。只返回 Markdown 文本。"
    "你是技术写作机器人（Writer Agent）。你会收到下列变量：{chapter_title}、{section_title}、{section_desc}、{audience}、{toc}、{topic}。"
    "任务：为该小节产出清晰、面向目标读者的 Markdown 正文。"
    "格式约束："
    " 1) 以一行 10-25 字的 TL;DR 开始（加粗或斜体）。"
    " 2) 正文应包含若干段落、必要时用小标题 (## 或 ###)、步骤列表或代码块。"
    " 3) 正文约 5000 ~ 50000 字（可根据节复杂度调整），应尽量详实。"
    " 4) 结尾处可添加一个“进一步阅读”小节（可选）。"
    " 5) 保证代码、公式、图片、表格等的可读性。例子：内联公式：$ E=mc^2 $，块级公式：$$\nE=mc^2\n$$，代码块：``python\nprint('hello world')\n```"
    " 6) 对于图片，可以插入来自网络的相关图片，使用Markdown语法插入，例如：![描述](图片URL)。因为无法生成图片，所以URL可以留空，后续再补充。"
    "必须严格只返回 Markdown 内容（无元信息、无多余解释）。"
    "重点："
    " 1) 逻辑连贯性：是否按教学进度由浅入深。\n"
    " 2) 完整性：是否覆盖理论、实践、案例、扩展。\n"
    " 3) 学习友好性：是否有引导性问题、总结、提示。\n"
    " 4) 技术正确性：代码、公式、术语是否正确。\n"
    " 5) 格式合规性：Markdown 是否规范，可否直接渲染。\n"
    " 6) 本节涉及到的内容，如果在前面章节没有提到过，请务必解释清楚。\n"
    " 7) 若存在公式，公式原理是否解释清楚，公式参数是否有详细描述解释。"
    "注意：现在生成结果可能出现“极”字符污染，输出token可能被随机替换为“极”，请注意纠正。"
)

reviewer_sys = (
    "你是教材审稿专家（Reviewer Agent），负责严格编辑和改进。"
    "输入会包含 'type': 'toc' 或 'content'。"
    "输出格式：必须为 JSON。不需要```json等标记。"
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
    " 6) 知识覆盖面：是否包含所有应有的知识点。\n"
    "禁止返回额外说明文本，仅返回 JSON。不需要```json等标记。"
    "注意：现在生成结果可能出现“极”字符污染，输出token可能被随机替换为“极”，请注意纠正。"
)

json_sys = (
    "你是json格式整理机器人（json format Agent），负责严格整理json格式。"
    "你的输入是一段json格式文本。其格式可能存在错误。"
    "你的任务是整理成规范的 JSON 格式并返回。"
    "注意：现在输入可能出现“极”字符污染，token可能被随机替换为“极”，请注意纠正。"
    "禁止返回额外说明文本，仅返回 JSON。不需要```json等标记。"
    '必须返回严格 JSON，格式如下：{\n  "title": "<整篇标题>",\n  "chapters": [ {"title":"..", "sections": [ {"title":"..", "desc":"一句话详述本节内容"} ] } ]\n}'
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}")
    return value


def _env_or_fallback(primary: str, fallback: str, default: str = "") -> str:
    return os.environ.get(primary, "").strip() or default


def _extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _create_chat_model(role: str) -> ChatOpenAI:
    if role == "writer":
        model = _env_or_fallback("WRITER_MODEL", "", "gemini-3.1-pro-preview")
        api_key = _require_env("WRITER_API_KEY")
        base_url = _env_or_fallback("WRITER_BASE_URL", "", "https://api.wow3.top/v1")
    else:
        model = _env_or_fallback("REVIEWER_MODEL", "", "deepseek-ai/DeepSeek-V3.2")
        api_key = _require_env("REVIEWER_API_KEY")
        base_url = _env_or_fallback(
            "REVIEWER_BASE_URL", "", "https://api.siliconflow.cn/v1"
        )

    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)


async def _ainvoke_text(model: ChatOpenAI, prompt: str, operation: str) -> str:
    last_error: Optional[Exception] = None
    partial_text = ""
    current_prompt = prompt

    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            chunks: List[str] = []
            async for chunk in model.astream(current_prompt):
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    chunks.append(content)
                elif isinstance(content, list):
                    chunks.extend(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("text")
                    )

            received = "".join(chunks)
            if received.strip():
                return partial_text + received

            last_error = ValueError(f"{operation} 返回空内容")
        except Exception as exc:
            if "chunks" in locals() and chunks:
                partial_text += "".join(chunks)
            last_error = exc

        if partial_text.strip() and attempt < MAX_STREAM_CONTINUATIONS:
            current_prompt = (
                "下面是已经生成成功的部分内容，请从最后一句自然继续续写，"
                "不要重复前文，不要重写开头，只补全后续缺失部分。\n\n"
                f"原始任务：{prompt}\n\n"
                f"已生成内容：\n{partial_text}"
            )
            continue

        if attempt < len(RETRY_DELAYS_SECONDS):
            wait_time = RETRY_DELAYS_SECONDS[attempt] + random.uniform(0, 1)
            logger.warning(
                "%s 失败，第 %d 次重试前等待 %.1f 秒: %s",
                operation,
                attempt + 1,
                wait_time,
                last_error,
            )
            await asyncio.sleep(wait_time)

    if partial_text.strip():
        return partial_text

    raise RuntimeError(f"{operation} 连续重试后仍失败") from last_error


class TocState(TypedDict, total=False):
    topic: str
    audience: str
    planning_notes: str
    raw_toc: str
    review_feedback: str
    review_round: int
    max_iter: int
    toc: Dict[str, Any]


class SectionState(TypedDict, total=False):
    topic: str
    audience: str
    planning_notes: str
    toc: Dict[str, Any]
    chapter_title: str
    section_title: str
    section_desc: str
    content: str
    review_feedback: str
    review_round: int
    max_iter: int


async def _toc_generate_node(state: TocState) -> Dict[str, Any]:
    notes_text = (
        f"\n补充要求：{state['planning_notes']}" if state.get("planning_notes") else ""
    )
    prompt = (
        toc_sys
        + "\n\n"
        + f"请为主题 `{state['topic']}`（面向 `{state['audience']}`）生成整篇教程目录，注意分章节并列出小节。{notes_text}"
    )
    model = _create_chat_model("writer")
    raw_toc = await _ainvoke_text(model, prompt, f"生成主题 {state['topic']} 的目录")
    return {"raw_toc": raw_toc, "review_round": 0}


async def _toc_review_node(state: TocState) -> Dict[str, Any]:
    review_input = json.dumps(
        {
            "type": "toc",
            "toc": state["raw_toc"],
            "audience": state["audience"],
            "topic": state["topic"],
            "planning_notes": state.get("planning_notes", ""),
        },
        ensure_ascii=False,
    )
    model = _create_chat_model("reviewer")
    prompt = reviewer_sys + "\n\n" + review_input
    feedback = await _ainvoke_text(model, prompt, f"评审主题 {state['topic']} 的目录")
    return {"review_feedback": feedback, "review_round": state["review_round"] + 1}


async def _toc_improve_node(state: TocState) -> Dict[str, Any]:
    prompt = (
        toc_sys
        + "\n\n请根据以下建议，返回最终 TOC（JSON）：\n"
        + state["review_feedback"]
    )
    model = _create_chat_model("writer")
    raw_toc = await _ainvoke_text(model, prompt, f"改进主题 {state['topic']} 的目录")
    return {"raw_toc": raw_toc}


def _toc_should_continue(state: TocState) -> str:
    if state["review_round"] < state["max_iter"]:
        return "review"
    return "normalize"


async def _toc_normalize_node(state: TocState) -> Dict[str, Any]:
    prompt = json_sys + "\n\n" + state["raw_toc"]
    model = _create_chat_model("reviewer")
    normalized_raw = await _ainvoke_text(
        model, prompt, f"整理主题 {state['topic']} 的目录 JSON"
    )
    normalized_text = _extract_json_text(normalized_raw)
    try:
        toc = json.loads(normalized_text)
    except Exception as exc:
        raise ValueError(f"解析 TOC JSON 失败: {exc}\n原始返回:\n{normalized_raw}")
    _validate_toc_structure(toc)
    return {"toc": toc}


def _build_toc_graph():
    graph = StateGraph(TocState)
    graph.add_node("generate", _toc_generate_node)
    graph.add_node("review", _toc_review_node)
    graph.add_node("improve", _toc_improve_node)
    graph.add_node("normalize", _toc_normalize_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "review")
    graph.add_edge("review", "improve")
    graph.add_conditional_edges(
        "improve",
        _toc_should_continue,
        {"review": "review", "normalize": "normalize"},
    )
    graph.add_edge("normalize", END)
    return graph.compile()


async def _section_write_node(state: SectionState) -> Dict[str, Any]:
    notes_text = (
        f"；整体补充要求：{state['planning_notes']}"
        if state.get("planning_notes")
        else ""
    )
    prompt = (
        writer_sys
        + "\n\n"
        + f"当前写作主题：主题 `{state['topic']}`，写作整体目录：目录 `{state['toc']}`，现在请为小节 `{state['section_title']}`（所属章节：{state['chapter_title']}）（面向 `{state['audience']}`）写正文（Markdown）。小节描述：{state['section_desc']}{notes_text}。不要只写概述，优先写成 5000 字以上的完整小节，尽量覆盖背景、原理、步骤、示例、注意事项与小结。"
    )
    model = _create_chat_model("writer")
    content = await _ainvoke_text(
        model,
        prompt,
        f"生成小节 {state['chapter_title']} - {state['section_title']} 的正文",
    )
    return {"content": content, "review_round": 0}


async def _section_review_node(state: SectionState) -> Dict[str, Any]:
    review_input = json.dumps(
        {
            "type": "content",
            "toc": state["toc"],
            "topic": state["topic"],
            "chapter": state["chapter_title"],
            "section": state["section_title"],
            "section_desc": state["section_desc"],
            "content": state["content"],
            "audience": state["audience"],
            "planning_notes": state.get("planning_notes", ""),
        },
        ensure_ascii=False,
    )
    model = _create_chat_model("reviewer")
    prompt = reviewer_sys + "\n\n" + review_input
    feedback = await _ainvoke_text(
        model, prompt, f"评审小节 {state['chapter_title']} - {state['section_title']}"
    )
    return {"review_feedback": feedback, "review_round": state["review_round"] + 1}


async def _section_revise_node(state: SectionState) -> Dict[str, Any]:
    notes_text = (
        f"；整体补充要求：{state['planning_notes']}"
        if state.get("planning_notes")
        else ""
    )
    prompt = (
        writer_sys
        + "\n\n"
        + f"当前写作主题：主题 `{state['topic']}`，写作整体目录：目录 `{state['toc']}`，现在请根据以下建议为小节 `{state['section_title']}`（所属章节：{state['chapter_title']}）（面向 `{state['audience']}`）改进正文（Markdown）。小节描述：{state['section_desc']}{notes_text}。不要只写概述，优先扩展成 5000 字以上的完整小节，尽量补充原理、步骤、示例、细节解释、注意事项与小结。\n建议：{state['review_feedback']}"
    )
    model = _create_chat_model("writer")
    content = await _ainvoke_text(
        model, prompt, f"改进小节 {state['chapter_title']} - {state['section_title']}"
    )
    return {"content": content}


def _section_should_continue(state: SectionState) -> str:
    if state["review_round"] < state["max_iter"]:
        return "review"
    return "done"


def _build_section_graph():
    graph = StateGraph(SectionState)
    graph.add_node("write", _section_write_node)
    graph.add_node("review", _section_review_node)
    graph.add_node("revise", _section_revise_node)
    graph.add_edge(START, "write")
    graph.add_edge("write", "review")
    graph.add_edge("review", "revise")
    graph.add_conditional_edges(
        "revise",
        _section_should_continue,
        {"review": "review", "done": END},
    )
    return graph.compile()


async def improve_toc_with_feedback(
    toc: Dict[str, Any],
    topic: str,
    audience: str,
    feedback: str,
    planning_notes: str = "",
) -> Dict[str, Any]:
    prompt = (
        toc_sys
        + "\n\n下面是当前教程目录（JSON）和用户反馈。"
        + "请基于用户反馈修改目录，并返回严格 JSON。"
        + f"\n主题：{topic}\n目标读者：{audience}"
        + (f"\n补充要求：{planning_notes}" if planning_notes.strip() else "")
        + f"\n用户反馈：{feedback}\n当前目录：{json.dumps(toc, ensure_ascii=False)}"
    )
    model = _create_chat_model("writer")
    raw = await _ainvoke_text(model, prompt, "根据反馈修改目录")
    normalizer = _create_chat_model("reviewer")
    normalized_raw = await _ainvoke_text(
        normalizer, json_sys + "\n\n" + raw, "整理修改后的目录 JSON"
    )
    normalized_text = _extract_json_text(normalized_raw)
    new_toc = json.loads(normalized_text)
    _validate_toc_structure(new_toc)
    return new_toc


async def generate_initial_toc(
    topic: str,
    audience: str,
    max_iter: int = MAX_TOC_ITER,
    force_regenerate: bool = False,
    save: bool = True,
    planning_notes: str = "",
) -> Dict[str, Any]:
    toc_path = _toc_path_for_topic(topic)
    print(f"TOC path: {toc_path}")
    if toc_path.exists() and not force_regenerate:
        print(f"Loading existing TOC from {toc_path}")
        with open(toc_path, "r", encoding="utf-8") as f:
            toc = json.load(f)
        _validate_toc_structure(toc)
        return toc

    graph = _build_toc_graph()
    state = await graph.ainvoke(
        {
            "topic": topic,
            "audience": audience,
            "planning_notes": planning_notes,
            "max_iter": max_iter,
        }
    )
    toc = state["toc"]
    if save:
        _save_toc_to_file(toc, topic)
    return toc


def _find_section_desc(
    toc: Dict[str, Any], chapter_title: str, section_title: str
) -> str:
    for chapter in toc.get("chapters", []):
        if chapter.get("title") != chapter_title:
            continue
        for section in chapter.get("sections", []):
            if section.get("title") == section_title:
                return section.get("desc", "")
    return ""


async def generate_and_improve_section(
    chapter_idx: int,
    section_idx: int,
    chapter_title: str,
    section_title: str,
    audience: str,
    topic: str,
    toc: Dict[str, Any],
    max_iter: int = MAX_SECTION_ITER,
    planning_notes: str = "",
) -> str:
    section_path = _section_path_for(
        chapter_idx, section_idx, chapter_title, section_title
    )
    if section_path.exists():
        print(f"Loading existing section from {section_path}")
        return _load_saved_section_content(section_path)

    section_desc = _find_section_desc(toc, chapter_title, section_title)
    graph = _build_section_graph()
    state = await graph.ainvoke(
        {
            "topic": topic,
            "audience": audience,
            "planning_notes": planning_notes,
            "toc": toc,
            "chapter_title": chapter_title,
            "section_title": section_title,
            "section_desc": section_desc,
            "max_iter": max_iter,
        }
    )
    return state["content"]


async def run_pipeline(
    topic: str,
    audience: str,
    concurrency: int = CONCURRENCY,
    planning_notes: str = "",
    toc_file: str = "",
) -> List[str]:
    accepted, force_regenerate, topic, audience, planning_notes = (
        _confirm_generation_brief(topic, audience, planning_notes)
    )
    if not accepted:
        print("已退出，未开始生成目录和章节内容。")
        return []

    if toc_file:
        toc = _load_toc_from_file(toc_file)
        _save_toc_to_file(toc, topic)
        print("已使用外部 TOC，跳过目录生成，直接开始正文写作...\n")
    else:
        toc = await generate_initial_toc(
            topic,
            audience,
            max_iter=MAX_TOC_ITER,
            force_regenerate=force_regenerate,
            save=False,
            planning_notes=planning_notes,
        )

        while True:
            _print_toc(toc)
            print("目录确认选项：")
            print("  [a] 接受目录并继续写作")
            print("  [m] 提意见并修改目录")
            print("  [r] 丢弃并重新生成目录")
            print("  [q] 退出")
            choice = input("请选择操作（默认 a）: ").strip().lower()

            if choice in ("", "a", "accept"):
                _save_toc_to_file(toc, topic)
                print("目录已确认，开始生成正文...\n")
                break
            if choice in ("m", "modify", "edit"):
                feedback = input("请输入你对目录的修改意见: ").strip()
                if not feedback:
                    print("修改意见不能为空，请重新输入。\n")
                    continue
                toc = await improve_toc_with_feedback(
                    toc, topic, audience, feedback, planning_notes=planning_notes
                )
                print("目录已根据你的反馈更新。\n")
                continue
            if choice in ("r", "regenerate", "regen"):
                toc = await generate_initial_toc(
                    topic,
                    audience,
                    max_iter=MAX_TOC_ITER,
                    force_regenerate=True,
                    save=False,
                    planning_notes=planning_notes,
                )
                print("目录已重新生成。\n")
                continue
            if choice in ("q", "quit", "exit"):
                print("已退出，未开始生成章节内容。")
                return []

    tasks_meta: List[Tuple[int, int, str, str]] = []
    for ci, ch in enumerate(toc.get("chapters", []), start=1):
        for si, sec in enumerate(ch.get("sections", []), start=1):
            tasks_meta.append((ci, si, ch.get("title", ""), sec.get("title", "")))

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(ci: int, si: int, chapter_title: str, section_title: str):
        async with semaphore:
            try:
                md = await generate_and_improve_section(
                    ci,
                    si,
                    chapter_title,
                    section_title,
                    audience,
                    topic,
                    toc,
                    max_iter=MAX_SECTION_ITER,
                    planning_notes=planning_notes,
                )
                path = save_section_md(ci, si, chapter_title, section_title, md)
                print(f"Saved: {path}")
                return path
            except Exception as exc:
                print(f"Error {chapter_title} - {section_title}: {exc}")
                return None

    completed = await asyncio.gather(
        *[worker(ci, si, ch_t, sec_t) for ci, si, ch_t, sec_t in tasks_meta]
    )
    results = [path for path in completed if path]
    print(f"全部任务完成，共保存 {len(results)} 个小节。")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph 异步并行多节点写作流水线")
    parser.add_argument(
        "--topic",
        required=False,
        default="如何用 LangChain 与 LangGraph 搭建多智能体写作流水线",
        help="Direct topic string or path to a .txt file containing the topic",
    )
    parser.add_argument("--audience", required=False, default="熟悉 Python 的工程师")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--max-toc-iter", type=int, default=2)
    parser.add_argument("--max-section-iter", type=int, default=1)
    parser.add_argument("--notes", required=False, default="")
    parser.add_argument(
        "--toc-file",
        required=False,
        default="",
        help="Path to an existing TOC JSON file; if provided, skip TOC generation and write sections directly",
    )
    args = parser.parse_args()

    MAX_TOC_ITER = args.max_toc_iter
    MAX_SECTION_ITER = args.max_section_iter

    topic = args.topic
    if topic.endswith(".txt") and os.path.isfile(topic):
        with open(topic, "r", encoding="utf-8") as f:
            topic = f.read().strip()
        print(f"Read topic from file: {topic}")
    else:
        print(f"Using direct topic: {topic}")

    asyncio.run(
        run_pipeline(
            topic,
            args.audience,
            concurrency=args.concurrency,
            planning_notes=args.notes,
            toc_file=args.toc_file,
        )
    )
