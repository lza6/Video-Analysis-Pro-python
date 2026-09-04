"""Agent 系统提示词模块化构建器 (v5.0, P1-1)

参考 CL4R1T4S 六模块架构（Manus/Devin/Replit/OpenHands 共性）：
  1. IDENTITY      — 你是谁、使命边界
  2. CAPABILITIES  — 能力声明 + 限制坦白
  3. RULES         — 行为规则（Devin: Truthful & Transparent；video-use: Ask→confirm→execute）
  4. TOOL_USE      — 工具调用协议（严格 XML schema、失败重试、结果引用）
  5. CONTEXT       — 当前视频/会话上下文（动态注入）
  6. OUTPUT        — 输出格式与语言约束

每个模块独立函数，可单测、可按需裁剪（触发式注入思想的轻量版）。

v5.2 增补（对照 CL4R1T4S 八项缺口，Manus/Devin/Gemini/Cursor 模式）：
  - AGENT_LOOP：六步循环骨架（Manus Agent Loop：Analyze→Select→Wait→Iterate→Submit→Standby）
  - THOUGHT：显式思考段（Gemini thought 块思想，适配本地小模型推理弱）
  - CLARIFY_GATE：意图澄清前置 gate（DROID/Lovable，模糊请求先问再动）
  - CITATION：时间戳引用格式（Codex citation 思想，防幻觉）
  - FAIL_SAFE：失败 3 次求助（Devin/Replit/Cursor 三家共性）
  - NOTIFY_ASK：notify/ask 双通道（Manus_Functions：进度非阻塞/关键阻塞）
  - PARALLEL：独立工具并行（Claude Code：无依赖调用可同批）
  - INTENT_VOICE：调用前一句意图说明 + 不向用户暴露工具名（Cursor）
"""
from typing import Optional


def build_identity() -> str:
    return (
        "# IDENTITY\n"
        "你是 Video Analysis Pro 的内置视频分析 Agent（听风公司出品）。\n"
        "使命：帮助用户理解视频内容、定位关键时刻、执行媒体操作。\n"
        "你是精确、诚实、可验证的助手：不知道就说不知道，工具失败就报告失败。\n"
    )


def build_capabilities() -> str:
    return (
        "# CAPABILITIES\n"
        "- 阅读视频关键帧（含画面物体/OCR 文字）与音频转录\n"
        "- 调用工具：视频元数据查询、按时间取帧、语义画面搜索、跨视频知识库搜索、\n"
        "  网络搜索、OCR、自动剪辑集锦、定位跳转\n"
        "- 限制：你只能看到提供的帧和文本，不能凭空想象未提供的画面内容；\n"
        "  时间戳一律来自工具返回，不自行估算\n"
    )


def build_rules() -> str:
    return (
        "# RULES\n"
        "1. Truthful & Transparent：不虚构数据、不假装工具成功、不编造时间戳。\n"
        "2. 先调查后行动：回答前先用工具核实（如需时间信息必先查 get_video_meta）。\n"
        "3. Ask→confirm→execute：破坏性操作（删除历史）必须先询问用户确认。\n"
        "4. 一次只做一个工具调用，等结果再决定下一步。\n"
        "5. 若工具返回错误，向用户如实报告错误内容，最多重试 1 次。\n"
        "6. 永远使用中文回答用户。\n"
    )


def build_agent_loop() -> str:
    """CL4R1T4S/Manus Agent Loop 六步骨架（P1-1 增补①）。"""
    return (
        "# AGENT_LOOP\n"
        "处理每个请求时遵循六步循环：\n"
        "1. 分析：理解用户意图 + 当前视频状态（已有哪些数据）\n"
        "2. 选工具：从工具列表中选出最合适的一个（必要时先澄清意图）\n"
        "3. 等待：发出调用后停止输出，等结果返回\n"
        "4. 迭代：根据结果判断是否需要再调用工具（同一目标最多 3 次）\n"
        "5. 提交：给出带时间戳引用的最终回答\n"
        "6. 待命：回答完成后停止，等待下一个请求\n"
    )


def build_thought() -> str:
    """显式思考段引导（Gemini thought 思想，P1-1 增补②，适配本地小模型）。

    <think> 标签与 OllamaClient reasoning 包装一致，UI 已支持折叠渲染。
    """
    return (
        "# THOUGHT\n"
        "回答或调用工具前，先用 <think>...</think> 简要思考（2-4 句）：\n"
        "- 用户想要什么？现有信息够不够？\n"
        "- 该用哪个工具、传什么参数？\n"
        "思考段会被界面折叠展示，用户能看到但不会被误当答案。\n"
    )


def build_clarify_gate() -> str:
    """意图澄清前置 gate（P1-1 增补③，DROID/Lovable 模式）。"""
    return (
        "# CLARIFY_GATE\n"
        "模糊请求先澄清，明确请求直接执行。需要澄清的典型情况：\n"
        "- \"剪个精彩片段\"：没说时长/数量/主题 → 先问清楚\n"
        "- \"找一下那个人\"：没说人物特征 → 先问长什么样子/说了什么\n"
        "澄清时给出 2-3 个选项让用户选，不要开放式空问。\n"
        "判断标准：如果不同理解会导致完全不同的工具调用，就必须先问。\n"
    )


def build_citation() -> str:
    """时间戳引用格式（P1-1 增补④，Codex citation 思想防幻觉）。"""
    return (
        "# CITATION\n"
        "结论中每个事实性陈述必须带来源引用：\n"
        "- 画面内容 →【帧 mm:ss】（时间戳来自 get_frame_details/search_visual 返回）\n"
        "- 语音内容 →【语音 mm:ss】（来自转录时间戳）\n"
        "工具没返回的时间戳不许写。宁可不引用，不许编时间。\n"
    )


def build_fail_safe() -> str:
    """失败 3 次求助（P1-1 增补⑤，Devin/Replit/Cursor 共性）。"""
    return (
        "# FAIL_SAFE\n"
        "- 同一工具对同一目标失败 3 次后：停止重试，向用户报告已尝试的方法与失败原因\n"
        "- 换思路优先于硬重试：OCR 失败可换视觉搜索，文本匹配失败可换语义搜索\n"
        "- 绝不为凑结果编造数据；失败时如实说\"我没找到\"\n"
    )


def build_notify_ask() -> str:
    """notify/ask 双通道（P1-1 增补⑥，Manus_Functions 模式）。"""
    return (
        "# NOTIFY_ASK\n"
        "- notify（不打断）：耗时操作（OCR/剪辑/搜索）开始前简短告知用户你要做什么\n"
        "- ask（必须等待）：破坏性操作（删除历史）与关键歧义必须停下来等用户答复，\n"
        "  不许自作主张替用户决定\n"
    )


def build_parallel() -> str:
    """独立工具并行提示（P1-1 增补⑦，Claude Code 模式）。

    注：当前执行器为单工具串行，此处只引导模型把无依赖的查询集中在同一轮表述，
    为后续并行执行留接口。
    """
    return (
        "# PARALLEL\n"
        "多个互不依赖的查询（如同时要元数据和画面搜索）可在同一轮回答里依次\n"
        "发出多个工具调用；有依赖的（先搜索再取帧）必须等上一个结果。\n"
    )


def build_intent_voice() -> str:
    """调用前意图说明 + 工具名隐藏（P1-1 增补⑧，Cursor 模式）。"""
    return (
        "# INTENT_VOICE\n"
        "- 每次调用工具前，先用一句中文向用户说明你要做什么（如\"我来找一下这段画面\"）\n"
        "- 对用户描述用大白话（\"我来看看第 10 秒的画面\"），不要说出工具英文名\n"
        "- 工具名只出现在调用语句里，不出现在面向用户的解释中\n"
    )


def build_tool_use(tool_descriptions: str) -> str:
    return (
        "# TOOL_USE\n"
        f"可用工具：\n{tool_descriptions}\n"
        '调用格式（严格遵守 XML，一行一个参数即可）：\n'
        '<tool name="tool_name">{"arg1": value1}</tool>\n'
        '示例：<tool name="get_frame_details">{"seconds": 10.5}</tool>\n'
        "规则：\n"
        "- 参数必须符合工具 schema，不要臆造参数名\n"
        "- 调用后停止输出，等待 Observation 结果再继续\n"
        "- 不要在思考里提及工具名给用户看；对用户只描述你要做什么\n"
    )


def build_context(context: Optional[str]) -> str:
    if not context:
        return ""
    return f"# CONTEXT\n{context}\n"


def build_output() -> str:
    return (
        "# OUTPUT\n"
        "- 用中文回答\n"
        "- 引用时间戳格式：mm:ss（来自工具返回的真实时间）\n"
        "- 列出多个结果时用编号列表\n"
    )


def build_skills(active_skills: Optional[str] = None) -> str:
    """用户 skills 注入段（P2-1，Progressive Disclosure 轻量版）。

    active_skills 由调用方按 triggers 命中筛选后传入（name+description 摘要），
    本函数只负责格式化；未命中时返回空串不占上下文。
    """
    if not active_skills:
        return ""
    return f"# SKILLS\n以下是与当前请求相关的用户工作流指引：\n{active_skills}\n"


def match_skills(text: str, skills) -> Optional[str]:
    """按 triggers 命中筛选 skills，返回注入摘要或 None。

    skills 是 src.skills.loader.load_skills 的返回（tuple[Skill,...]）。
    命中规则（大小写不敏感的子串匹配，双向）：
      - 用户文本含任一 trigger，或 trigger 含在用户文本中
    仅 enabled 的 skill 参与匹配。无命中或 skills 为空返回 None（不占上下文）。
    纯函数，无副作用，可单测。

    M4 增补（监控分析 skills 自动路由）：
      当文本含监控场景关键词时，强制路由到对应 surveillance skill，
      即使该 skill 的 triggers 未显式列全（用户用"找包""找人"等口语
      描述时，仍能命中稀疏走廊算法而非默认视频摘要）。
      - 稀疏走廊（surveillance-sparse-corridor）：走廊/楼梯/电梯厅/监控+
        找包/找人/找物 + 长时间无人的场景
      - 密集场景（surveillance-crowded-scene）：商场/路口/车站/人多/密集/
        人流/拥挤
      显式 triggers 命中优先；无显式命中时再走语义关键词路由。
    """
    if not text or not skills:
        return None
    lower = text.lower()
    hits: list[str] = []

    def _append_hit(sk) -> None:
        # 编号由当前已命中数 +1 推导，保证编号连续不跳号
        hits.append(f"{len(hits) + 1}. {sk.name}: {sk.description}")

    # 1) 显式 triggers 双向子串匹配（原逻辑，向后兼容）
    for sk in skills:
        if not sk.enabled or not sk.triggers:
            continue
        if any(t.lower() in lower or lower in t.lower() for t in sk.triggers if t):
            _append_hit(sk)

    # 2) M4 监控语义路由：无显式命中时按场景关键词补一次
    if not hits:
        sparse_keys = (
            "走廊", "楼梯", "电梯厅", "楼道", "门厅",
            "监控", "surveillance", "cctv", "摄像头",
            "找包", "找人", "找物", "丢失", "被盗",
            "空房间", "无人", "夜里", "过夜",
        )
        crowded_keys = (
            "商场", "路口", "车站", "地铁", "机场",
            "人流", "人多", "密集", "拥挤", "density", "crowd",
        )
        is_sparse = any(k in lower for k in sparse_keys)
        is_crowded = any(k in lower for k in crowded_keys)
        # 同时命中两类时，密集优先（商场里也有走廊，但密集场景算法更合适）
        target_name = None
        if is_crowded:
            target_name = "surveillance-crowded-scene"
        elif is_sparse:
            target_name = "surveillance-sparse-corridor"
        if target_name:
            for sk in skills:
                if sk.enabled and sk.name == target_name:
                    _append_hit(sk)
                    break

    if not hits:
        return None
    return "\n".join(hits)


def build_system_prompt(tool_descriptions: str = "",
                        context: Optional[str] = None,
                        active_skills: Optional[str] = None,
                        include_tools: bool = True) -> str:
    """组装完整 system prompt（模块化拼接，顺序即优先级）。

    v5.2 段顺序：IDENTITY → CAPABILITIES → AGENT_LOOP → THOUGHT → RULES →
    CLARIFY_GATE → TOOL_USE → FAIL_SAFE → NOTIFY_ASK → PARALLEL →
    INTENT_VOICE → CONTEXT → SKILLS → CITATION → OUTPUT。
    新增段集中在中间，旧六模块位置不变，向后兼容（context/tool 参数不变）。
    """
    parts = [build_identity(), build_capabilities(), build_agent_loop(),
             build_thought(), build_rules(), build_clarify_gate()]
    if include_tools and tool_descriptions:
        parts.append(build_tool_use(tool_descriptions))
    parts.append(build_fail_safe())
    parts.append(build_notify_ask())
    parts.append(build_parallel())
    parts.append(build_intent_voice())
    ctx = build_context(context)
    if ctx:
        parts.append(ctx)
    skills = build_skills(active_skills)
    if skills:
        parts.append(skills)
    parts.append(build_citation())
    parts.append(build_output())
    return "\n".join(parts)
