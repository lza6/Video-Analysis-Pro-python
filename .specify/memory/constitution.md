# Video Analysis Pro — 项目宪法（Constitution v1.0）

> 本文件是所有开发决策的最高准绳。任何代码变更、架构选择、发布动作都必须能对照本宪法给出依据。
> 创建于 2026-09-04（spec-kit 初始化），对应版本 v5.3.0。

## 核心原则

### I. 真实闭环（NON-NEGOTIABLE）
任何"完成"声明必须附真实执行证据：命令 + 真实输出。Mock 只证明隔离逻辑，永远不得表述为真实集成已通过。
- 违规示例：把 `pytest tests/unit/` 全绿说成"全链路已验证"。
- 合规做法：区分四层标签——`已验证`（实际运行）/ `静态确认`（读代码）/ `合理推断` / `待验证`。

### II. 本地隐私优先
用户的视频/转录/分析数据默认不出本机。任何联网能力（API LLM、web 搜索）必须显式配置 + UI 可见，默认关闭或用户主动填 Key。
- 新依赖不得引入"静默上传"行为。
- headless 服务必须鉴权（VAP_HEADLESS_TOKEN）后才能暴露到非本机网络。

### III. 单用户桌面形态优先，服务化能力按需
本项目核心是单机桌面应用。不为假想的分布式场景引入 Kafka/Redis/微服务；headless HTTP 是可选增强，不是架构重心。
- 判断标准：该复杂度在"一台用户电脑"上是否有对应收益？没有就不做。

### IV. 不推翻重写，渐进增强
三阶段流水线（Extract→Analyze→Media）+ Agent ReAct 面板 + ChromaDB KB 是既定骨架。新功能以接线、增补段、可选 skills 包形式落地。
- 死代码处置规则：已写未接线的模块要么接线要么标注 experimental 并在文档披露，不允许第三种"沉默共存"。

### V. 可回滚
每个批次提交独立可回滚（原子 commit）；发布打 tag；CI 全绿才发版。回滚路径 = revert commit + tag 重打。

### VI. Windows 生产环境一等公民
用户生产环境是 Windows 10。所有脚本必须兼容 PowerShell/`bat`；`.sh` 禁止；torch 先于 PyQt6 的 DLL 加载顺序是铁律，任何新 UI 模块必须带 torch 守卫。

### VII. 文档即代码
README/SOP/CHANGELOG 与代码同批更新。功能宣称不得超出已验证范围（例：基础 matplotlib 折线不得写作"高级可视化报表"）。

## 技术标准

### 代码质量
- Python：PEP8（ruff F 规则零容忍）+ mypy 全清零 + 类型注解渐进推进
- 测试：pytest 全绿为发版门槛；核心链路（Phase1/2/3/Agent/headless）必须有测试
- 单文件 <800 行（main_window.py God Object 是已知技术债，拆分需独立批次+Critic）

### Agent 提示词
- 遵循 agent_prompt.py 模块化结构；总 prompt 预算 <4000 字符（本地小模型）
- 每新增模块段必须有对应单测（段落顺序 + 预算上限）

### 安全
- 密钥只走 keyring / 环境变量；日志脱敏；subprocess 用列表参数
- skills 导入等文件操作必须做路径校验

## 决策框架

技术选型冲突时按序权衡：
1. 是否符合"真实闭环"（可验证性）
2. 是否保护本地隐私边界
3. 单用户桌面场景的真实收益
4. 对现有骨架的侵入程度（越小越好）
5. 维护成本

## 发布流程

1. 批次实现 → 定向验证 → 全量 pytest → mypy → ruff
2. CI 六平台矩阵 + build-windows 打包全绿
3. 打 tag（语义化版本）→ GitHub Release（含 artifact 冒烟记录）
4. CHANGELOG + 文档同步

## 审计与验收记录

- 每次终局审计的验证范围与结论记录在项目记忆（`workflow_status.md` / `.wolf/` 若存在），避免重复盲扫已验证区域。
- 审计问题分级：P0 阻塞 / P1 高 / P2 中 / P3 增强；P0/P1 必须修复或明确阻塞原因后才可发版。
