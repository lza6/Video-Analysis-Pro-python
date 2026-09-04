# Feature Specification: v5.4 终局审计补漏

## Problem Statement
项目已部署生产使用，但双审计（盲点扫描 + 生产架构）发现 20 项真实缺口：headless 并发雪崩、.env.example 死配置、Docker 默认无鉴权、假按钮、死功能（checkpoints）、资源泄漏、文档漂移等。需补齐至"真实可生产"。

## 用户故事与验收标准

### US1: headless 服务接真实调用方不雪崩
作为 API 集成方，我希望并发上传多个视频时服务不 OOM/不崩，超时/重复提交有合理处理。
- [ ] 并发 2 请求串行化（信号量），不重复加载 Whisper
- [ ] /analyze 响应不含服务器内部路径
- [ ] 鉴权文档强制说明（README + docker-compose）

### US2: 小白按文档配置能跑通
作为新用户，我希望按 .env.example / README 配置监控分析能真实生效。
- [ ] .env.example 的每个变量在代码中有消费点，或文件明确标注真实配置途径
- [ ] 未加载模型点 Phase2/3 有中文弹窗而非静默

### US3: 数据清理不留孤儿
作为维护者，我希望删除历史/清空全部后 ChromaDB 无残留。
- [ ] clear_all_history 清理孤儿 KB 向量（session 已不存在的）
- [ ] user_preferences 有清理入口

### US4: 资源不泄漏
- [ ] highlight_cut VideoFileClip try/finally 关闭 + 唯一文件名
- [ ] CLIP 语义去重分批 + 关闭 Image
- [ ] 关窗时 ChatWorker 网络流主动 close

### US5: 文档与实现一致
- [ ] README 工具数/测试数/结构树与实际对齐
- [ ] TECHNICAL_DOC 行数/测试数对齐
- [ ] Python 版本口径统一

## 非功能性
- 兼容性：Windows 一等公民，torch 先于 PyQt6
- 稳定性：关窗无 QThread 崩溃
- 可维护：死代码标注或删除

## 成功指标
- pytest 全绿（含新增测试）
- mypy/ruff 0
- Docker 形态 ADVANCED_FEATURES_AVAILABLE=True（seaborn 装上）

## Out of Scope
- main_window God Object 拆分（独立批次）
- 真实付费 API 调用
- 高并发分布式基础设施（单机桌面不适用）
