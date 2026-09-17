/**
 * 工具审批前端映射（v10.5.0 P1-8）。
 *
 * 单一事实来源：后端工具注册表(wiring_baseline)里所有写/危险写工具都必须
 * 有中文名 + 后果一句话说明，否则审批弹窗降级为裸工具名（小白不可读）。
 * 守护测试: tests/test_approval_maps.py 动态枚举比对，新增工具忘映射即红。
 */
export const APPROVAL_TOOL_CN: Record<string, string> = {
  delete_history: "删除历史记录",
  delete_video: "删除视频",
  highlight_cut: "剪辑落盘(高光片段导出)",
  trigger_batch: "启动批量任务",
  start_rtsp_monitor: "启动监控流",
  generate_skill: "生成技能",
  make_subtitle: "生成字幕文件",
  make_short_video: "生成竖屏短视频",
  make_voiceover: "生成配音",
  create_cut_clip: "剪辑视频片段",
  web_browser_trigger: "浏览器点击",
  web_browser_update: "浏览器填写表单",
  cdp_evaluate: "在页面执行脚本(任意 JS)",
  cdp_eval_write: "在页面执行写脚本",
  send_message: "发送消息",
};

export const APPROVAL_CONSEQUENCE: Record<string, string> = {
  delete_history: "将删除该条分析历史记录,删除后不可恢复。",
  delete_video: "将删除视频文件,删除后不可恢复。",
  highlight_cut: "将把选中的片段剪辑并写入磁盘。",
  trigger_batch: "将启动批量处理任务,占用 CPU/GPU 资源。",
  start_rtsp_monitor: "将连接监控摄像头并开始持续分析。",
  make_subtitle: "将在磁盘上写入一个字幕文件(SRT/VTT)。",
  make_short_video: "将在磁盘上写入一个 9:16 短视频文件。",
  make_voiceover: "将在磁盘上写入一个配音音频文件。",
  create_cut_clip: "将在磁盘上写入剪辑后的视频片段。",
  web_browser_trigger: "将在网页上执行一次点击操作。",
  web_browser_update: "将在网页表单中填写内容。",
  cdp_evaluate: "将在网页里执行任意 JS 脚本 —— 请确认你信任该操作。",
  cdp_eval_write: "将在网页里执行写操作脚本。",
  send_message: "将通过消息渠道向外发送内容。",
};