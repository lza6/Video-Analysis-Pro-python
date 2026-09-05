import { Placeholder } from "@/components/shell/Placeholder";

export default function SkillsPage() {
  return (
    <Placeholder
      title="Skills 管理"
      desc="管理本地沉淀的分析技能(如稀疏走廊/密集人群 skill),启用/禁用/导入。"
      origin="src/ui/skills_manager_tab.py + src/skills + skill_generator.py"
      capabilities={[
        "skills 列表 + 详情(名称/描述/触发词/路径)",
        "启用/禁用切换(写 config/skills_state.json)",
        "导入外部 skill 文件夹",
        "Agent 按场景自动选 skill 或重写",
      ]}
    />
  );
}
