import { AppShell } from "@/components/shell/AppShell";
import { StatusBar } from "@/components/shell/StatusBar";

/**
 * 应用外壳布局:侧边导航 + 主内容 + 底部状态条。
 * 所有业务页面共享此 shell(对应此前评审"把 Header/Footer 移进 layout"的修复)。
 */
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <AppShell footer={<StatusBar />}>{children}</AppShell>
  );
}
