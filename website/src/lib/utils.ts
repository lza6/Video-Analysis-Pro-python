/**
 * 轻量 className 合并工具（不依赖 clsx / tailwind-merge，避免改 package.json）。
 * 行为：过滤 falsy、拍平嵌套数组、空格 join、去重首尾空格。
 */
export type ClassValue = string | false | null | undefined | ClassValue[];

function pushClass(out: string[], input: ClassValue): void {
  if (Array.isArray(input)) {
    for (const item of input) pushClass(out, item);
    return;
  }
  if (input) out.push(input);
}

export function cn(...inputs: ClassValue[]): string {
  const out: string[] = [];
  for (const input of inputs) pushClass(out, input);
  return out.join(" ").trim();
}
