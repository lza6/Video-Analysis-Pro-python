/**
 * 轻量 className 合并工具(与 website/src/lib/utils.ts 同源)。
 * 过滤 falsy、拍平嵌套数组、空格 join。
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
