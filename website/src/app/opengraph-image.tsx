// OG 图生成器 — Video Analysis Pro 社交分享卡
// 通过 ImageResponse 在构建时生成 1200×630 的 OG 图
import { ImageResponse } from "next/og";
import { readFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export const alt = "Video Analysis Pro — 您的私人 AI 视频深度分析专家";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function OGImage() {
  let logoData: Buffer | null = null;
  try {
    logoData = await readFile(join(__dirname, "..", "..", "public", "logo.png"));
  } catch {
    /* fallback: 无 logo 也能生成 */
  }

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "80px",
          background:
            "linear-gradient(135deg, #151726 0%, #1d1f3a 50%, #2a1f4a 100%)",
          fontFamily: "sans-serif",
          position: "relative",
        }}
      >
        {/* 极光光斑 */}
        <div
          style={{
            position: "absolute",
            top: "-200px",
            right: "-100px",
            width: "600px",
            height: "600px",
            borderRadius: "9999px",
            background:
              "radial-gradient(circle, rgba(124,211,252,0.35), transparent 70%)",
            filter: "blur(60px)",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: "-150px",
            left: "-100px",
            width: "500px",
            height: "500px",
            borderRadius: "9999px",
            background:
              "radial-gradient(circle, rgba(167,139,250,0.3), transparent 70%)",
            filter: "blur(60px)",
          }}
        />

        {/* 顶部 logo + 品牌 */}
        <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
          {logoData && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={`data:image/png;base64,${logoData.toString("base64")}`}
              width={64}
              height={64}
              alt=""
              style={{ borderRadius: "12px" }}
            />
          )}
          <div style={{ color: "white", fontSize: 28, fontWeight: 700 }}>
            Video Analysis Pro
          </div>
        </div>

        {/* 主标题 */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <div
            style={{
              fontSize: 72,
              fontWeight: 900,
              color: "white",
              letterSpacing: "-0.02em",
              lineHeight: 1.05,
            }}
          >
            让 AI 替你「看」完整个视频
          </div>
          <div
            style={{
              fontSize: 30,
              color: "#9ca3c4",
              lineHeight: 1.4,
              maxWidth: "900px",
            }}
          >
            本地运行 · 隐私至上 · 开源 · YOLOv11 + Whisper + LLM
          </div>
        </div>

        {/* 底部标签 */}
        <div style={{ display: "flex", gap: "16px" }}>
          {["多模态分析", "智能 Agent", "自动集锦", "离线运行"].map((t) => (
            <div
              key={t}
              style={{
                padding: "12px 24px",
                borderRadius: "9999px",
                background: "rgba(255,255,255,0.1)",
                border: "1px solid rgba(255,255,255,0.15)",
                color: "white",
                fontSize: 24,
              }}
            >
              {t}
            </div>
          ))}
        </div>
      </div>
    ),
    { ...size },
  );
}
