import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import sharp from "sharp";

const OUT = "docs/screenshots";
const OPT = "docs/screenshots/opt";
mkdirSync(OUT, { recursive: true });
mkdirSync(OPT, { recursive: true });
mkdirSync(join(OPT, "full"), { recursive: true });

const URL = process.env.URL || "http://localhost:3000/";

const VIEWS = [
  { name: "desktop-1440", width: 1440, height: 900 },
  { name: "tablet-768", width: 768, height: 1024 },
  { name: "mobile-375", width: 375, height: 812 },
];

const SECTIONS = [
  { id: "top", label: "hero" },
  { id: "features", label: "features" },
  { id: "how", label: "how" },
  { id: "download", label: "download" },
  { id: "faq", label: "faq" },
];

const chromiumExe = join(
  process.env.LOCALAPPDATA || "",
  "ms-playwright",
  "chromium-1234",
  "chrome-win64",
  "chrome.exe",
);

async function optimize(src, dst) {
  await sharp(src)
    .resize({ width: 1600, withoutEnlargement: true })
    .png({ quality: 82, compressionLevel: 9 })
    .toFile(dst);
}

(async () => {
  const browser = await chromium.launch({ executablePath: chromiumExe });

  for (const v of VIEWS) {
    const ctx = await browser.newContext({
      viewport: { width: v.width, height: v.height },
      deviceScaleFactor: 2,
    });
    const page = await ctx.newPage();

    await page.goto(URL, { waitUntil: "networkidle", timeout: 60000 });
    await page.waitForTimeout(2500);

    if (v.name !== "desktop-1440") {
      const raw = join(OUT, "full", `${v.name}.png`);
      await page.screenshot({ path: raw, fullPage: true });
      await optimize(raw, join(OPT, "full", `${v.name}.png`));
    }

    if (v.name === "desktop-1440") {
      for (const s of SECTIONS) {
        const el = page.locator(`#${s.id}`);
        if (await el.count()) {
          await el.scrollIntoViewIfNeeded();
          await page.waitForTimeout(900);
          const raw = join(OUT, `section-${s.label}.png`);
          await el.screenshot({ path: raw });
          await optimize(raw, join(OPT, `section-${s.label}.png`));
        }
      }
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(500);
      const raw = join(OUT, "full", `${v.name}-hero-top.png`);
      await page.screenshot({ path: raw });
      await optimize(raw, join(OPT, "full", `${v.name}-hero-top.png`));
    }
    await ctx.close();
  }

  await browser.close();
  console.log("✓ Screenshots captured + optimized → docs/screenshots/opt/");
})();
