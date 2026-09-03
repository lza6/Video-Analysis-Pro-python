import sharp from "sharp";
import { readdirSync, mkdirSync, statSync, existsSync } from "node:fs";
import { join, basename } from "node:path";

const SRC = "docs/screenshots";
const OUT = "docs/screenshots/opt";
mkdirSync(OUT, { recursive: true });
mkdirSync(join(OUT, "full"), { recursive: true });

function walk(dir) {
  const out = [];
  for (const f of readdirSync(dir)) {
    const p = join(dir, f);
    const s = statSync(p);
    if (s.isDirectory()) out.push(...walk(p));
    else if (/\.png$/i.test(f)) out.push(p);
  }
  return out;
}

const files = walk(SRC).filter((f) => !f.includes("/opt/"));
let totalSaved = 0;

for (const f of files) {
  const rel = f.slice(SRC.length + 1);
  const dst = join(OUT, rel);
  mkdirSync(join(dst, ".."), { recursive: true });
  const before = statSync(f).size;
  await sharp(f)
    .resize({ width: 1600, withoutEnlargement: true })
    .png({ quality: 82, compressionLevel: 9 })
    .toFile(dst);
  const after = statSync(dst).size;
  totalSaved += before - after;
  console.log(
    `${rel}: ${Math.round(before / 1024)}KB → ${Math.round(after / 1024)}KB`,
  );
}

console.log(`✓ Optimized ${files.length} images, saved ~${Math.round(totalSaved / 1024 / 1024)}MB`);
