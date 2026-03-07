import { copyFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const outDir = join(process.cwd(), "dist", "hotel-insights", "browser");

const csr = join(outDir, "index.csr.html");
const idx = join(outDir, "index.html");

if (!existsSync(csr)) {
  console.error("No existe index.csr.html en:", csr);
  process.exit(1);
}

copyFileSync(csr, idx);
console.log("OK: index.csr.html -> index.html");
