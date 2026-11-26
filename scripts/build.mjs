// Assemble le dossier du plugin dans package/ (hors git) : c'est lui que Bobine charge en mode
// développeur et que `bobine-plugin check` / `pack` vérifient.
//   node scripts/build.mjs               logique JS + modèles + programme natif de cette machine
//   node scripts/build.mjs --no-worker   sans compiler (les binaires déjà dans package/bin restent)
//   node scripts/build.mjs --release     exige les quatre programmes natifs (paquet publié)
//
// Le programme natif est compilé dans $CARGO_TARGET_DIR (sinon worker/target), jamais dans
// package/. Les modèles viennent de models/ (scripts/fetch-models.sh) et sont vérifiés contre
// models/SHA256SUMS. package/bobine.json ne garde dans `native` que les plateformes dont le
// binaire est présent : un dossier de développement n'a que celui de sa machine.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "package");
const args = new Set(process.argv.slice(2));
const manifest = JSON.parse(readFileSync(join(root, "bobine.json"), "utf8"));

/** Plateforme de Bobine pour cette machine (`crates/extensions/src/native.rs::current_platform`). */
function currentPlatform() {
  const key = `${process.platform}-${process.arch}`;
  const known = {
    "darwin-arm64": "darwin-arm64",
    "darwin-x64": "darwin-x64",
    "linux-x64": "linux-x64",
    "win32-x64": "windows-x64",
  };
  return known[key] ?? null;
}

// 1. Logique : un script classique (Bobine le charge par un <script> simple, pas en module).
rmSync(join(out, "dist"), { recursive: true, force: true });
mkdirSync(join(out, "dist"), { recursive: true });
await build({
  entryPoints: [join(root, "src/main.ts")],
  bundle: true,
  format: "iife",
  target: "es2022",
  outfile: join(out, "dist/main.js"),
  logLevel: "warning",
});

// 2. Modèles, vérifiés contre leurs sommes SHA-256.
mkdirSync(join(out, "models"), { recursive: true });
const sums = readFileSync(join(root, "models/SHA256SUMS"), "utf8").trim().split("\n");
for (const line of sums) {
  const [sum, name] = line.trim().split(/\s+/);
  const source = join(root, "models", name);
  if (!existsSync(source)) throw new Error(`missing model ${name}: run scripts/fetch-models.sh`);
  const actual = createHash("sha256").update(readFileSync(source)).digest("hex");
  if (actual !== sum) throw new Error(`model ${name}: SHA-256 mismatch`);
  copyFileSync(source, join(out, "models", name));
}
copyFileSync(join(root, "models/SHA256SUMS"), join(out, "models/SHA256SUMS"));

// 3. Programme natif de cette machine.
if (!args.has("--no-worker")) {
  const platform = currentPlatform();
  const target = platform && manifest.native[platform];
  if (!target)
    throw new Error(`no native program declared for ${process.platform}-${process.arch}`);
  execFileSync(
    process.env.CARGO ?? "cargo",
    ["build", "--release", "--locked", "--manifest-path", join(root, "worker/Cargo.toml")],
    { stdio: "inherit" },
  );
  const targetDir = process.env.CARGO_TARGET_DIR ?? join(root, "worker/target");
  const exe = process.platform === "win32" ? "scaline-worker.exe" : "scaline-worker";
  mkdirSync(join(out, "bin"), { recursive: true });
  copyFileSync(join(targetDir, "release", exe), join(out, target));
}

// 4. Manifeste : seulement les plateformes dont le binaire est là.
const native = {};
const missing = [];
for (const [platform, path] of Object.entries(manifest.native)) {
  if (existsSync(join(out, path))) native[platform] = path;
  else missing.push(platform);
}
if (args.has("--release") && missing.length > 0) {
  throw new Error(`missing native programs for: ${missing.join(", ")}`);
}
writeFileSync(join(out, "bobine.json"), `${JSON.stringify({ ...manifest, native }, null, 2)}\n`);

// 5. Licences : celle de Scaline, celles des modèles et des bibliothèques compilées.
copyFileSync(join(root, "LICENSE"), join(out, "LICENSE"));
copyFileSync(join(root, "THIRD_PARTY_NOTICES.md"), join(out, "THIRD_PARTY_NOTICES.md"));
rmSync(join(out, "licenses"), { recursive: true, force: true });
cpSync(join(root, "licenses"), join(out, "licenses"), { recursive: true });
console.log(
  `package/ ready: ${Object.keys(native).join(", ") || "no native program"}${
    missing.length > 0 ? ` (missing: ${missing.join(", ")})` : ""
  }`,
);
