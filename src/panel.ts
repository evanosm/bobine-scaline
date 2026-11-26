// Panneau de Scaline (onglet de la bibliothèque) : le clip vidéo SÉLECTIONNÉ (image sous la tête
// de lecture, nom, faits), un préréglage, l'action et ses réglages libres, une partie « Avancé » et
// le détail du résultat (images, durée, temps mesuré sur le GPU, fichier). « Rendre » envoie le
// rendu à la logique (`bobine.post`), qui le mène même si le panneau se ferme ; la carte « En
// cours » le suit et l'annule, même sans clip sélectionné.
import {
  activeSequence,
  applyTheme,
  bobine,
  type Project,
  type Selection,
} from "@bobine/plugin-sdk";
import {
  formatBytes,
  formatCount,
  formatDuration,
  formatNumber,
  LANG,
  say,
  type TextKey,
} from "./i18n";
import {
  type Found,
  fileName,
  fpsOf,
  type Media,
  mediaOf,
  mediaTimeAt,
  selectedVideo,
  targetRate,
} from "./model";
import type { Estimate } from "./native";
import {
  ACTIONS,
  type Action,
  asPanelMessage,
  blocker,
  clampFactor,
  type Facts,
  factsOf,
  fpsShortcuts,
  initialOptions,
  type Job,
  MAX_FPS,
  MODEL_SCALE,
  type Options,
  outcome,
  PRESETS,
  type PresetId,
  presetOf,
  remaining,
  sequenceWidth,
  widthShortcuts,
} from "./options";
import { secondsPerSource } from "./render";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const panel = $<HTMLElement>("panel");
document.documentElement.lang = LANG;

// ── Textes fixes ─────────────────────────────────────────────────────────────────────────────
const LABELS: [string, TextKey][] = [
  ["preset-label", "fieldPreset"],
  ["action-label", "fieldAction"],
  ["fps-label", "fieldFps"],
  ["fps-unit", "unitFps"],
  ["factor-label", "fieldFactor"],
  ["width-label", "fieldWidth"],
  ["width-unit", "unitPx"],
  ["model-label", "fieldModel"],
  ["advanced-label", "advanced"],
  ["dedupe-label", "dedupeLabel"],
  ["dedupe-hint", "dedupeHint"],
  ["cuts-label", "cutsLabel"],
  ["cuts-hint", "cutsHint"],
  ["engine-label", "engine"],
  ["result-title", "result"],
  ["job-title", "jobTitle"],
  ["job-cancel", "jobCancel"],
];
for (const [id, key] of LABELS) $(id).textContent = say(key);

const PRESET_TEXT: Record<PresetId | "custom", TextKey> = {
  double: "presetDouble",
  fps120: "presetFps120",
  upscale4k: "presetUpscale4k",
  slow4: "presetSlow4",
  custom: "presetCustom",
};
const ACTION_TEXT: Record<Action, TextKey> = {
  framerate: "actionFramerate",
  slowmo: "actionSlowmo",
  upscale: "actionUpscale",
};

function fillSelect(select: HTMLSelectElement, items: [string, string][]): void {
  select.replaceChildren(
    ...items.map(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    }),
  );
}

const presetEl = $<HTMLSelectElement>("preset");
const actionEl = $<HTMLSelectElement>("action");
const modelEl = $<HTMLSelectElement>("model");
fillSelect(presetEl, [
  ["custom", say("presetCustom")],
  ...PRESETS.map((p): [string, string] => [p.id, say(PRESET_TEXT[p.id])]),
]);
fillSelect(
  actionEl,
  ACTIONS.map((a) => [a, say(ACTION_TEXT[a])]),
);
fillSelect(modelEl, [
  ["photo", say("modelPhoto")],
  ["anime", say("modelAnime")],
]);

const fpsEl = $<HTMLInputElement>("fps");
const factorEl = $<HTMLInputElement>("factor");
const widthEl = $<HTMLInputElement>("width");
const dedupeEl = $<HTMLInputElement>("dedupe");
const cutsEl = $<HTMLInputElement>("cuts");
const runEl = $<HTMLButtonElement>("run");
const hintEl = $<HTMLElement>("hint");
const canvas = $<HTMLCanvasElement>("thumb");

// ── État ─────────────────────────────────────────────────────────────────────────────────────
type Current = { found: Found; media: Media; facts: Facts };

let project: Project | null = null;
let selection: Selection = { clips: [], playhead: 0 };
let current: Current | null = null;
let options: Options | null = null;
let busy = false;
let job: Job | null = null;
let base = { model: "photo" as "photo" | "anime", factor: 2, targetFps: 60, width: 3840 };

/** Le clip vidéo sélectionné ; sans sélection, le panneau le demande (pas de repli). */
function pick(): Current | { reason: TextKey } {
  const sequence = activeSequence(project);
  if (!project || !sequence) return { reason: "panelEmpty" };
  const found = selectedVideo(sequence, selection.clips);
  if (!found) return { reason: "panelEmpty" };
  const media = mediaOf(project, found.clip);
  if (!media) return { reason: "panelNotReady" };
  if (media.info.kind !== "video") return { reason: "notVideo" };
  return { found, media, facts: factsOf(found.clip, media.info, sequence) };
}

/** D'un clip à l'autre, les réglages restent ; une valeur devenue sans effet est remplacée. */
function adapt(o: Options, facts: Facts): Options {
  const next = { ...o };
  if (next.fps <= facts.fps + 0.01) next.fps = Math.min(MAX_FPS, Math.round(facts.fps * 2));
  if (next.width <= facts.frame.width * 1.01) next.width = sequenceWidth(facts) ?? 3840;
  return next;
}

function update(): void {
  const found = pick();
  const note = $<HTMLElement>("note");
  if ("reason" in found) {
    current = null;
    panel.dataset.state = "empty";
    note.textContent = say(found.reason);
    canvas.classList.remove("shown");
    lastThumb = "";
  } else {
    const changed = current?.found.clip.id !== found.found.clip.id;
    current = found;
    panel.dataset.state = "clip";
    note.textContent = "";
    const name = fileName(found.media.path) || found.found.clip.id;
    const nameEl = $<HTMLElement>("name");
    nameEl.textContent = name;
    nameEl.title = name;
    $("meta").textContent = say("clipMeta", {
      size: `${found.media.info.width} × ${found.media.info.height}`,
      fps: formatNumber(found.facts.fps),
      duration: formatNumber(found.facts.clipSeconds, 1),
    });
    if (!options) options = initialOptions(base, found.facts);
    else if (changed) options = adapt(options, found.facts);
    // Autre clip : on le montre depuis le haut (son image, puis ses réglages).
    if (changed) window.scrollTo(0, 0);
    scheduleThumb();
  }
  draw();
}

// ── Image du clip ────────────────────────────────────────────────────────────────────────────
let lastThumb = "";
let thumbTimer: ReturnType<typeof setTimeout> | undefined;

/** Image sous la tête de lecture (ou au milieu du clip), demandée après 200 ms de calme. */
function scheduleThumb(): void {
  clearTimeout(thumbTimer);
  thumbTimer = setTimeout(() => void loadThumb(), 200);
}

async function loadThumb(): Promise<void> {
  if (!current) return;
  const media = current.media.id;
  // Un dixième de seconde près : le cache de vignettes de Bobine travaille au même pas.
  const time = Math.round(mediaTimeAt(current.found.clip, selection.playhead) * 10) / 10;
  const key = `${media}@${time}`;
  if (key === lastThumb) return;
  lastThumb = key;
  try {
    const thumb = await bobine.media.thumbnail({ media, time, height: 256 });
    if (key !== lastThumb) return;
    const bytes = Uint8ClampedArray.from(atob(thumb.rgba_base64), (c) => c.charCodeAt(0));
    canvas.width = thumb.width;
    canvas.height = thumb.height;
    canvas.getContext("2d")?.putImageData(new ImageData(bytes, thumb.width, thumb.height), 0, 0);
    canvas.classList.add("shown");
  } catch {
    // Pas d'image (média absent, analyse en cours) : le cadre reste vide.
    canvas.classList.remove("shown");
  }
}

// ── Temps estimé : une mesure par moteur et taille d'image, gardée ────────────────────────────
const measures = new Map<string, Estimate | "pending">();
let measureTimer: ReturnType<typeof setTimeout> | undefined;

function measureKey(o: Options, facts: Facts): string {
  const task = o.action === "upscale" ? `upscale:${o.model}` : "interpolate";
  return `${task}:${facts.frame.width}x${facts.frame.height}`;
}

/** Mesure le GPU quand le panneau en a besoin (jamais pendant un rendu : un appel à la fois). */
function scheduleMeasure(): void {
  clearTimeout(measureTimer);
  if (!current || !options || busy) return;
  const o = options;
  const facts = current.facts;
  const key = measureKey(o, facts);
  if (measures.has(key)) return;
  measureTimer = setTimeout(() => {
    measures.set(key, "pending");
    bobine.native
      .call<Estimate>(
        "estimate",
        {
          task: o.action === "upscale" ? "upscale" : "interpolate",
          model: o.model,
          width: facts.frame.width,
          height: facts.frame.height,
        },
        { timeout: 120 },
      )
      .then((estimate) => measures.set(key, estimate))
      .catch(() => measures.delete(key))
      .finally(draw);
  }, 600);
}

// ── Dessin ───────────────────────────────────────────────────────────────────────────────────
const HINTS: Record<string, TextKey> = {
  alreadyFps: "hintAlreadyFps",
  tooFast: "hintTooFast",
  badFactor: "badFactor",
  alreadyBig: "hintAlreadyBig",
  tooBig: "tooBig",
};

function quick(
  id: string,
  items: { label: string; value: number }[],
  pressed: number,
  choose: (value: number) => void,
): void {
  $(id).replaceChildren(
    ...items.map((item) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = item.label;
      b.setAttribute("aria-pressed", String(item.value === pressed));
      b.addEventListener("click", () => choose(item.value));
      return b;
    }),
  );
}

/** Pose la valeur d'un champ, sauf s'il est en cours de saisie (le curseur ne saute pas). */
function setValue(input: HTMLInputElement, value: number): void {
  if (document.activeElement !== input) input.value = String(value);
}

function line(label: TextKey, value: string): HTMLElement[] {
  const dt = document.createElement("dt");
  dt.textContent = say(label);
  const dd = document.createElement("dd");
  dd.textContent = value;
  return [dt, dd];
}

const size = (w: number, h: number) => `${w} × ${h}`;

function draw(): void {
  drawJob();
  const o = options;
  const c = current;
  if (!o || !c) return;
  const facts = c.facts;
  const fps = formatNumber(facts.fps);
  const result = outcome(o, facts);
  const fps2 = formatNumber(fpsOf(result.rate));
  const interpolate = result.kind === "interpolate";

  presetEl.value = presetOf(o, facts);
  actionEl.value = o.action;
  modelEl.value = o.model;
  dedupeEl.checked = o.dedupe;
  cutsEl.checked = o.cuts;
  for (const group of document.querySelectorAll<HTMLElement>(".group")) {
    group.hidden = group.dataset.action !== o.action;
  }
  for (const r of document.querySelectorAll<HTMLElement>(".switch-row")) r.hidden = !interpolate;

  // Cadence.
  setValue(fpsEl, o.fps);
  quick(
    "fps-quick",
    fpsShortcuts(facts.fps).map((value) => ({ label: String(value), value })),
    o.fps,
    (value) => change({ ...o, fps: value }),
  );
  const ntsc = facts.sequence.fps.den === 1001;
  $("fps-note").textContent =
    say("noteFps", { ratio: formatNumber(o.fps / facts.fps, 2) }) +
    (ntsc
      ? say("noteNtsc", { fps: formatNumber(fpsOf(targetRate(o.fps, facts.sequence.fps))) })
      : "");

  // Ralenti.
  setValue(factorEl, o.factor);
  quick(
    "factor-quick",
    [2, 4, 8].map((value) => ({ label: `×${value}`, value })),
    o.factor,
    (value) => change({ ...o, factor: value }),
  );
  $("factor-note").textContent = say("noteFactor", {
    fps,
    fps2: formatNumber(facts.fps * o.factor),
    from: formatNumber(facts.clipSeconds, 1),
    to: formatNumber(facts.clipSeconds * o.factor, 1),
  });

  // Agrandissement.
  setValue(widthEl, o.width);
  quick(
    "width-quick",
    widthShortcuts(facts).map((s) => ({
      label: s.id === "sequence" ? say("quickSequence") : s.id === "4k" ? "4K" : "1440p",
      value: s.width,
    })),
    o.width,
    (value) => change({ ...o, width: value }),
  );
  const ratio = result.out.width / facts.frame.width;
  $("width-note").textContent =
    say("noteWidth", {
      size: size(result.out.width, result.out.height),
      ratio: formatNumber(ratio, 2),
    }) + (ratio > MODEL_SCALE + 0.01 ? say("noteStretch") : "");

  // Avancé : moteur.
  $("engine").textContent = interpolate
    ? say("engineRife")
    : say("engineEsrgan", { model: say(o.model === "anime" ? "modelAnime" : "modelPhoto") });

  // Résultat.
  const rows = [
    line(
      "resIn",
      say("resFrames", {
        size: size(facts.frame.width, facts.frame.height),
        fps,
        frames: formatCount(result.sources),
      }),
    ),
    line(
      "resOut",
      say("resFrames", {
        size: size(result.out.width, result.out.height),
        fps: fps2,
        frames: formatCount(result.outputs),
      }),
    ),
    line(
      "resLength",
      o.action === "slowmo"
        ? `${formatNumber(facts.clipSeconds, 1)} s → ${formatNumber(result.clipSeconds, 1)} s`
        : `${formatNumber(facts.clipSeconds, 1)} s`,
    ),
    line(
      "resWork",
      say(interpolate ? "workInterp" : "workUpscale", { n: formatCount(result.computed) }),
    ),
  ];
  const measure = measures.get(measureKey(o, facts));
  if (measure === "pending") rows.push(line("resTime", say("timeMeasuring")));
  else if (measure) {
    const work = { kind: result.kind, fps: facts.rate, rate: result.rate };
    const seconds = result.sources * secondsPerSource(work, measure.seconds_per_frame);
    rows.push(
      line(
        "resTime",
        say("timeValue", { time: formatDuration(seconds), gpu: measure.adapter }) +
          (measure.software ? say("timeSoftware") : ""),
      ),
    );
  }
  rows.push(line("resFile", say("fileValue", { size: formatBytes(result.bytes) })));
  $("result").replaceChildren(...rows.flat());

  // Bouton et raison d'un refus.
  const reason = blocker(o, facts);
  runEl.textContent =
    o.action === "upscale"
      ? say("runUpscale", { size: size(result.out.width, result.out.height) })
      : o.action === "slowmo"
        ? say("runSlowmo", { factor: formatNumber(o.factor) })
        : say("runFramerate", { fps: fps2 });
  hintEl.textContent = reason
    ? say(HINTS[reason] ?? "hintAlreadyBig", { fps })
    : busy
      ? say("running")
      : "";
  runEl.disabled = busy || reason !== null;
  const invalid: [HTMLInputElement, boolean][] = [
    [fpsEl, o.action === "framerate" && reason !== null],
    [factorEl, o.action === "slowmo" && reason !== null],
    [widthEl, o.action === "upscale" && reason !== null],
  ];
  for (const [input, bad] of invalid) input.parentElement?.toggleAttribute("data-invalid", bad);
  scheduleMeasure();
}

function change(next: Options): void {
  options = next;
  draw();
}

// ── Champs ───────────────────────────────────────────────────────────────────────────────────
presetEl.addEventListener("change", () => {
  const preset = PRESETS.find((p) => p.id === presetEl.value);
  if (preset && options && current) change(preset.apply(options, current.facts));
});
actionEl.addEventListener("change", () => {
  if (options) change({ ...options, action: actionEl.value as Action });
});
modelEl.addEventListener("change", () => {
  if (options) change({ ...options, model: modelEl.value === "anime" ? "anime" : "photo" });
});
dedupeEl.addEventListener("change", () => {
  if (options) change({ ...options, dedupe: dedupeEl.checked });
});
cutsEl.addEventListener("change", () => {
  if (options) change({ ...options, cuts: cutsEl.checked });
});

/** Champ numérique : chaque frappe met à jour le résultat ; la sortie du champ range la valeur. */
function numeric(
  input: HTMLInputElement,
  set: (o: Options, value: number) => Options,
  tidy: (value: number) => number,
): void {
  input.addEventListener("input", () => {
    const value = input.valueAsNumber;
    if (options && Number.isFinite(value)) change(set(options, value));
  });
  input.addEventListener("change", () => {
    const value = input.valueAsNumber;
    if (options && Number.isFinite(value)) options = set(options, tidy(value));
    // Valeur rangée (ou l'ancienne si le champ est vide) : `draw` la repose une fois sorti du champ.
    input.blur();
    draw();
  });
}
numeric(fpsEl, (o, v) => ({ ...o, fps: v }), Math.round);
numeric(factorEl, (o, v) => ({ ...o, factor: v }), clampFactor);
numeric(
  widthEl,
  (o, v) => ({ ...o, width: v }),
  (v) => Math.max(2, Math.round(v / 2) * 2),
);
$("form").addEventListener("submit", (e) => e.preventDefault());
// Chevron d'« Avancé » : l'attribut posé ici (le sélecteur `[open]` ne redessine pas toujours le
// pseudo-élément dans WebKit).
const advanced = $<HTMLDetailsElement>("advanced");
advanced.addEventListener("toggle", () => advanced.toggleAttribute("data-open", advanced.open));

runEl.addEventListener("click", () => {
  if (!current || !options || busy) return;
  bobine.post({ type: "run", clip: current.found.clip.id, options }).catch(() => {
    // Bobine antérieure à 0.4.2 : pas de messages entre panneau et logique.
    hintEl.textContent = say("hintOldBobine");
  });
});

// ── Rendu en cours ───────────────────────────────────────────────────────────────────────────
const jobEl = $<HTMLElement>("job");
const jobBar = $<HTMLElement>("job-bar");
const jobCancel = $<HTMLButtonElement>("job-cancel");
let jobTimer: ReturnType<typeof setInterval> | undefined;

/** Carte du rendu en cours, même sans clip sélectionné ; le temps restant suit chaque seconde. */
function drawJob(): void {
  jobEl.hidden = job === null;
  clearInterval(jobTimer);
  if (!job) return;
  jobTimer = setInterval(drawJob, 1000);
  const name = $<HTMLElement>("job-name");
  name.textContent = job.name;
  name.title = job.name;
  $("job-label").textContent = job.label;
  const known = job.progress !== null;
  jobBar.toggleAttribute("data-indeterminate", !known);
  const percent = Math.round((job.progress ?? 0) * 100);
  $("job-fill").style.transform = known ? `scaleX(${job.progress})` : "";
  if (known) jobBar.setAttribute("aria-valuenow", String(percent));
  else jobBar.removeAttribute("aria-valuenow");
  const left = remaining(job, Date.now());
  $("job-status").textContent = job.cancelling
    ? say("jobCancelling")
    : !known
      ? say("jobPreparing")
      : left === null
        ? say("jobPercent", { percent })
        : say("jobLeft", { percent, time: formatDuration(left) });
  jobCancel.disabled = job.cancelling;
}

jobCancel.addEventListener("click", () => {
  if (!job || job.cancelling) return;
  job = { ...job, cancelling: true };
  drawJob();
  void bobine.post({ type: "cancel" }).catch(() => undefined);
});

// ── Événements de Bobine ─────────────────────────────────────────────────────────────────────
bobine.on("theme", (theme) => applyTheme(theme));
bobine.on("selection", (next) => {
  selection = next;
  update();
});
bobine.on("project", () => {
  void bobine.project.get().then((p) => {
    project = p;
    update();
  });
});
bobine.on("message", (data) => {
  const message = asPanelMessage(data);
  if (message?.type !== "status") return;
  busy = message.busy;
  job = message.job;
  draw();
});

async function init(): Promise<void> {
  applyTheme(await bobine.theme.get());
  const s = await bobine.settings.get();
  base = {
    model: s.model === "animation" ? "anime" : "photo",
    factor: Number(s["slowmo-factor"]) || 2,
    targetFps: typeof s["target-fps"] === "number" ? s["target-fps"] : 60,
    width: typeof s.width === "number" ? s.width : 3840,
  };
  [project, selection] = await Promise.all([bobine.project.get(), bobine.selection.get()]);
  update();
  // La logique répond par son état (un rendu en cours ?).
  await bobine.post({ type: "hello" });
}

void init();
