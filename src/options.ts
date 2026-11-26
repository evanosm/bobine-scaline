// Réglages d'un rendu, préréglages et descriptif du résultat (fonctions pures, testées). Une même
// forme sert aux commandes (réglages de « Mes extensions ») et au panneau (clip sélectionné).
import type { Clip, MediaInfo, Rational, Sequence } from "@bobine/plugin-sdk";
import { fpsOf, frameSize, seconds, targetRate, upscaleTo } from "./model";

/** Ce que fait le rendu : cadence plus haute à vitesse égale, ralenti, ou agrandissement. */
export type Action = "framerate" | "slowmo" | "upscale";
export const ACTIONS: Action[] = ["framerate", "slowmo", "upscale"];

export type Options = {
  action: Action;
  /** `framerate` : cadence visée (images par seconde). */
  fps: number;
  /** `slowmo` : vitesse divisée par `factor` (1,5 à 8, par demi). */
  factor: number;
  /** `upscale` : largeur visée (pixels), la hauteur suit. */
  width: number;
  model: "photo" | "anime";
  /** Images intermédiaires : recalculer les images répétées (doublons d'enregistrement). */
  dedupe: boolean;
  /** Images intermédiaires : rien d'inventé entre deux plans. */
  cuts: boolean;
};

export const MAX_FPS = 240;
export const MIN_FACTOR = 1.5;
export const MAX_FACTOR = 8;
export const MAX_WIDTH = 7680;
/** Agrandissement du modèle : au-delà, l'image est seulement étirée (aucun détail en plus). */
export const MODEL_SCALE = 4;

/** Ce qui compte du clip et de sa séquence pour proposer et décrire un rendu. */
export type Facts = {
  /** Cadence du média (fraction et nombre). */
  rate: Rational;
  fps: number;
  /** Taille des images envoyées au programme natif. */
  frame: { width: number; height: number };
  sequence: { width: number; height: number; fps: Rational };
  /** Durée du clip sur la timeline et durée du média qu'il lit (vitesse comprise), en secondes. */
  clipSeconds: number;
  sourceSeconds: number;
};

export function factsOf(clip: Clip, info: MediaInfo, sequence: Sequence): Facts {
  const speed = clip.props.speed?.rate ?? 1;
  const clipSeconds = seconds(clip.duration);
  return {
    rate: info.fps,
    fps: fpsOf(info.fps),
    frame: frameSize(info),
    sequence: { width: sequence.width, height: sequence.height, fps: sequence.fps },
    clipSeconds,
    sourceSeconds: clipSeconds * speed,
  };
}

/** Largeur qui fait tenir l'image entière dans la séquence, si la séquence est plus grande. */
export function sequenceWidth(facts: Facts): number | null {
  const { frame, sequence } = facts;
  const fit = Math.min(sequence.width / frame.width, sequence.height / frame.height);
  return fit > 1.01 ? Math.round(frame.width * fit) : null;
}

/** Raccourcis de cadence : le double du clip, puis 60, 120, 144, 240 au-dessus de lui. */
export function fpsShortcuts(source: number): number[] {
  const all = [Math.round(source * 2), 60, 120, 144, 240].filter(
    (f) => f > source + 0.01 && f <= MAX_FPS,
  );
  return [...new Set(all)].sort((a, b) => a - b);
}

export type WidthShortcut = { id: "sequence" | "1440p" | "4k"; width: number };

/** Raccourcis de largeur : la séquence (si plus grande), 1440p, 4K, au-dessus de l'image. */
export function widthShortcuts(facts: Facts): WidthShortcut[] {
  const list: WidthShortcut[] = [];
  const fit = sequenceWidth(facts);
  if (fit) list.push({ id: "sequence", width: fit });
  for (const [id, width] of [
    ["1440p", 2560],
    ["4k", 3840],
  ] as const) {
    if (width > facts.frame.width * 1.01 && width !== fit) list.push({ id, width });
  }
  return list;
}

export type Blocker = "alreadyFps" | "tooFast" | "badFactor" | "alreadyBig" | "tooBig";

/** Le rendu est-il possible pour ce clip ? Sinon, la raison (clé de texte). */
export function blocker(o: Options, facts: Facts): Blocker | null {
  if (o.action === "framerate") {
    if (!(o.fps > facts.fps + 0.01)) return "alreadyFps";
    return o.fps <= MAX_FPS ? null : "tooFast";
  }
  if (o.action === "slowmo") {
    if (!(o.factor >= MIN_FACTOR && o.factor <= MAX_FACTOR)) return "badFactor";
    return facts.fps * o.factor <= MAX_FPS + 0.01 ? null : "tooFast";
  }
  if (!(o.width > facts.frame.width * 1.01)) return "alreadyBig";
  return o.width <= MAX_WIDTH ? null : "tooBig";
}

/** Ce que donnera le rendu : de quoi construire la tâche et l'expliquer dans le panneau. */
export type Outcome = {
  kind: "upscale" | "interpolate";
  /** Cadence du nouveau média (temps du média d'origine). */
  rate: Rational;
  out: { width: number; height: number };
  /** Images lues, images rendues, images calculées par le modèle. */
  sources: number;
  outputs: number;
  computed: number;
  /** Durée du clip après le rendu (le ralenti l'allonge). */
  clipSeconds: number;
  /** Taille approchée du fichier ProRes 422 (octets). */
  bytes: number;
};

/** Débit approché du ProRes 422 : ~2,4 bits par pixel et par image (147 Mbit/s en 1080p30). */
const PRORES_BITS_PER_PIXEL = 2.4;

export function outcome(o: Options, facts: Facts): Outcome {
  let rate = facts.rate;
  if (o.action === "framerate") rate = targetRate(o.fps, facts.sequence.fps);
  // Facteur au demi près : la fraction reste exacte (29,97 × 2,5 = 150000 / 2002).
  if (o.action === "slowmo") rate = { num: Math.round(rate.num * o.factor * 2), den: rate.den * 2 };
  const out =
    o.action === "upscale" ? (upscaleTo(facts.frame, o.width) ?? facts.frame) : facts.frame;
  const sources = Math.max(1, Math.round(facts.sourceSeconds * facts.fps));
  const outputs =
    o.action === "upscale" ? sources : Math.max(1, Math.round(facts.sourceSeconds * fpsOf(rate)));
  return {
    kind: o.action === "upscale" ? "upscale" : "interpolate",
    rate,
    out: { width: out.width, height: out.height },
    sources,
    outputs,
    computed: o.action === "upscale" ? sources : Math.max(0, outputs - sources),
    clipSeconds: o.action === "slowmo" ? facts.clipSeconds * o.factor : facts.clipSeconds,
    bytes: (out.width * out.height * outputs * PRORES_BITS_PER_PIXEL) / 8,
  };
}

export type PresetId = "double" | "fps120" | "upscale4k" | "slow4";

/** Préréglages : chacun pose l'action et sa valeur, le reste ne bouge pas. */
export const PRESETS: { id: PresetId; apply: (o: Options, facts: Facts) => Options }[] = [
  { id: "double", apply: (o, f) => ({ ...o, action: "framerate", fps: Math.round(f.fps * 2) }) },
  { id: "fps120", apply: (o) => ({ ...o, action: "framerate", fps: 120 }) },
  { id: "upscale4k", apply: (o) => ({ ...o, action: "upscale", width: 3840 }) },
  { id: "slow4", apply: (o) => ({ ...o, action: "slowmo", factor: 4 }) },
];

/** Le préréglage qui donne exactement ces réglages, sinon `custom` (« Personnalisé »). */
export function presetOf(o: Options, facts: Facts): PresetId | "custom" {
  const same = (a: Options, b: Options) =>
    a.action === b.action &&
    (a.action !== "framerate" || a.fps === b.fps) &&
    (a.action !== "slowmo" || a.factor === b.factor) &&
    (a.action !== "upscale" || a.width === b.width);
  return PRESETS.find((p) => same(p.apply(o, facts), o))?.id ?? "custom";
}

/** Réglages de départ pour un clip, d'après ceux de « Mes extensions ». */
export function initialOptions(
  settings: { model: "photo" | "anime"; factor: number; targetFps: number; width: number },
  facts: Facts,
): Options {
  return {
    action: "framerate",
    fps: settings.targetFps > facts.fps + 0.01 ? settings.targetFps : Math.round(facts.fps * 2),
    factor: clampFactor(settings.factor),
    width: sequenceWidth(facts) ?? settings.width,
    model: settings.model,
    dedupe: true,
    cuts: true,
  };
}

/** Facteur de ralenti au demi près, entre 1,5 et 8. */
export function clampFactor(value: number): number {
  const half = Math.round(value * 2) / 2;
  return Math.min(MAX_FACTOR, Math.max(MIN_FACTOR, Number.isFinite(half) ? half : 2));
}

/** Le rendu en cours, tel que les panneaux le montrent (même sans clip sélectionné). */
export type Job = {
  /** Clip rendu et nom de son fichier. */
  clip: string;
  name: string;
  /** Ce que fait le rendu (« Rendre à 60 im./s »). */
  label: string;
  /** 0 à 1 ; `null` tant que le calcul n'a pas commencé (mesure du GPU, chargement du modèle). */
  progress: number | null;
  /** Début (millisecondes, horloge du plugin) : le panneau en tire le temps restant. */
  started: number;
  /** Annulation demandée, en attente de l'arrêt du programme natif. */
  cancelling: boolean;
};

/** Messages entre le panneau et la logique (`bobine.post`). */
export type PanelMessage =
  | { type: "hello" }
  | { type: "run"; clip: string; options: Options }
  | { type: "cancel" }
  | { type: "status"; busy: boolean; job: Job | null };

export function asPanelMessage(data: unknown): PanelMessage | null {
  if (typeof data !== "object" || data === null) return null;
  const m = data as Record<string, unknown>;
  if (m.type === "hello") return { type: "hello" };
  if (m.type === "cancel") return { type: "cancel" };
  if (m.type === "status" && typeof m.busy === "boolean") {
    return { type: "status", busy: m.busy, job: isJob(m.job) ? m.job : null };
  }
  if (m.type === "run" && typeof m.clip === "string" && isOptions(m.options)) {
    return { type: "run", clip: m.clip, options: m.options };
  }
  return null;
}

function isJob(value: unknown): value is Job {
  if (typeof value !== "object" || value === null) return false;
  const j = value as Record<string, unknown>;
  return (
    typeof j.clip === "string" &&
    typeof j.name === "string" &&
    typeof j.label === "string" &&
    (j.progress === null || (typeof j.progress === "number" && Number.isFinite(j.progress))) &&
    typeof j.started === "number" &&
    typeof j.cancelling === "boolean"
  );
}

/** Temps restant (secondes) d'après le temps écoulé et la progression ; `null` trop tôt pour le dire. */
export function remaining(job: Job, now: number): number | null {
  if (job.progress === null || job.progress < 0.02) return null;
  const elapsed = (now - job.started) / 1000;
  return (elapsed * (1 - job.progress)) / job.progress;
}

function isOptions(value: unknown): value is Options {
  if (typeof value !== "object" || value === null) return false;
  const o = value as Record<string, unknown>;
  const number = (v: unknown, min: number, max: number) =>
    typeof v === "number" && Number.isFinite(v) && v >= min && v <= max;
  return (
    ACTIONS.includes(o.action as Action) &&
    number(o.fps, 1, MAX_FPS) &&
    number(o.factor, MIN_FACTOR, MAX_FACTOR) &&
    number(o.width, 2, MAX_WIDTH) &&
    (o.model === "photo" || o.model === "anime") &&
    typeof o.dedupe === "boolean" &&
    typeof o.cuts === "boolean"
  );
}
