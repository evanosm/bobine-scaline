// Lecture du projet et calculs du rendu (fonctions pures, testées) : clip choisi, plage du média
// lue par le clip, taille de sortie, découpage en lots, point d'entrée dans le nouveau média.
import {
  type Clip,
  FLICKS_PER_SECOND,
  type MediaInfo,
  type Project,
  type Rational,
  type Sequence,
} from "@bobine/plugin-sdk";

export type Found = { clip: Clip; trackIndex: number };

/** Le clip `id` de la séquence, avec l'index de sa piste. */
export function findClip(sequence: Sequence, id: string): Found | null {
  for (const [trackIndex, track] of sequence.tracks.entries()) {
    const clip = track.clips.find((c) => c.id === id);
    if (clip) return { clip, trackIndex };
  }
  return null;
}

function isVideoClip(sequence: Sequence, found: Found): boolean {
  return sequence.tracks[found.trackIndex]?.kind === "video" && found.clip.source.type === "media";
}

/** Le clip vidéo sélectionné, sinon le plus haut sous la tête de lecture (`at` en flicks). */
export function chosenVideo(sequence: Sequence, selected: string[], at: number): Found | null {
  for (const id of selected) {
    const found = findClip(sequence, id);
    if (found && isVideoClip(sequence, found)) return found;
  }
  for (const [trackIndex, track] of sequence.tracks.entries()) {
    if (track.kind !== "video" || track.muted) continue;
    const clip = track.clips.find(
      (c) =>
        c.enabled &&
        c.source.type === "media" &&
        c.timeline_start <= at &&
        at < c.timeline_start + c.duration,
    );
    if (clip) return { clip, trackIndex };
  }
  return null;
}

export type Media = { id: string; path: string; info: MediaInfo };

/** Le média d'un clip et son analyse, ou `null` s'il n'est pas encore analysé. */
export function mediaOf(project: Project, clip: Clip): Media | null {
  if (clip.source.type !== "media") return null;
  const id = clip.source.id;
  const asset = project.media.find((m) => m.id === id) as
    | { id: string; path?: string; info?: MediaInfo | null }
    | undefined;
  const info = asset?.info;
  if (!asset || !info || info.width <= 0 || info.height <= 0) return null;
  return { id, path: typeof asset.path === "string" ? asset.path : "", info };
}

export function seconds(flicks: number): number {
  return flicks / FLICKS_PER_SECOND;
}

export function fpsOf(rate: Rational): number {
  return rate.den > 0 ? rate.num / rate.den : 0;
}

/** Plus grand côté des images que Bobine envoie au programme natif. */
export const MAX_FRAME_SIDE = 1920;

/** Taille des images reçues : la taille du média, réduite pour tenir dans 1920 × 1920. */
export function frameSize(info: { width: number; height: number }): {
  width: number;
  height: number;
} {
  const k = Math.min(1, MAX_FRAME_SIDE / info.width, MAX_FRAME_SIDE / info.height);
  return {
    width: Math.max(1, Math.round(info.width * k)),
    height: Math.max(1, Math.round(info.height * k)),
  };
}

/** Côté pair le plus proche (l'encodeur l'exige). */
function even(value: number): number {
  return Math.max(2, Math.round(value / 2) * 2);
}

/** Plafonds de `media.encode.begin`. */
const MAX_SIDE = 8192;
const MAX_AREA = 8192 * 4320;

/**
 * Taille de sortie de l'agrandissement : celle de la séquence (l'image y tient entière) si la
 * séquence est plus grande que les images, sinon `width` de large. Proportions gardées, côtés
 * pairs, dans les plafonds de l'encodeur. `null` si l'image est déjà à cette taille.
 */
export function upscaleSize(
  frame: { width: number; height: number },
  sequence: { width: number; height: number },
  width: number,
): { width: number; height: number; factor: number } | null {
  const fit = Math.min(sequence.width / frame.width, sequence.height / frame.height);
  let factor = fit > 1.01 ? fit : width / frame.width;
  factor = Math.min(
    factor,
    MAX_SIDE / frame.width,
    MAX_SIDE / frame.height,
    Math.sqrt(MAX_AREA / (frame.width * frame.height)),
  );
  if (factor <= 1.01) return null;
  const out = { width: even(frame.width * factor), height: even(frame.height * factor) };
  // L'arrondi pair peut dépasser d'un pixel : on reste dans les plafonds.
  while (out.width > MAX_SIDE || out.width * out.height > MAX_AREA) out.width -= 2;
  while (out.height > MAX_SIDE || out.width * out.height > MAX_AREA) out.height -= 2;
  return { ...out, factor };
}

/**
 * Plage du média lue par le clip, en secondes (temps du média), élargie d'une image de chaque
 * côté : l'image affichée au point d'entrée commence avant lui, et l'arrondi des instants ne doit
 * pas couper la dernière.
 */
export function sourceRange(clip: Clip, info: MediaInfo): { from: number; to: number } {
  const fps = fpsOf(info.fps) || 30;
  const rate = clip.props.speed?.rate ?? 1;
  const start = seconds(clip.source_in);
  const end = start + seconds(clip.duration) * rate;
  const length = seconds(info.duration);
  return {
    from: Math.max(0, start - 1 / fps),
    to: length > 0 ? Math.min(length, end + 1 / fps) : end + 1 / fps,
  };
}

/** Volume d'un lot d'images demandé à Bobine (son plafond est 512 Mo). */
export const LOT_BYTES = 384 * 1024 * 1024;

/**
 * Lots `[from, to)` d'au plus `LOT_BYTES` d'images. Chaque lot reprend une image du précédent :
 * elle sert d'image précédente aux images intermédiaires, et le programme natif ne la rend pas
 * deux fois (`after`).
 */
export function lots(
  range: { from: number; to: number },
  fps: number,
  frame: { width: number; height: number },
): { from: number; to: number }[] {
  const perFrame = frame.width * frame.height * 4;
  const count = Math.max(2, Math.min(240, Math.floor(LOT_BYTES / perFrame)));
  const step = (count - 1) / fps;
  const out: { from: number; to: number }[] = [];
  for (let start = range.from; start < range.to - 1e-9; start += step) {
    const from = out.length === 0 ? start : start - 1 / fps;
    out.push({ from: Math.max(range.from, from), to: Math.min(range.to, start + step) });
  }
  return out;
}

/** Nombre d'images du média dans la plage (pour l'estimation et la progression). */
export function frameCount(range: { from: number; to: number }, fps: number): number {
  return Math.max(1, Math.round((range.to - range.from) * fps));
}

/**
 * Point d'entrée du clip dans le nouveau média (flicks) : le nouveau média commence à la première
 * image rendue, `first` secondes dans l'ancien.
 */
export function newSourceIn(clip: Clip, first: number): number {
  return Math.max(0, clip.source_in - Math.round(first * FLICKS_PER_SECOND));
}

/** Nom du fichier rendu : celui du média d'origine (sans dossier ni extension) et un suffixe. */
export function renderName(path: string, suffix: string): string {
  const base = path.split(/[\\/]/).pop() ?? "";
  const stem = base.includes(".") ? base.slice(0, base.lastIndexOf(".")) : base;
  return `${(stem || "clip").slice(0, 150)} ${suffix}`;
}
