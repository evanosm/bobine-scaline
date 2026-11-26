// Rendu d'un clip par le programme natif, lot par lot : images du média (`media.frames`) →
// calcul (`upscale` ou `interpolate`) → encodage (`media.encode.append`) → nouveau média.
// Indépendant de Bobine (`Host`) : testé avec un faux hôte.
import { BobineError } from "@bobine/plugin-sdk";
import { fpsOf, frameCount, lots } from "./model";
import type { Estimate, Host, Rendered } from "./native";

type Rate = { num: number; den: number };

export type Job = {
  kind: "upscale" | "interpolate";
  media: string;
  /** Plage du média lue (secondes). */
  range: { from: number; to: number };
  /** Cadence du média. */
  fps: Rate;
  /**
   * Cadence du nouveau média, en temps du média d'origine : celle du média pour l'agrandissement,
   * plus haute pour les images intermédiaires (×2 à ×8 pour un ralenti, 60 ou 120 pour la cadence).
   */
  rate: Rate;
  /** Taille des images envoyées au programme natif. */
  frame: { width: number; height: number };
  /** Taille de sortie (agrandissement) ; celle des images pour les images intermédiaires. */
  out: { width: number; height: number };
  model: "photo" | "anime";
  /** Images intermédiaires : recalculer les doublons, respecter les coupes (vrai par défaut). */
  dedupe?: boolean;
  cuts?: boolean;
  name: string;
};

/** Délai maximal d'un appel au programme natif accepté par Bobine (secondes). */
const MAX_TIMEOUT_S = 600;
/** Durée visée d'un appel : assez court pour la barre de progression et le délai. */
const CALL_TARGET_S = 45;
/** Volume des images écrites par un appel (lues puis supprimées par `media.encode.append`). */
const CALL_OUT_BYTES = 256 * 1024 * 1024;

/** Mesure la durée d'une image sur ce GPU. */
export function estimate(host: Host, job: Job): Promise<Estimate> {
  return host.call<Estimate>(
    "estimate",
    { task: job.kind, model: job.model, width: job.frame.width, height: job.frame.height },
    MAX_TIMEOUT_S,
  );
}

/** Images de sortie par image source (1 pour l'agrandissement). */
export function outPerSource(job: Pick<Job, "fps" | "rate">): number {
  return Math.max(1, fpsOf(job.rate) / (fpsOf(job.fps) || 30));
}

/**
 * Calcul par image source, d'après la mesure d'UNE image (`estimate`) : une image intermédiaire
 * par sortie qui ne tombe pas sur une image source (×2 : 1, ×4 : 3, 24 → 60 : 1,5).
 */
export function secondsPerSource(
  job: Pick<Job, "kind" | "fps" | "rate">,
  secondsPerFrame: number,
): number {
  return job.kind === "interpolate"
    ? secondsPerFrame * Math.max(1, outPerSource(job) - 1)
    : secondsPerFrame;
}

/** Images sources par appel : ~45 s de calcul et 256 Mo écrits au plus. */
export function framesPerCall(job: Job, secondsPerFrame: number): number {
  const bytes = job.out.width * job.out.height * 4 * Math.ceil(outPerSource(job));
  const byTime = Math.floor(CALL_TARGET_S / Math.max(secondsPerSource(job, secondsPerFrame), 1e-3));
  const byBytes = Math.floor(CALL_OUT_BYTES / bytes);
  return Math.max(1, Math.min(240, byTime, byBytes));
}

/**
 * Rend le clip et importe le résultat : rend l'id du nouveau média et l'instant (secondes, temps
 * de l'ancien média) de sa première image. En cas d'erreur, la session d'encodage est annulée.
 * `signal.cancelled` (bouton Annuler) arrête le rendu à l'étape suivante avec l'erreur `cancelled`.
 */
export async function render(
  host: Host,
  job: Job,
  secondsPerFrame: number,
  signal: { cancelled: boolean } = { cancelled: false },
): Promise<{ media: string; first: number }> {
  const fps = fpsOf(job.fps) || 30;
  const total = frameCount(job.range, fps);
  const perCall = framesPerCall(job, secondsPerFrame);
  const timeout = Math.min(
    MAX_TIMEOUT_S,
    Math.ceil(perCall * secondsPerSource(job, secondsPerFrame) * 3 + 60),
  );
  const check = () => {
    if (signal.cancelled) throw new BobineError("cancelled", "the render was cancelled");
  };
  const { session } = await host.encodeBegin({
    width: job.out.width,
    height: job.out.height,
    fps: job.rate,
    name: job.name,
  });
  try {
    let first: number | null = null;
    let after: number | null = null;
    let done = 0;
    const plan = lots(job.range, fps, job.frame);
    for (const [index, lot] of plan.entries()) {
      check();
      const frames = await host.frames({
        media: job.media,
        from: lot.from,
        to: lot.to,
        every_frame: true,
        width: job.frame.width,
        height: job.frame.height,
        format: "rgba",
      });
      for (let start = 0; start < frames.count; start += perCall) {
        const count = Math.min(perCall, frames.count - start);
        const last = index === plan.length - 1 && start + count >= frames.count;
        // Part de la tâche entière (les images en recouvrement comptent : bornée à 1).
        const base = Math.min(1, done / total);
        const params: Record<string, unknown> = {
          frames: frames.header,
          first: start,
          count,
          after,
          progress: [base, Math.min(count / total, 1 - base)],
          ...(job.kind === "upscale"
            ? { model: job.model, width: job.out.width, height: job.out.height }
            : {
                fps: job.rate,
                origin: first,
                last,
                dedupe: job.dedupe ?? true,
                cuts: job.cuts ?? true,
              }),
        };
        check();
        const result: Rendered = await host.call<Rendered>(job.kind, params, timeout);
        check();
        done += count;
        if (result.first_time !== null && first === null) first = result.first_time;
        if (result.last_time !== null) after = result.last_time;
        if (result.count > 0) await host.encodeAppend({ session, raw: result.raw });
      }
    }
    check();
    if (first === null) throw new Error("no_frames");
    const { media } = await host.encodeFinish({ session });
    return { media, first };
  } catch (error) {
    await host.encodeCancel({ session }).catch(() => null);
    throw error;
  }
}
