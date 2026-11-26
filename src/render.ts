// Rendu d'un clip par le programme natif, lot par lot : images du média (`media.frames`) →
// calcul (`upscale` ou `interpolate`) → encodage (`media.encode.append`) → nouveau média.
// Indépendant de Bobine (`Host`) : testé avec un faux hôte.
import { fpsOf, frameCount, lots } from "./model";
import type { Estimate, Host, Rendered } from "./native";

export type Job = {
  kind: "upscale" | "interpolate";
  media: string;
  /** Plage du média lue (secondes). */
  range: { from: number; to: number };
  fps: { num: number; den: number };
  /** Taille des images envoyées au programme natif. */
  frame: { width: number; height: number };
  /** Taille de sortie (agrandissement) ; celle des images pour les images intermédiaires. */
  out: { width: number; height: number };
  model: "photo" | "anime";
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

/** Images sources par appel : ~45 s de calcul et 256 Mo écrits au plus. */
export function framesPerCall(job: Job, secondsPerFrame: number): number {
  const outPerSource = job.kind === "interpolate" ? 2 : 1;
  const bytes = job.out.width * job.out.height * 4 * outPerSource;
  const byTime = Math.floor(CALL_TARGET_S / Math.max(secondsPerFrame, 1e-3));
  const byBytes = Math.floor(CALL_OUT_BYTES / bytes);
  return Math.max(1, Math.min(240, byTime, byBytes));
}

/** Cadence du nouveau média : doublée pour les images intermédiaires. */
export function outputFps(job: Job): { num: number; den: number } {
  return job.kind === "interpolate" ? { num: job.fps.num * 2, den: job.fps.den } : job.fps;
}

/**
 * Rend le clip et importe le résultat : rend l'id du nouveau média et l'instant (secondes, temps
 * de l'ancien média) de sa première image. En cas d'erreur, la session d'encodage est annulée.
 */
export async function render(
  host: Host,
  job: Job,
  secondsPerFrame: number,
): Promise<{ media: string; first: number }> {
  const fps = fpsOf(job.fps) || 30;
  const total = frameCount(job.range, fps);
  const perCall = framesPerCall(job, secondsPerFrame);
  const timeout = Math.min(MAX_TIMEOUT_S, Math.ceil(perCall * secondsPerFrame * 3 + 60));
  const { session } = await host.encodeBegin({
    width: job.out.width,
    height: job.out.height,
    fps: outputFps(job),
    name: job.name,
  });
  try {
    let first: number | null = null;
    let after: number | null = null;
    let done = 0;
    const plan = lots(job.range, fps, job.frame);
    for (const [index, lot] of plan.entries()) {
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
            : { last }),
        };
        const result: Rendered = await host.call<Rendered>(job.kind, params, timeout);
        done += count;
        if (result.first_time !== null && first === null) first = result.first_time;
        if (result.last_time !== null) after = result.last_time;
        if (result.count > 0) await host.encodeAppend({ session, raw: result.raw });
      }
    }
    if (first === null) throw new Error("no_frames");
    const { media } = await host.encodeFinish({ session });
    return { media, first };
  } catch (error) {
    await host.encodeCancel({ session }).catch(() => null);
    throw error;
  }
}
