// Logique de Scaline : tourne dans l'iframe isolée du plugin. Bobine décode les images
// (`media.frames`), le programme natif calcule sur le GPU (`native.call`), Bobine encode et
// importe le résultat (`media.encode.*`), puis le plugin propose de remplacer le média du clip
// (`ReplaceClipMedia`) ; l'utilisateur applique ou annule, l'original reste dans le projet.
// Les rendus partent des commandes (réglages de « Mes extensions ») ou du panneau (message `run`
// pour le clip qu'il montre) : c'est toujours la logique qui rend, le panneau peut se fermer.
import {
  activeSequence,
  BobineError,
  bobine,
  flicks,
  type PluginCommand,
  type Project,
  type Sequence,
} from "@bobine/plugin-sdk";
import { formatDuration, formatNumber, say, type TextKey } from "./i18n";
import {
  chosenVideo,
  type Found,
  fileName,
  findClip,
  fpsOf,
  frameCount,
  type Media,
  mediaOf,
  newSourceIn,
  renderName,
  sourceRange,
} from "./model";
import { bobineHost, replaceClipMedia } from "./native";
import {
  asPanelMessage,
  blocker,
  factsOf,
  type Options,
  outcome,
  type Job as PanelJob,
  sequenceWidth,
} from "./options";
import { estimate, type Job, render, secondsPerSource } from "./render";

export type Settings = {
  model: "photo" | "anime";
  width: number;
  /** Ralenti fluide : vitesse divisée par 2, 4 ou 8 (réglage), ou un demi près (panneau). */
  factor: number;
  /** Augmenter la cadence : images par seconde visées. */
  targetFps: number;
};

const host = bobineHost(bobine);

async function settings(): Promise<Settings> {
  const s = await bobine.settings.get();
  const factor = Number(s["slowmo-factor"]);
  return {
    model: s.model === "animation" ? "anime" : "photo",
    width: typeof s.width === "number" ? s.width : 3840,
    factor: factor === 4 || factor === 8 ? factor : 2,
    targetFps: typeof s["target-fps"] === "number" ? s["target-fps"] : 60,
  };
}

type Target = { project: Project; sequence: Sequence; found: Found; media: Media };

/**
 * Projet, séquence active, clip vidéo visé (`clip`, sinon le sélectionné ou celui sous la tête de
 * lecture) et son média ; ou `null` (l'utilisateur est prévenu).
 */
async function target(clip?: string): Promise<Target | null> {
  const [project, selection] = await Promise.all([bobine.project.get(), bobine.selection.get()]);
  const sequence = activeSequence(project);
  if (!project || !sequence) return null;
  const found = clip
    ? findClip(sequence, clip)
    : chosenVideo(sequence, selection.clips, flicks(selection.playhead));
  if (!found || sequence.tracks[found.trackIndex]?.kind !== "video") {
    await bobine.ui.notify("info", say("noVideo"));
    return null;
  }
  const media = mediaOf(project, found.clip);
  if (!media) {
    await bobine.ui.notify("info", say("notReady"));
    return null;
  }
  if (media.info.kind !== "video") {
    await bobine.ui.notify("info", say("notVideo"));
    return null;
  }
  return { project, sequence, found, media };
}

/** Codes d'erreur connus (Bobine ou programme natif) → message clair ; les autres sont nommés. */
const KNOWN: Record<string, TextKey> = {
  not_ready: "notReady",
  no_gpu: "noGpu",
  timeout: "tooSlow",
  busy: "busy",
  decode_failed: "decodeFailed",
};

async function explain(error: unknown): Promise<void> {
  const code = error instanceof BobineError ? error.code : null;
  const known = code ? KNOWN[code] : undefined;
  if (known) return bobine.ui.notify("info", say(known));
  await bobine.ui.notify("error", say("failed", { code: code ?? String(error).slice(0, 80) }));
  throw error;
}

/** Un rendu à la fois ; les panneaux le montrent (carte « En cours ») et peuvent l'annuler. */
let busy = false;
let current: PanelJob | null = null;
let signal = { cancelled: false };
let lastPost = 0;

/** Envoie l'état aux panneaux ; la progression au plus 4 fois par seconde. */
function publish(force = true): void {
  const now = Date.now();
  if (!force && now - lastPost < 250) return;
  lastPost = now;
  void bobine.post({ type: "status", busy, job: current }).catch(() => undefined);
}

function setBusy(value: boolean, job: PanelJob | null = null): void {
  busy = value;
  current = job;
  publish();
}

bobine.on("progress", ({ value }) => {
  if (!current || value === null) return;
  current = { ...current, progress: value };
  publish(false);
});

/** Bouton Annuler d'un panneau : le rendu s'arrête à l'étape suivante, le calcul natif tout de suite. */
async function cancel(): Promise<void> {
  if (!busy || !current || signal.cancelled) return;
  signal.cancelled = true;
  current = { ...current, cancelling: true };
  publish();
  await bobine.native.stop().catch(() => undefined);
}

/** Annonce la durée estimée, rend, puis propose le remplacement (et la vitesse). */
async function run(
  job: Job,
  found: Found,
  name: string,
  title: string,
  announce: (time: string, gpu: string, frames: number) => string,
  extra: (clip: string) => PluginCommand[],
  label: string,
  done: string,
): Promise<void> {
  if (busy) return bobine.ui.notify("info", say("busy"));
  signal = { cancelled: false };
  setBusy(true, {
    clip: found.clip.id,
    name: name || found.clip.id,
    label: title,
    progress: null,
    started: Date.now(),
    cancelling: false,
  });
  try {
    const measure = await estimate(host, job);
    const sources = frameCount(job.range, fpsOf(job.fps) || 30);
    const frames = frameCount(job.range, fpsOf(job.rate) || 30);
    let text = announce(
      formatDuration(sources * secondsPerSource(job, measure.seconds_per_frame)),
      measure.adapter,
      frames,
    );
    if (measure.software) text += say("software");
    await bobine.ui.notify("info", text);
    if (current) current = { ...current, progress: 0, started: Date.now() };
    const { media, first } = await render(host, job, measure.seconds_per_frame, signal);
    // Le rendu est fini : un autre peut partir pendant que la proposition attend une réponse.
    setBusy(false);
    const commands = [
      replaceClipMedia(found.clip.id, media, newSourceIn(found.clip, first)),
      ...extra(found.clip.id),
    ];
    const { applied } = await bobine.project.propose({ label, commands });
    await bobine.ui.notify(applied ? "ok" : "info", applied ? done : say("cancelled"));
  } catch (error) {
    // Annulé : le programme natif a été arrêté (`stopped`) ou le rendu s'est arrêté à une étape.
    if (signal.cancelled) await bobine.ui.notify("info", say("renderCancelled"));
    else await explain(error);
  } finally {
    if (busy) setBusy(false);
  }
}

/** Lance le rendu `options` sur le clip de `t`, ou dit pourquoi ce n'est pas possible. */
async function start(t: Target, options: Options): Promise<void> {
  const { found, media } = t;
  const facts = factsOf(found.clip, media.info, t.sequence);
  const fps = formatNumber(facts.fps);
  const reason = blocker(options, facts);
  if (reason) return bobine.ui.notify("info", say(reason, { ...facts.frame, fps }));
  const result = outcome(options, facts);
  const frame = facts.frame;
  const fps2 = formatNumber(fpsOf(result.rate));
  const size = `${result.out.width} × ${result.out.height}`;
  const upscale = result.kind === "upscale";
  const slow = options.action === "slowmo";
  const suffix = upscale
    ? `${result.out.width}x${result.out.height}`
    : slow
      ? `x${formatNumber(options.factor)}`
      : `${Math.round(fpsOf(result.rate))}fps`;
  const job: Job = {
    kind: result.kind,
    media: media.id,
    range: sourceRange(found.clip, media.info),
    fps: media.info.fps,
    rate: result.rate,
    frame,
    out: result.out,
    model: options.model,
    name: renderName(media.path, suffix),
    dedupe: options.dedupe,
    cuts: options.cuts,
  };
  const model = options.model === "anime" ? "animation" : "photo";
  const speed = found.clip.props.speed?.rate ?? 1;
  return run(
    job,
    found,
    fileName(media.path),
    upscale
      ? say("runUpscale", { size })
      : slow
        ? say("runSlowmo", { factor: formatNumber(options.factor) })
        : say("runFramerate", { fps: fps2 }),
    (time, gpu, frames) =>
      upscale
        ? say("upscaleStart", {
            from: `${frame.width} × ${frame.height}`,
            to: size,
            model,
            frames,
            time,
            gpu,
          })
        : say("slowmoStart", { fps, fps2, frames, time, gpu }),
    // Ralenti : même matière lue `factor` fois moins vite, le clip s'allonge (ripple : les clips
    // suivants se décalent au lieu de couper le ralenti).
    (clip) =>
      slow
        ? [
            {
              type: "SetClipSpeed",
              clips: [clip],
              rate: speed / options.factor,
              ripple: true,
            } as PluginCommand,
          ]
        : [],
    upscale ? say("upscaleLabel") : slow ? say("slowmoLabel") : say("framerateLabel"),
    upscale
      ? say("upscaled", { size })
      : slow
        ? say("slowed", { factor: formatNumber(options.factor) })
        : say("framerated", { fps: fps2 }),
  );
}

/** Réglages d'une commande : ceux de « Mes extensions » (taille de la séquence si plus grande). */
function commandOptions(s: Settings, action: Options["action"], t: Target): Options {
  const facts = factsOf(t.found.clip, t.media.info, t.sequence);
  return {
    action,
    fps: s.targetFps,
    factor: s.factor,
    width: sequenceWidth(facts) ?? s.width,
    model: s.model,
    dedupe: true,
    cuts: true,
  };
}

for (const [command, action] of [
  ["upscale", "upscale"],
  ["slowmo", "slowmo"],
  ["framerate", "framerate"],
] as const) {
  bobine.command(command, async () => {
    const t = await target();
    if (!t) return;
    await start(t, commandOptions(await settings(), action, t));
  });
}

bobine.on("message", (data) => {
  const message = asPanelMessage(data);
  if (message?.type === "hello") publish();
  if (message?.type === "cancel") void cancel();
  if (message?.type !== "run") return;
  void (async () => {
    const t = await target(message.clip);
    if (!t) return;
    await start(t, message.options);
  })().catch((error: unknown) => console.error(error));
});
