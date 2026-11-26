// Logique de Scaline : tourne dans l'iframe isolée du plugin. Bobine décode les images
// (`media.frames`), le programme natif calcule sur le GPU (`native.call`), Bobine encode et
// importe le résultat (`media.encode.*`), puis le plugin propose de remplacer le média du clip
// (`ReplaceClipMedia`) ; l'utilisateur applique ou annule, l'original reste dans le projet.
import {
  activeSequence,
  BobineError,
  bobine,
  flicks,
  type PluginCommand,
  type Project,
} from "@bobine/plugin-sdk";
import { formatDuration, formatNumber, say, type TextKey } from "./i18n";
import {
  chosenVideo,
  type Found,
  fpsOf,
  frameCount,
  frameSize,
  type Media,
  mediaOf,
  newSourceIn,
  renderName,
  sourceRange,
  upscaleSize,
} from "./model";
import { bobineHost, replaceClipMedia } from "./native";
import { estimate, type Job, outputFps, render } from "./render";

type Settings = { model: "photo" | "anime"; width: number; keepSpeed: boolean };

const host = bobineHost(bobine);

async function settings(): Promise<Settings> {
  const s = await bobine.settings.get();
  return {
    model: s.model === "animation" ? "anime" : "photo",
    width: typeof s.width === "number" ? s.width : 3840,
    keepSpeed: s["keep-speed"] === true,
  };
}

/** Projet, séquence active, clip vidéo choisi et son média ; ou `null` (l'utilisateur est prévenu). */
async function target(): Promise<{
  project: Project;
  sequence: NonNullable<ReturnType<typeof activeSequence>>;
  found: Found;
  media: Media;
} | null> {
  const [project, selection] = await Promise.all([bobine.project.get(), bobine.selection.get()]);
  const sequence = activeSequence(project);
  if (!project || !sequence) return null;
  const found = chosenVideo(sequence, selection.clips, flicks(selection.playhead));
  if (!found) {
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

/** Annonce la durée estimée, rend, puis propose le remplacement (et la vitesse). */
async function run(
  job: Job,
  found: Found,
  announce: (time: string, gpu: string, frames: number) => string,
  extra: (clip: string) => PluginCommand[],
  label: string,
  done: string,
): Promise<void> {
  try {
    const measure = await estimate(host, job);
    const frames = frameCount(job.range, fpsOf(job.fps) || 30);
    let text = announce(
      formatDuration(frames * measure.seconds_per_frame),
      measure.adapter,
      frames,
    );
    if (measure.software) text += say("software");
    await bobine.ui.notify("info", text);
    const { media, first } = await render(host, job, measure.seconds_per_frame);
    const commands = [
      replaceClipMedia(found.clip.id, media, newSourceIn(found.clip, first)),
      ...extra(found.clip.id),
    ];
    const { applied } = await bobine.project.propose({ label, commands });
    await bobine.ui.notify(applied ? "ok" : "info", applied ? done : say("cancelled"));
  } catch (error) {
    await explain(error);
  }
}

bobine.command("upscale", async () => {
  const t = await target();
  if (!t) return;
  const { sequence, found, media } = t;
  const options = await settings();
  const frame = frameSize(media.info);
  const out = upscaleSize(frame, sequence, options.width);
  if (!out) return bobine.ui.notify("info", say("alreadyBig", frame));
  const size = `${out.width} × ${out.height}`;
  const job: Job = {
    kind: "upscale",
    media: media.id,
    range: sourceRange(found.clip, media.info),
    fps: media.info.fps,
    frame,
    out,
    model: options.model,
    name: renderName(media.path, `${out.width}x${out.height}`),
  };
  await run(
    job,
    found,
    (time, gpu, frames) =>
      say("upscaleStart", {
        from: `${frame.width} × ${frame.height}`,
        to: size,
        model: options.model === "anime" ? "animation" : "photo",
        frames,
        time,
        gpu,
      }),
    () => [],
    say("upscaleLabel"),
    say("upscaled", { size }),
  );
});

bobine.command("slowmo", async () => {
  const t = await target();
  if (!t) return;
  const { found, media } = t;
  const options = await settings();
  const frame = frameSize(media.info);
  const job: Job = {
    kind: "interpolate",
    media: media.id,
    range: sourceRange(found.clip, media.info),
    fps: media.info.fps,
    frame,
    out: frame,
    model: options.model,
    name: renderName(media.path, "x2"),
  };
  const fps2 = fpsOf(outputFps(job));
  if (fps2 > 240) return bobine.ui.notify("info", say("tooFast"));
  const fps = formatNumber(fpsOf(job.fps));
  const rate = found.clip.props.speed?.rate ?? 1;
  await run(
    job,
    found,
    (time, gpu, frames) => say("slowmoStart", { fps, fps2: formatNumber(fps2), frames, time, gpu }),
    // Ralenti : même matière lue deux fois moins vite, le clip s'allonge (ripple : les clips
    // suivants se décalent au lieu de couper le ralenti).
    (clip) =>
      options.keepSpeed
        ? []
        : [{ type: "SetClipSpeed", clips: [clip], rate: rate / 2, ripple: true } as PluginCommand],
    options.keepSpeed ? say("cadenceLabel") : say("slowmoLabel"),
    options.keepSpeed ? say("doubled", { fps: formatNumber(fps2) }) : say("slowed"),
  );
});
