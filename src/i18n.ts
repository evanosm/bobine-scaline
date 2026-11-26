// Textes du plugin en français et en anglais, choisis d'après la langue du système (la logique
// tourne dans une iframe isolée : `navigator.language` est tout ce qu'elle voit).
const french =
  typeof navigator !== "undefined" && navigator.language.toLowerCase().startsWith("fr");

const TEXTS = {
  noVideo: {
    fr: "Choisis d'abord un clip vidéo, ou place la tête de lecture sur un clip vidéo.",
    en: "Select a video clip first, or put the playhead over a video clip.",
  },
  notReady: {
    fr: "Ce média est encore en cours d'analyse : réessaie dans un instant.",
    en: "This media is still being analysed: try again in a moment.",
  },
  notVideo: {
    fr: "Ce clip montre une image fixe : Scaline traite les vidéos.",
    en: "This clip shows a still image: Scaline works on videos.",
  },
  alreadyBig: {
    fr: "Ce clip fait déjà {width} × {height} : rien à agrandir. Change la largeur visée dans les réglages de Scaline.",
    en: "This clip is already {width} × {height}: nothing to upscale. Change the target width in Scaline's settings.",
  },
  tooFast: {
    fr: "Ce ralenti dépasserait 240 images par seconde : choisis un facteur plus petit dans les réglages de Scaline.",
    en: "This slow motion would go past 240 frames per second: pick a smaller factor in Scaline's settings.",
  },
  alreadyFps: {
    fr: "Ce clip tourne déjà à {fps} im./s : choisis une cadence plus haute dans les réglages de Scaline.",
    en: "This clip already runs at {fps} fps: pick a higher frame rate in Scaline's settings.",
  },
  upscaleStart: {
    fr: "Agrandissement {from} → {to} ({model}) : {frames} images, environ {time} sur {gpu}.",
    en: "Upscaling {from} → {to} ({model}): {frames} frames, about {time} on {gpu}.",
  },
  slowmoStart: {
    fr: "Images intermédiaires ({fps} → {fps2} im./s) : {frames} images, environ {time} sur {gpu}.",
    en: "In-between frames ({fps} → {fps2} fps): {frames} frames, about {time} on {gpu}.",
  },
  software: {
    fr: " Aucun GPU utilisable : calcul logiciel, très lent.",
    en: " No usable GPU: software rendering, very slow.",
  },
  upscaleLabel: { fr: "Agrandir en 4K", en: "Upscale to 4K" },
  slowmoLabel: { fr: "Ralenti fluide", en: "Smooth slow motion" },
  framerateLabel: { fr: "Augmenter la cadence", en: "Increase the frame rate" },
  upscaled: {
    fr: "Clip agrandi en {size}. Le média d'origine reste dans le projet.",
    en: "Clip upscaled to {size}. The original media stays in the project.",
  },
  slowed: {
    fr: "Ralenti fluide appliqué (vitesse ÷ {factor}, son lié compris). Le média d'origine reste dans le projet.",
    en: "Smooth slow motion applied (speed ÷ {factor}, linked sound included). The original media stays in the project.",
  },
  framerated: {
    fr: "Clip passé à {fps} im./s, même vitesse. Le média d'origine reste dans le projet.",
    en: "Clip now runs at {fps} fps, same speed. The original media stays in the project.",
  },
  cancelled: {
    fr: "Rien n'a changé dans le montage ; le média rendu reste dans la liste des médias.",
    en: "Nothing changed in the edit; the rendered media stays in the media list.",
  },
  noGpu: {
    fr: "Aucun GPU utilisable (Metal, DirectX 12 ou Vulkan) sur cette machine : Scaline ne peut pas calculer.",
    en: "No usable GPU (Metal, DirectX 12 or Vulkan) on this computer: Scaline cannot compute.",
  },
  tooSlow: {
    fr: "Le calcul a dépassé le délai de Bobine (GPU trop lent ou occupé). Essaie sur un clip plus court.",
    en: "The computation went past Bobine's time limit (GPU too slow or busy). Try a shorter clip.",
  },
  busy: {
    fr: "Scaline rend déjà un clip : attends la fin de ce rendu.",
    en: "Scaline is already rendering a clip: wait for it to finish.",
  },
  decodeFailed: {
    fr: "Ce média n'a pas pu être lu.",
    en: "This media could not be read.",
  },
  failed: {
    fr: "Le rendu a échoué ({code}). Le montage n'a pas changé.",
    en: "Rendering failed ({code}). The edit did not change.",
  },
  badFactor: {
    fr: "Le ralenti va de ×1,5 à ×8.",
    en: "Slow motion goes from ×1.5 to ×8.",
  },
  tooBig: {
    fr: "Au-delà de 7680 px de large, l'encodeur refuse l'image.",
    en: "Beyond 7680 px wide, the encoder refuses the frame.",
  },
  renderCancelled: {
    fr: "Rendu annulé. Le montage n'a pas changé.",
    en: "Render cancelled. The edit did not change.",
  },
  // Panneau : clip.
  panelEmpty: { fr: "Sélectionner un clip", en: "Select a clip" },
  panelNotReady: { fr: "Analyse du média en cours…", en: "Analysing the media…" },
  clipMeta: { fr: "{size} · {fps} im./s · {duration} s", en: "{size} · {fps} fps · {duration} s" },
  // Panneau : réglages.
  fieldPreset: { fr: "Préréglage", en: "Preset" },
  presetCustom: { fr: "Personnalisé", en: "Custom" },
  presetDouble: { fr: "Doubler la cadence", en: "Double the frame rate" },
  presetFps120: { fr: "Passer à 120 im./s", en: "Go to 120 fps" },
  presetUpscale4k: { fr: "Agrandir en 4K", en: "Upscale to 4K" },
  presetSlow4: { fr: "Ralenti ×4", en: "Slow motion ×4" },
  fieldAction: { fr: "Action", en: "Action" },
  actionFramerate: { fr: "Cadence", en: "Frame rate" },
  actionSlowmo: { fr: "Ralenti", en: "Slow motion" },
  actionUpscale: { fr: "Agrandissement", en: "Upscale" },
  fieldFps: { fr: "Cadence visée", en: "Target frame rate" },
  unitFps: { fr: "im./s", en: "fps" },
  noteFps: { fr: "×{ratio} par rapport au clip", en: "×{ratio} the clip's rate" },
  noteNtsc: { fr: " · {fps} dans cette séquence NTSC", en: " · {fps} in this NTSC sequence" },
  fieldFactor: { fr: "Ralenti", en: "Slow down" },
  noteFactor: {
    fr: "{fps} → {fps2} im./s · durée {from} s → {to} s",
    en: "{fps} → {fps2} fps · length {from} s → {to} s",
  },
  fieldWidth: { fr: "Largeur", en: "Width" },
  unitPx: { fr: "px", en: "px" },
  noteWidth: { fr: "{size} · ×{ratio}", en: "{size} · ×{ratio}" },
  noteStretch: {
    fr: " · au-delà de ×4, l'image est étirée sans détail en plus",
    en: " · beyond ×4 the frame is stretched with no extra detail",
  },
  quickSequence: { fr: "Séquence", en: "Sequence" },
  fieldModel: { fr: "Image", en: "Footage" },
  modelPhoto: { fr: "Prise de vue réelle", en: "Live action" },
  modelAnime: { fr: "Animation", en: "Animation" },
  advanced: { fr: "Avancé", en: "Advanced" },
  dedupeLabel: { fr: "Recalculer les images répétées", en: "Rebuild repeated frames" },
  dedupeHint: {
    fr: "Un jeu à 30 ou 45 im./s enregistré à 60 répète des images : elles sont recalculées et le mouvement devient régulier.",
    en: "A game at 30 or 45 fps recorded at 60 repeats frames: they are rebuilt and the motion becomes even.",
  },
  cutsLabel: { fr: "Respecter les changements de plan", en: "Respect shot changes" },
  cutsHint: {
    fr: "Entre deux plans, l'image la plus proche est reprise au lieu d'un fondu inventé.",
    en: "Between two shots, the nearest frame is reused instead of an invented blend.",
  },
  engine: { fr: "Moteur", en: "Engine" },
  engineRife: {
    fr: "RIFE 4.25, chaque image à son instant exact",
    en: "RIFE 4.25, each frame at its exact instant",
  },
  engineEsrgan: {
    fr: "Real-ESRGAN ×4 ({model}), puis mise à la taille",
    en: "Real-ESRGAN ×4 ({model}), then resized",
  },
  // Panneau : résultat.
  result: { fr: "Résultat", en: "Result" },
  resIn: { fr: "Entrée", en: "Input" },
  resOut: { fr: "Sortie", en: "Output" },
  resFrames: {
    fr: "{size} · {fps} im./s · {frames} images",
    en: "{size} · {fps} fps · {frames} frames",
  },
  resLength: { fr: "Durée", en: "Length" },
  resWork: { fr: "Calcul", en: "Work" },
  workInterp: { fr: "{n} images intermédiaires", en: "{n} in-between frames" },
  workUpscale: { fr: "{n} images agrandies", en: "{n} upscaled frames" },
  resTime: { fr: "Temps estimé", en: "Estimated time" },
  timeMeasuring: { fr: "Mesure sur le GPU…", en: "Measuring on the GPU…" },
  timeValue: { fr: "≈ {time} sur {gpu}", en: "≈ {time} on {gpu}" },
  timeSoftware: { fr: " (logiciel, très lent)", en: " (software, very slow)" },
  resFile: { fr: "Fichier", en: "File" },
  fileValue: { fr: "ProRes 422 · ≈ {size} · sans son", en: "ProRes 422 · ≈ {size} · no sound" },
  runFramerate: { fr: "Rendre à {fps} im./s", en: "Render at {fps} fps" },
  runSlowmo: { fr: "Ralentir ×{factor}", en: "Slow down ×{factor}" },
  runUpscale: { fr: "Agrandir en {size}", en: "Upscale to {size}" },
  running: { fr: "Un rendu est en cours.", en: "A render is running." },
  hintAlreadyFps: {
    fr: "Le clip tourne déjà à {fps} im./s : vise une cadence plus haute.",
    en: "The clip already runs at {fps} fps: aim for a higher frame rate.",
  },
  hintTooFast: {
    fr: "Au-delà de 240 im./s, l'encodeur refuse.",
    en: "Beyond 240 fps, the encoder refuses.",
  },
  hintAlreadyBig: {
    fr: "Le clip fait déjà cette largeur : vise plus large.",
    en: "The clip is already this wide: aim wider.",
  },
  hintOldBobine: {
    fr: "Mets Bobine à jour pour lancer un rendu depuis ce panneau ; les commandes du clip marchent.",
    en: "Update Bobine to render from this panel; the clip commands work.",
  },
  // Panneau : rendu en cours.
  jobTitle: { fr: "En cours", en: "In progress" },
  jobCancel: { fr: "Annuler", en: "Cancel" },
  jobCancelling: { fr: "Annulation…", en: "Cancelling…" },
  jobPreparing: { fr: "Préparation du GPU…", en: "Preparing the GPU…" },
  jobPercent: { fr: "{percent} %", en: "{percent}%" },
  jobLeft: { fr: "{percent} % · encore {time} environ", en: "{percent}% · about {time} left" },
} as const;

export type TextKey = keyof typeof TEXTS;

/** Texte dans la langue du système, `{nom}` remplacés par `values`. */
export function say(key: TextKey, values: Record<string, string | number> = {}): string {
  const text: string = TEXTS[key][french ? "fr" : "en"];
  return text.replace(/\{(\w+)\}/g, (_, name: string) => String(values[name] ?? ""));
}

/** Nombre lisible (virgule décimale en français, décimales inutiles retirées). */
export function formatNumber(value: number, digits = 2): string {
  const text = String(Number(value.toFixed(digits)));
  return french ? text.replace(".", ",") : text;
}

/** Durée lisible : « 40 s », « 12 min », « 1 h 20 min ». */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m === 0 ? `${h} h` : `${h} h ${m} min`;
}

/** Nombre entier lisible, séparateur de milliers de la langue (« 3 600 », « 3,600 »). */
export function formatCount(value: number): string {
  return new Intl.NumberFormat(french ? "fr-FR" : "en-US").format(Math.round(value));
}

/** Taille de fichier lisible : « 820 Mo », « 9,8 Go ». */
export function formatBytes(bytes: number): string {
  const giga = bytes / 1e9;
  if (giga >= 1) return `${formatNumber(giga, 1)} ${french ? "Go" : "GB"}`;
  return `${Math.max(1, Math.round(bytes / 1e6))} ${french ? "Mo" : "MB"}`;
}

/** Langue des textes (attribut `lang` du panneau). */
export const LANG = french ? "fr" : "en";
