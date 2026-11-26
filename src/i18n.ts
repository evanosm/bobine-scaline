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
    fr: "Doubler la cadence de ce clip dépasserait 240 images par seconde.",
    en: "Doubling this clip's frame rate would go past 240 frames per second.",
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
  cadenceLabel: { fr: "Doubler la cadence", en: "Double the frame rate" },
  upscaled: {
    fr: "Clip agrandi en {size}. Le média d'origine reste dans le projet.",
    en: "Clip upscaled to {size}. The original media stays in the project.",
  },
  slowed: {
    fr: "Ralenti fluide appliqué (vitesse ÷ 2, son lié compris). Le média d'origine reste dans le projet.",
    en: "Smooth slow motion applied (speed ÷ 2, linked sound included). The original media stays in the project.",
  },
  doubled: {
    fr: "Cadence doublée ({fps} im./s), même vitesse. Le média d'origine reste dans le projet.",
    en: "Frame rate doubled ({fps} fps), same speed. The original media stays in the project.",
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
