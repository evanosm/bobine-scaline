// Méthodes du protocole 1 pas encore typées par le SDK 0.1 (Bobine 0.4.0) : `media.frames` avec
// `every_frame`, `native.call`, `media.encode.*`, et la commande `ReplaceClipMedia`.
import type { Bobine, Method, PluginCommand, Rational } from "@bobine/plugin-sdk";

export type FramesRequest = {
  media: string;
  /** Secondes, temps du média : `[from, to)`. */
  from: number;
  to: number;
  /** Chaque image décodée du média (la cadence `fps` est ignorée). */
  every_frame: true;
  /** Boîte dans laquelle l'image est réduite, proportions gardées (1920 au plus, jamais agrandie). */
  width: number;
  height: number;
  format: "rgba";
};

/** Chemins relatifs au dossier de travail du programme natif. */
export type Frames = { raw: string; header: string; count: number; width: number; height: number };

/** Réponse de `upscale` et `interpolate` (programme natif). */
export type Rendered = {
  /** Images RGBA bout à bout, à passer à `media.encode.append`. */
  raw: string;
  count: number;
  width: number;
  height: number;
  /** Instants (secondes, temps du média) de la première et de la dernière image source rendues. */
  first_time: number | null;
  last_time: number | null;
};

export type Estimate = { seconds_per_frame: number; adapter: string; software: boolean };

/** `ReplaceClipMedia` (Bobine 0.4.0) : le clip lit un autre média ; `source_in` en flicks. */
export type ReplaceClipMedia = {
  type: "ReplaceClipMedia";
  clip: string;
  media: string;
  source_in: number;
};

// `Method` et `PluginCommand` du SDK 0.1 ne connaissent pas encore ces noms ; Bobine les sert.
const FRAMES = "media.frames" as string as Method;
const NATIVE = "native.call" as string as Method;
const ENCODE_BEGIN = "media.encode.begin" as string as Method;
const ENCODE_APPEND = "media.encode.append" as string as Method;
const ENCODE_FINISH = "media.encode.finish" as string as Method;
const ENCODE_CANCEL = "media.encode.cancel" as string as Method;

/** Ce dont le rendu a besoin de Bobine (remplacé par un faux dans les tests). */
export type Host = {
  frames(request: FramesRequest): Promise<Frames>;
  call<T>(method: string, params: unknown, timeout: number): Promise<T>;
  encodeBegin(request: { width: number; height: number; fps: Rational; name: string }): Promise<{
    session: string;
  }>;
  encodeAppend(request: { session: string; raw: string }): Promise<{ frames: number }>;
  encodeFinish(request: { session: string }): Promise<{ media: string }>;
  encodeCancel(request: { session: string }): Promise<null>;
};

export function bobineHost(bobine: Bobine): Host {
  return {
    frames: (request) => bobine.call(FRAMES, request) as Promise<Frames>,
    call: <T>(method: string, params: unknown, timeout: number) =>
      bobine.call(NATIVE, { method, params, timeout }) as Promise<T>,
    encodeBegin: (request) => bobine.call(ENCODE_BEGIN, request) as Promise<{ session: string }>,
    encodeAppend: (request) => bobine.call(ENCODE_APPEND, request) as Promise<{ frames: number }>,
    encodeFinish: (request) => bobine.call(ENCODE_FINISH, request) as Promise<{ media: string }>,
    encodeCancel: (request) => bobine.call(ENCODE_CANCEL, request) as Promise<null>,
  };
}

export function replaceClipMedia(clip: string, media: string, sourceIn: number): PluginCommand {
  const command: ReplaceClipMedia = { type: "ReplaceClipMedia", clip, media, source_in: sourceIn };
  return command as unknown as PluginCommand;
}
