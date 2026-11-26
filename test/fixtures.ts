// Clips, séquences et projets minimaux pour les tests (seuls les champs lus par le plugin comptent).
import type { Clip, MediaInfo, Project, Sequence, Track } from "@bobine/plugin-sdk";

export const SECOND = 705_600_000;

export function videoClip(id: string, start = 0, duration = 4 * SECOND, sourceIn = 0): Clip {
  return {
    id,
    duration,
    enabled: true,
    timeline_start: start,
    source_in: sourceIn,
    source: { type: "media", id: `m-${id}` },
    props: {
      volume_db: { type: "static", value: 0 },
      opacity: { type: "static", value: 1 },
      transform: {
        position: { type: "static", value: { x: 0, y: 0 } },
        scale: { type: "static", value: 1 },
        rotation: { type: "static", value: 0 },
      },
    },
  } as Clip;
}

export function sequence(
  tracks: { kind: "video" | "audio"; clips: Clip[] }[],
  size = { width: 1920, height: 1080 },
): Sequence {
  return {
    id: "s1",
    name: "Séquence",
    ...size,
    fps: { num: 30, den: 1 },
    sample_rate: 48_000,
    markers: [],
    tracks: tracks.map(
      (t, i): Track => ({
        id: `t${i}`,
        kind: t.kind,
        name: `Piste ${i}`,
        muted: false,
        solo: false,
        locked: false,
        clips: t.clips,
      }),
    ),
  };
}

export function info(over: Partial<MediaInfo> = {}): MediaInfo {
  return {
    audio_streams: 1,
    duration: 10 * SECOND,
    fps: { num: 25, den: 1 },
    width: 1920,
    height: 1080,
    kind: "video",
    ...over,
  };
}

export function project(seq: Sequence, media: { id: string; info: MediaInfo | null }[]): Project {
  return {
    id: "p1",
    format_version: 6,
    media: media.map((m) => ({ ...m, path: `/rushs/${m.id}.mov`, fingerprint: "x" })),
    sequences: [seq],
    active_sequence: seq.id,
    fonts: [],
  } as Project;
}
