import { describe, expect, it } from "vitest";
import {
  asPanelMessage,
  blocker,
  clampFactor,
  type Facts,
  fpsShortcuts,
  initialOptions,
  type Options,
  outcome,
  PRESETS,
  presetOf,
  remaining,
  sequenceWidth,
  widthShortcuts,
} from "../src/options";

/** Clip 1080p à 30 im./s de 10 s, dans une séquence 1080p à 30. */
const hd30: Facts = {
  rate: { num: 30, den: 1 },
  fps: 30,
  frame: { width: 1920, height: 1080 },
  sequence: { width: 1920, height: 1080, fps: { num: 30, den: 1 } },
  clipSeconds: 10,
  sourceSeconds: 10,
};
const base: Options = {
  action: "framerate",
  fps: 60,
  factor: 2,
  width: 3840,
  model: "photo",
  dedupe: true,
  cuts: true,
};

describe("réglages du panneau", () => {
  it("propose le double de la cadence du clip puis les cadences courantes au-dessus", () => {
    expect(fpsShortcuts(30)).toEqual([60, 120, 144, 240]);
    expect(fpsShortcuts(25)).toEqual([50, 60, 120, 144, 240]);
    expect(fpsShortcuts(60)).toEqual([120, 144, 240]);
    expect(fpsShortcuts(240)).toEqual([]);
  });

  it("propose la taille de la séquence seulement si elle est plus grande que le clip", () => {
    expect(sequenceWidth(hd30)).toBeNull();
    expect(widthShortcuts(hd30).map((s) => s.id)).toEqual(["1440p", "4k"]);
    const in4k = { ...hd30, sequence: { ...hd30.sequence, width: 3840, height: 2160 } };
    expect(widthShortcuts(in4k)).toEqual([
      { id: "sequence", width: 3840 },
      { id: "1440p", width: 2560 },
    ]);
  });

  it("accepte toute valeur utile et dit pourquoi une autre est refusée", () => {
    expect(blocker({ ...base, fps: 30 }, hd30)).toBe("alreadyFps");
    expect(blocker({ ...base, fps: 90 }, hd30)).toBeNull();
    expect(blocker({ ...base, fps: 300 }, hd30)).toBe("tooFast");
    expect(blocker({ ...base, action: "slowmo", factor: 2.5 }, hd30)).toBeNull();
    expect(blocker({ ...base, action: "slowmo", factor: 1 }, hd30)).toBe("badFactor");
    expect(blocker({ ...base, action: "slowmo", factor: 8 }, { ...hd30, fps: 60 })).toBe("tooFast");
    expect(blocker({ ...base, action: "upscale", width: 1920 }, hd30)).toBe("alreadyBig");
    expect(blocker({ ...base, action: "upscale", width: 3000 }, hd30)).toBeNull();
    expect(blocker({ ...base, action: "upscale", width: 9000 }, hd30)).toBe("tooBig");
  });

  it("range un facteur de ralenti au demi près entre 1,5 et 8", () => {
    expect(clampFactor(2.3)).toBe(2.5);
    expect(clampFactor(1)).toBe(1.5);
    expect(clampFactor(12)).toBe(8);
  });

  it("décrit le résultat : images, durée, calcul et taille du fichier", () => {
    const double = outcome(base, hd30);
    expect(double).toMatchObject({
      kind: "interpolate",
      rate: { num: 60, den: 1 },
      sources: 300,
      outputs: 600,
      computed: 300,
      clipSeconds: 10,
    });
    // ProRes 422 : ~2,4 bits par pixel et par image.
    expect(double.bytes).toBeCloseTo((1920 * 1080 * 600 * 2.4) / 8);
    // Ralenti ×2,5 d'un clip NTSC : fraction exacte, clip 2,5 fois plus long.
    const ntsc = { ...hd30, rate: { num: 30000, den: 1001 }, fps: 30000 / 1001 };
    const slow = outcome({ ...base, action: "slowmo", factor: 2.5 }, ntsc);
    expect(slow.rate).toEqual({ num: 150000, den: 2002 });
    expect(slow.clipSeconds).toBe(25);
    const up = outcome({ ...base, action: "upscale", width: 2560 }, hd30);
    expect(up).toMatchObject({
      kind: "upscale",
      out: { width: 2560, height: 1440 },
      computed: 300,
    });
  });

  it("suit la cadence NTSC de la séquence", () => {
    const seq = { ...hd30, sequence: { ...hd30.sequence, fps: { num: 30000, den: 1001 } } };
    expect(outcome(base, seq).rate).toEqual({ num: 60000, den: 1001 });
  });

  it("reconnaît le préréglage choisi, et « Personnalisé » dès qu'un réglage s'en écarte", () => {
    const double = PRESETS.find((p) => p.id === "double")?.apply(base, hd30);
    expect(double && presetOf(double, hd30)).toBe("double");
    expect(presetOf({ ...base, fps: 90 }, hd30)).toBe("custom");
    expect(presetOf({ ...base, action: "upscale", width: 3840 }, hd30)).toBe("upscale4k");
    expect(presetOf({ ...base, action: "slowmo", factor: 4 }, hd30)).toBe("slow4");
  });

  it("part des réglages de l'utilisateur, ou du double si sa cadence n'est pas plus haute", () => {
    const settings = { model: "anime" as const, factor: 4, targetFps: 60, width: 3840 };
    expect(initialOptions(settings, hd30)).toMatchObject({ fps: 60, factor: 4, model: "anime" });
    expect(initialOptions(settings, { ...hd30, fps: 60 }).fps).toBe(120);
    expect(initialOptions(settings, hd30)).toMatchObject({ width: 3840, dedupe: true, cuts: true });
  });

  it("n'accepte de la logique que des messages bien formés", () => {
    expect(asPanelMessage({ type: "run", clip: "c1", options: base })).toEqual({
      type: "run",
      clip: "c1",
      options: base,
    });
    expect(asPanelMessage({ type: "run", clip: "c1", options: { ...base, fps: 999 } })).toBeNull();
    expect(
      asPanelMessage({ type: "run", clip: "c1", options: { ...base, cuts: "oui" } }),
    ).toBeNull();
    expect(asPanelMessage({ type: "status", busy: true })).toEqual({
      type: "status",
      busy: true,
      job: null,
    });
    expect(asPanelMessage({ type: "cancel" })).toEqual({ type: "cancel" });
    expect(asPanelMessage("hello")).toBeNull();
  });

  it("transmet le rendu en cours et en tire le temps restant", () => {
    const job = {
      clip: "c1",
      name: "partie.mp4",
      label: "Rendre à 60 im./s",
      progress: 0.25,
      started: 0,
      cancelling: false,
    };
    expect(asPanelMessage({ type: "status", busy: true, job })).toEqual({
      type: "status",
      busy: true,
      job,
    });
    // Un quart en 60 s : encore 180 s.
    expect(remaining(job, 60_000)).toBeCloseTo(180);
    expect(remaining({ ...job, progress: null }, 60_000)).toBeNull();
  });
});
