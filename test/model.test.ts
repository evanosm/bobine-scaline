import { describe, expect, it } from "vitest";
import {
  chosenVideo,
  frameSize,
  lots,
  mediaOf,
  newSourceIn,
  renderName,
  sourceRange,
  upscaleSize,
} from "../src/model";
import { info, project, SECOND, sequence, videoClip } from "./fixtures";

describe("clip choisi", () => {
  it("prend le clip vidéo sélectionné, sinon celui sous la tête de lecture", () => {
    const a = videoClip("a", 0, 2 * SECOND);
    const b = videoClip("b", 2 * SECOND);
    const seq = sequence([
      { kind: "video", clips: [a, b] },
      { kind: "audio", clips: [videoClip("son")] },
    ]);
    expect(chosenVideo(seq, ["b"], 0)?.clip.id).toBe("b");
    expect(chosenVideo(seq, ["son"], SECOND)?.clip.id).toBe("a");
    expect(chosenVideo(seq, [], 3 * SECOND)?.clip.id).toBe("b");
    expect(chosenVideo(seq, [], 9 * SECOND)).toBeNull();
  });

  it("attend l'analyse du média", () => {
    const clip = videoClip("a");
    const seq = sequence([{ kind: "video", clips: [clip] }]);
    expect(mediaOf(project(seq, [{ id: "m-a", info: null }]), clip)).toBeNull();
    expect(mediaOf(project(seq, [{ id: "m-a", info: info() }]), clip)?.path).toBe("/rushs/m-a.mov");
  });
});

describe("tailles", () => {
  it("réduit les images reçues à 1920 de côté, comme Bobine", () => {
    expect(frameSize({ width: 3840, height: 2160 })).toEqual({ width: 1920, height: 1080 });
    expect(frameSize({ width: 1080, height: 1920 })).toEqual({ width: 1080, height: 1920 });
    expect(frameSize({ width: 1280, height: 720 })).toEqual({ width: 1280, height: 720 });
  });

  it("agrandit jusqu'à la séquence quand elle est plus grande, sinon à la largeur visée", () => {
    const hd = { width: 1920, height: 1080 };
    expect(upscaleSize(hd, { width: 3840, height: 2160 }, 3840)).toMatchObject({
      width: 3840,
      height: 2160,
      factor: 2,
    });
    expect(upscaleSize(hd, hd, 3840)).toMatchObject({ width: 3840, height: 2160 });
    // 720p dans une séquence 1080p : la séquence suffit.
    expect(upscaleSize({ width: 1280, height: 720 }, hd, 3840)).toMatchObject({
      width: 1920,
      height: 1080,
    });
    // Vertical : l'image tient dans la séquence 2160 × 3840.
    expect(
      upscaleSize({ width: 1080, height: 1920 }, { width: 2160, height: 3840 }, 3840),
    ).toMatchObject({ width: 2160, height: 3840 });
  });

  it("garde des côtés pairs et les plafonds de l'encodeur", () => {
    const out = upscaleSize({ width: 853, height: 480 }, { width: 853, height: 480 }, 3840);
    expect(out?.width).toBe(3840);
    expect((out?.height ?? 1) % 2).toBe(0);
    const big = upscaleSize({ width: 1920, height: 1080 }, { width: 1920, height: 1080 }, 7680);
    expect(big).toMatchObject({ width: 7680, height: 4320 });
    const tall = upscaleSize({ width: 1000, height: 1920 }, { width: 100, height: 100 }, 7680);
    expect(tall && tall.height <= 8192 && tall.width * tall.height <= 8192 * 4320).toBe(true);
  });

  it("ne propose rien quand l'image est déjà à la taille visée", () => {
    expect(upscaleSize({ width: 3840, height: 2160 }, { width: 1920, height: 1080 }, 3840)).toBe(
      null,
    );
  });
});

describe("plage et lots", () => {
  it("lit la plage du clip, vitesse comprise, avec une image de marge de chaque côté", () => {
    const clip = videoClip("a", 0, 2 * SECOND, 3 * SECOND);
    clip.props.speed = { type: "constant", rate: 1.5 };
    const r = sourceRange(clip, info());
    expect(r.from).toBeCloseTo(3 - 1 / 25);
    expect(r.to).toBeCloseTo(6 + 1 / 25);
    // Bornée au média.
    const late = videoClip("b", 0, 4 * SECOND, 8 * SECOND);
    expect(sourceRange(late, info()).to).toBeCloseTo(10);
  });

  it("découpe en lots sous le plafond de Bobine, chacun reprenant une image du précédent", () => {
    const frame = { width: 1920, height: 1080 };
    const plan = lots({ from: 0, to: 10 }, 25, frame);
    const perLot = Math.floor((384 * 1024 * 1024) / (1920 * 1080 * 4));
    for (const lot of plan) {
      expect((lot.to - lot.from) * 25).toBeLessThanOrEqual(perLot + 1e-6);
    }
    expect(plan[0]?.from).toBe(0);
    expect(plan.at(-1)?.to).toBe(10);
    for (let i = 1; i < plan.length; i++) {
      const prev = plan[i - 1];
      const lot = plan[i];
      if (!prev || !lot) throw new Error("lot");
      expect(prev.to - lot.from).toBeCloseTo(1 / 25);
    }
  });

  it("recale le point d'entrée sur la première image rendue", () => {
    const clip = videoClip("a", 0, SECOND, 3 * SECOND);
    expect(newSourceIn(clip, 2.96)).toBe(Math.round(0.04 * SECOND));
    expect(newSourceIn(clip, 3.5)).toBe(0);
  });

  it("nomme le rendu d'après le fichier d'origine", () => {
    expect(renderName("/rushs/plage.mov", "3840x2160")).toBe("plage 3840x2160");
    expect(renderName("C:\\vidéos\\saut.final.mp4", "x2")).toBe("saut.final x2");
    expect(renderName("", "x2")).toBe("clip x2");
  });
});
