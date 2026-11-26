// Le rendu de bout en bout contre un faux Bobine et un faux programme natif qui suivent le
// protocole : lots d'images d'un média à cadence fixe (`every_frame`), images déjà rendues
// sautées (`after`), images intermédiaires, encodage dans l'ordre.
import { describe, expect, it } from "vitest";
import type { Frames, FramesRequest, Host, Rendered } from "../src/native";
import { framesPerCall, type Job, render, secondsPerSource } from "../src/render";

/** Faux hôte : un média de `count` images à `fps`, qui note tout ce que le plugin demande. */
function fakeHost(fps: number, count: number, options: { failAt?: number } = {}) {
  const times = Array.from({ length: count }, (_, i) => i / fps);
  const sets = new Map<string, number[]>();
  const appended: { session: string; raw: string; count: number }[] = [];
  const outputs = new Map<string, number>();
  const calls: { method: string; params: Record<string, unknown>; timeout: number }[] = [];
  const log: string[] = [];
  let n = 0;
  const host: Host = {
    async frames(request: FramesRequest): Promise<Frames> {
      log.push(`frames ${request.from.toFixed(3)} ${request.to.toFixed(3)}`);
      expect(request.every_frame).toBe(true);
      const set = times.filter((t) => t >= request.from - 1e-6 && t < request.to - 1e-6);
      n += 1;
      sets.set(`frames-${n}.json`, set);
      return {
        header: `frames-${n}.json`,
        raw: `frames-${n}.raw`,
        count: set.length,
        width: request.width,
        height: request.height,
      };
    },
    async call<T>(method: string, params: unknown, timeout: number): Promise<T> {
      const p = params as Record<string, unknown>;
      calls.push({ method, params: p, timeout });
      if (options.failAt !== undefined && calls.length === options.failAt) {
        throw new Error("gpu_failed");
      }
      const set = sets.get(String(p.frames)) ?? [];
      const first = Number(p.first);
      const after = p.after as number | null;
      const slice = set.slice(first, first + Number(p.count));
      const emitted: number[] = [];
      if (method === "interpolate") {
        // Comme `retime` : sorties à `origin + k / fps` jusqu'à la dernière image de l'appel, ou
        // jusqu'à la fin de son intervalle au dernier appel.
        const rate = p.fps as { num: number; den: number };
        const origin = (p.origin as number | null) ?? (slice[0] as number);
        const lastTime = slice.at(-1) as number;
        const until = p.last === true ? lastTime + 1 / fps - 2e-4 : lastTime + 1e-4;
        for (let k = 0; ; k++) {
          const t = origin + (k * rate.den) / rate.num;
          if (t > until) break;
          if (after === null || t > after + 1e-4) emitted.push(t);
        }
      } else {
        for (const t of slice) if (after === null || t > after + 1e-4) emitted.push(t);
      }
      const written = emitted.length;
      const raw = `scaline-${calls.length}.raw`;
      outputs.set(raw, written);
      const result: Rendered = {
        raw,
        count: written,
        width: 8,
        height: 8,
        first_time: emitted[0] ?? null,
        last_time: emitted.at(-1) ?? null,
      };
      return result as T;
    },
    async encodeBegin(request) {
      log.push(`begin ${request.width}x${request.height} ${request.fps.num}/${request.fps.den}`);
      return { session: "s1" };
    },
    async encodeAppend(request) {
      appended.push({ ...request, count: outputs.get(request.raw) ?? -1 });
      return { frames: appended.reduce((a, b) => a + b.count, 0) };
    },
    async encodeFinish() {
      log.push("finish");
      return { media: "m-rendu" };
    },
    async encodeCancel() {
      log.push("cancel");
      return null;
    },
  };
  return { host, appended, calls, log };
}

function job(
  kind: Job["kind"],
  range: { from: number; to: number },
  fps = 25,
  rate = kind === "interpolate" ? 2 * fps : fps,
): Job {
  return {
    kind,
    media: "m1",
    range,
    fps: { num: fps, den: 1 },
    rate: { num: rate, den: 1 },
    frame: { width: 1920, height: 1080 },
    out: kind === "upscale" ? { width: 3840, height: 2160 } : { width: 1920, height: 1080 },
    model: "photo",
    name: "plage 3840x2160",
  };
}

describe("rendu par lots", () => {
  it("agrandit chaque image de la plage une fois, dans l'ordre, sur plusieurs lots", async () => {
    const { host, appended, calls, log } = fakeHost(25, 250);
    const result = await render(host, job("upscale", { from: 2, to: 8 }), 0.5);
    expect(result).toEqual({ media: "m-rendu", first: 2 });
    // 6 s à 25 im./s : 150 images, chacune rendue une fois malgré les recouvrements.
    expect(appended.reduce((a, b) => a + b.count, 0)).toBe(150);
    expect(log.filter((l) => l.startsWith("frames")).length).toBeGreaterThan(1);
    expect(log[0]).toBe("begin 3840x2160 25/1");
    expect(log.at(-1)).toBe("finish");
    // Réglages transmis au programme natif, un appel sous le délai de Bobine.
    const first = calls[0];
    expect(first?.method).toBe("upscale");
    expect(first?.params).toMatchObject({ model: "photo", width: 3840, height: 2160 });
    expect(first?.timeout).toBeLessThanOrEqual(600);
    // Progression : chaque appel donne sa part de la tâche entière.
    const spans = calls.map((c) => c.params.progress as [number, number]);
    expect(spans[0]?.[0]).toBe(0);
    expect(spans.every(([base, part]) => base >= 0 && base + part <= 1.01)).toBe(true);
  });

  it("double les images : 2 × N images à cadence double, la dernière répétée", async () => {
    const { host, appended, log } = fakeHost(30, 300);
    const result = await render(host, job("interpolate", { from: 0, to: 5 }, 30), 0.2);
    expect(result.first).toBe(0);
    expect(appended.reduce((a, b) => a + b.count, 0)).toBe(2 * 150);
    expect(log[0]).toBe("begin 1920x1080 60/1");
  });

  it("passe de 24 à 60 im./s sur plusieurs lots sans répéter ni sauter d'image", async () => {
    const { host, appended, calls, log } = fakeHost(24, 240);
    const result = await render(host, job("interpolate", { from: 0, to: 5 }, 24, 60), 0.05);
    expect(result.first).toBe(0);
    // 5 s à 60 im./s.
    expect(appended.reduce((a, b) => a + b.count, 0)).toBe(300);
    expect(log[0]).toBe("begin 1920x1080 60/1");
    expect(log.filter((l) => l.startsWith("frames")).length).toBeGreaterThan(1);
    // Chaque appel reçoit la cadence de sortie et, après le premier, l'origine de la sortie.
    expect(calls.every((c) => (c.params.fps as { num: number }).num === 60)).toBe(true);
    expect(calls[0]?.params.origin).toBeNull();
    expect(calls.slice(1).every((c) => c.params.origin === 0)).toBe(true);
  });

  it("compte le calcul par image source d'après le facteur de cadence", () => {
    expect(secondsPerSource(job("interpolate", { from: 0, to: 1 }, 30), 1)).toBe(1);
    expect(secondsPerSource(job("interpolate", { from: 0, to: 1 }, 30, 120), 1)).toBe(3);
    expect(secondsPerSource(job("interpolate", { from: 0, to: 1 }, 24, 60), 1)).toBe(1.5);
    expect(secondsPerSource(job("upscale", { from: 0, to: 1 }), 1)).toBe(1);
  });

  it("s'arrête à l'étape suivante quand le rendu est annulé, et annule l'encodage", async () => {
    const { host, calls, log } = fakeHost(25, 250);
    const signal = { cancelled: false };
    const call = host.call.bind(host);
    // Bouton Annuler pendant le premier calcul.
    host.call = (async (method: string, params: unknown, timeout: number) => {
      const result = await call(method, params, timeout);
      signal.cancelled = true;
      return result;
    }) as Host["call"];
    await expect(
      render(host, job("upscale", { from: 0, to: 4 }), 0.5, signal),
    ).rejects.toMatchObject({ code: "cancelled" });
    expect(calls).toHaveLength(1);
    expect(log.at(-1)).toBe("cancel");
    expect(log).not.toContain("finish");
  });

  it("annule l'encodage si un calcul échoue, et laisse passer l'erreur", async () => {
    const fail = fakeHost(25, 250, { failAt: 2 });
    await expect(render(fail.host, job("upscale", { from: 0, to: 4 }), 2)).rejects.toThrow(
      "gpu_failed",
    );
    expect(fail.log.at(-1)).toBe("cancel");
    expect(fail.log).not.toContain("finish");
  });

  it("borne chaque appel en temps et en volume écrit", () => {
    expect(framesPerCall(job("upscale", { from: 0, to: 1 }), 10)).toBe(4);
    // 4K : 33 Mo par image, 256 Mo au plus par appel.
    expect(framesPerCall(job("upscale", { from: 0, to: 1 }), 0.1)).toBe(8);
    expect(framesPerCall(job("interpolate", { from: 0, to: 1 }), 0.2)).toBe(16);
    expect(framesPerCall(job("upscale", { from: 0, to: 1 }), 900)).toBe(1);
  });
});
