import { G0, GAMMA_AIR, R_SPECIFIC_AIR } from "./constants";

const R0 = 6_356_766;
const G_R = G0 / R_SPECIFIC_AIR;

export type Air = {
  temperatureK: number;
  pressurePa: number;
  density: number;
  speedOfSound: number;
};

const LAYERS: [number, number, number, number][] = [
  [0, -0.0065, 288.15, 101_325],
  [11_000, 0.0, 216.65, 22_632.1],
  [20_000, 0.001, 216.65, 5_474.89],
  [32_000, 0.0028, 228.65, 868.019],
  [47_000, 0.0, 270.65, 110.906],
  [51_000, -0.0028, 270.65, 66.9389],
  [71_000, -0.002, 214.65, 3.95642],
];
const H_TOP = 84_852;
const T_TOP = 186.87;
const P_TOP = 0.3734;

function fromTp(t: number, p: number): Air {
  const density = p / (R_SPECIFIC_AIR * t);
  return {
    temperatureK: t,
    pressurePa: p,
    density,
    speedOfSound: Math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * t),
  };
}

function geopotential(hGeom: number) {
  return (R0 * hGeom) / (R0 + hGeom);
}

export function lookup(altitudeM: number): Air {
  const h = Math.max(-200, altitudeM);
  if (h < 0) return fromTp(288.15, 101_325);
  const hGeop = geopotential(h);
  if (hGeop > H_TOP || h > 86_000) {
    const f = Math.exp(-(Math.max(0, h - 86_000) / 6_000));
    const t = Math.min(1000, T_TOP + 0.002 * (h - 86_000));
    return fromTp(t, Math.max(1e-8, P_TOP * f));
  }
  let idx = 0;
  for (let i = 0; i < LAYERS.length; i++) {
    if (hGeop >= LAYERS[i][0]) idx = i;
  }
  const [hB, lapse, tB, pB] = LAYERS[idx];
  const dh = hGeop - hB;
  const t = tB + lapse * dh;
  const p =
    Math.abs(lapse) < 1e-9
      ? pB * Math.exp((-G_R * dh) / tB)
      : pB * Math.pow(t / tB, -G0 / (lapse * R_SPECIFIC_AIR));
  return fromTp(t, p);
}
