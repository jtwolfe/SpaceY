/** Falcon 9 Block 5 first-stage class numbers — public figures, not telemetry. */

export const EARTH_RADIUS_EQ = 6_378_137;
export const G0 = 9.80665;
export const R_SPECIFIC_AIR = 287.05287;
export const GAMMA_AIR = 1.4;

export const PAD_LAT_DEG = 28.4856;
export const PAD_LON_DEG = -80.5444;

/** LZ-1 concrete disk — 282 ft. Apron is extra packed soil around it. */
export const LZ1_DIAMETER_M = 86;
export const LZ1_APRON_M = 15.24;
/** LZ-2 sits ~1,017 ft northwest of LZ-1. */
export const LZ2_NORTHWEST_M = 310;

export const STAGE_DIAMETER_M = 3.66;
export const STAGE_LENGTH_M = 47.0;
export const STAGE_RADIUS_M = STAGE_DIAMETER_M * 0.5;
export const REF_AREA_M2 = Math.PI * STAGE_RADIUS_M * STAGE_RADIUS_M;

export const DRY_MASS_KG = 25_600;
export const START_FUEL_KG = 40_000;
export const HOVER_FUEL_KG = 2_800;
export const SUICIDE_FUEL_KG = 8_500;
export const GLIDE_FUEL_KG = 18_000;

export const MERLIN_THRUST_SL_N = 845_000;
export const MERLIN_THRUST_VAC_N = 914_000;
export const MERLIN_ISP_SL_S = 282;
export const MERLIN_ISP_VAC_S = 311;
export const N_ENGINES_ENTRY = 3;
export const N_ENGINES_LANDING = 1;
export const THROTTLE_MIN = 0.4;
export const THROTTLE_MAX = 1.0;
export const GIMBAL_MAX_RAD = (5 * Math.PI) / 180;
export const GIMBAL_SLEW_RAD_S = (25 * Math.PI) / 180;
export const FIN_SLEW_RAD_S = (40 * Math.PI) / 180;

export const ENGINE_MIN_BURN_S = 2.5;
export const ENGINE_RESTART_DELAY_S = 6.0;

export const FIN_AREA_M2 = 1.8;
export const FIN_MAX_DEFLECT_RAD = (28 * Math.PI) / 180;
export const FIN_ARM_M = 18.5;
export const FIN_CD0 = 0.55;
export const FIN_CL_DELTA = 0.55;
export const BODY_CP_X_M = -1.6;
export const FIN_Q_FADE_PA = 4_000;
export const FIN_Q_FULL_PA = 8_000;
/** Unlit high-q divert window: score range killed above this, tax late leans below it. */
export const COAST_ALT_M = 400;

export const Q_DESTROY_PA = 110_000;
export const G_DESTROY = 12;
export const Q_ALPHA_DESTROY = 2_800;
export const AOA_DESTROY_RAD = (42 * Math.PI) / 180;
export const Q_FOR_AOA_DESTROY_PA = 12_000;
export const RATE_DESTROY_RAD_S = 3.5;

export const RCS_ANG_ACCEL = 0.55;
export const RCS_Q_HANDOFF_PA = 160_000;

export const SUCCESS_ENGINE_ALT_M = 12;
export const SUCCESS_SPEED_MPS = 16;
export const SUCCESS_HVEL_MPS = 8;
export const SUCCESS_PAD_OFFSET_M = 35;
export const SUCCESS_TILT_RAD = (16 * Math.PI) / 180;
export const IMPACT_SPEED_MPS = 20;
export const IMPACT_TILT_RAD = (20 * Math.PI) / 180;
/** Engine-nozzle height at gear-on-deck. Landing burn stays lit above this. */
export const GEAR_ENGINE_ALT_M = 8;

export const DT = 0.02;
export const POLICY_DT = 0.1;

export function wetMass(fuel: number) {
  return DRY_MASS_KG + Math.max(0, fuel);
}

export function inertiaDiag(mass: number) {
  const r2 = STAGE_RADIUS_M * STAGE_RADIUS_M;
  const l2 = STAGE_LENGTH_M * STAGE_LENGTH_M;
  return {
    x: 0.5 * mass * r2,
    y: (1 / 12) * mass * (3 * r2 + l2),
    z: (1 / 12) * mass * (3 * r2 + l2),
  };
}
