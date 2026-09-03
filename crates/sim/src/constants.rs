//! Approximate Falcon 9 Block 5 first-stage class numbers and Earth constants.
//!
//! These are **public, order-of-magnitude figures** assembled from SpaceX user's
//! guides, FAA/environmental filings, and commonly cited secondary sources
//! (Wikipedia summaries of those documents). They are **not** live telemetry
//! and are **not** a proprietary aero database. Enough to dimension a coherent
//! 6DOF landing sim — not a NASA-grade reconstruction.

use crate::math::Vec3;

/// WGS-84 / IERS conventional constants.
pub const EARTH_MU: f64 = 3.986_004_418e14; // m^3/s^2, IERS 2010
pub const EARTH_RADIUS_EQ: f64 = 6_378_137.0; // m, WGS-84 a
pub const EARTH_RADIUS_POL: f64 = 6_356_752.314_245; // m, WGS-84 b
pub const EARTH_E2: f64 = 6.694_379_990_14e-3; // WGS-84 first eccentricity²
pub const EARTH_J2: f64 = 1.082_626_68e-3; // EGM96 J2
pub const EARTH_OMEGA: f64 = 7.292_115e-5; // rad/s, sidereal
pub const G0: f64 = 9.80665; // m/s², standard gravity (Isp definition)
pub const R_SPECIFIC_AIR: f64 = 287.05287; // J/(kg·K), US76 dry air
pub const GAMMA_AIR: f64 = 1.4;

/// Landing Zone 1, Cape Canaveral SFS — public coordinates (~28.4856 N, 80.5444 W).
pub const PAD_LAT_DEG: f64 = 28.4856;
pub const PAD_LON_DEG: f64 = -80.5444;
pub const PAD_ALT_M: f64 = 4.0;

/// Stage geometry. Falcon 9 first stage is 3.66 m diameter; length of the
/// booster + interstage is commonly given as ~47 m (full stack ~70 m).
pub const STAGE_DIAMETER_M: f64 = 3.66;
pub const STAGE_LENGTH_M: f64 = 47.0;
pub const STAGE_RADIUS_M: f64 = STAGE_DIAMETER_M * 0.5;
pub const REF_AREA_M2: f64 = std::f64::consts::PI * STAGE_RADIUS_M * STAGE_RADIUS_M;

/// Mass breakdown (approximate).
///
/// * Stage-1 dry mass is not officially published as a single number; public
///   estimates for Block 5 cluster around 22–27 t. We use 25 600 kg.
/// * Ascent usable propellant is ~395–411 t LOX/RP-1. This scenario starts
///   *after* ascent + boostback, so tanks are mostly empty.
/// * Remaining propellant at the high-altitude start (~40 t) is a conservative
///   RTLS/ASDS-class load: enough for a 3-engine entry burn and a 1–3 engine
///   landing burn with margin, not a full tank.
pub const DRY_MASS_KG: f64 = 25_600.0;
pub const START_FUEL_KG: f64 = 40_000.0;
pub const TANK_CAPACITY_KG: f64 = 395_700.0;

/// Merlin 1D (sea-level engine) — SpaceX public figures for Block 5:
/// ~845 kN SL / ~914 kN vac, Isp ~282 s SL / ~311 s vac (user's guide / ASDS
/// filings; Wikipedia aggregates the same numbers). Landing uses the center
/// engine or a 3-engine cluster; we model up to 3 engines.
pub const MERLIN_THRUST_SL_N: f64 = 845_000.0;
pub const MERLIN_THRUST_VAC_N: f64 = 914_000.0;
pub const MERLIN_ISP_SL_S: f64 = 282.0;
pub const MERLIN_ISP_VAC_S: f64 = 311.0;
pub const N_ENGINES_ENTRY: u8 = 3;
pub const N_ENGINES_LANDING: u8 = 1;
pub const THROTTLE_MIN: f64 = 0.40; // Merlin 1D deep-throttle ballpark
pub const THROTTLE_MAX: f64 = 1.00;
pub const GIMBAL_MAX_RAD: f64 = 5.0 * std::f64::consts::PI / 180.0;
/// Once lit, a Merlin cannot chatter off at the 10 Hz policy tick.
pub const ENGINE_MIN_BURN_S: f64 = 2.5;
/// Shutdown → restart delay. PWM-by-relight is not a throttle.
pub const ENGINE_RESTART_DELAY_S: f64 = 6.0;
/// Fitness cost per extra ignition after the first. A land still wins.
pub const RELIGHT_FITNESS: f64 = 320.0;

/// Titanium grid fins (Block 5): four surfaces, roughly 1.2 × 1.5 m planform
/// in public photos / patent drawings. Lattice Cd at δ=0 is the weathercock
/// (flow through the waffle still has a lot of wetted area). Not a CFD table.
pub const N_GRID_FINS: u8 = 4;
pub const FIN_AREA_M2: f64 = 1.8;
pub const FIN_MAX_DEFLECT_RAD: f64 = 28.0 * std::f64::consts::PI / 180.0;
pub const FIN_ARM_M: f64 = 18.5; // CG → fin plane along +X (interstage end)
/// Axial lattice drag coefficient of one deployed grid at zero deflection.
pub const FIN_CD0: f64 = 0.55;
/// Side-force slope vs deflection (rad⁻¹), order-of-magnitude Cl_δ.
pub const FIN_CL_DELTA: f64 = 0.55;
/// Body+engine CP along +X. Negative = engine-side of the CG (destabilizing
/// alone). Grids at +FIN_ARM_M pull the net CP aft of the CG.
pub const BODY_CP_X_M: f64 = -1.6;

/// Structural / load limits used when destruction is enabled. Falcon 9 max-Q
/// on ascent is publicly ~30 kPa; reentry with an entry burn stays in a
/// similar band. We trip a bit higher so a clean entry survives and an
/// uncontrolled skip does not.
pub const Q_DESTROY_PA: f64 = 110_000.0;
pub const G_DESTROY: f64 = 12.0;
pub const Q_ALPHA_DESTROY: f64 = 2_800.0; // kPa·deg
pub const AOA_DESTROY_RAD: f64 = 42.0 * std::f64::consts::PI / 180.0;
pub const Q_FOR_AOA_DESTROY_PA: f64 = 12_000.0;
pub const RATE_DESTROY_RAD_S: f64 = 3.5;

/// Cold-gas RCS angular acceleration (vacuum attitude hold). F9-class
/// nitrogen thrusters; not deducted from RP-1. Autopilot demo only.
pub const RCS_ANG_ACCEL: f64 = 0.18;
/// RCS fades as Q rises so grid fins own the air.
pub const RCS_Q_HANDOFF_PA: f64 = 160_000.0;
pub const RTLS_TIMEOUT_S: f64 = 420.0;

/// Landing success box (engine-bell / pad frame). A toy-soft F9 landing:
/// legs ~8 m, a few m/s, on the pad, upright. Not a 26 m/s hard hit.
pub const SUCCESS_ENGINE_ALT_M: f64 = 12.0;
pub const SUCCESS_SPEED_MPS: f64 = 8.0;
pub const SUCCESS_HVEL_MPS: f64 = 4.0;
pub const SUCCESS_PAD_OFFSET_M: f64 = 20.0;
pub const SUCCESS_TILT_RAD: f64 = 12.0 * std::f64::consts::PI / 180.0;
/// Structural slap. Hotter than the success box, still well below a 26 m/s RUD.
/// A 10 m/s pad sit-down is a miss, not an explosion — otherwise CMA treats
/// every near-land as a fireball.
pub const IMPACT_SPEED_MPS: f64 = 20.0;
pub const IMPACT_TILT_RAD: f64 = 20.0 * std::f64::consts::PI / 180.0;

pub const HOVER_TIMEOUT_S: f64 = 45.0;
pub const SUICIDE_TIMEOUT_S: f64 = 90.0;
pub const GLIDE_TIMEOUT_S: f64 = 200.0;
pub const HOVER_FUEL_KG: f64 = 2_800.0;
pub const SUICIDE_FUEL_KG: f64 = 8_500.0;
pub const GLIDE_FUEL_KG: f64 = 18_000.0;

pub fn wet_mass(fuel: f64) -> f64 {
    DRY_MASS_KG + fuel.max(0.0)
}

pub fn inertia_diag(mass: f64) -> Vec3 {
    let r2 = STAGE_RADIUS_M * STAGE_RADIUS_M;
    let l2 = STAGE_LENGTH_M * STAGE_LENGTH_M;
    Vec3::new(
        0.5 * mass * r2,
        (1.0 / 12.0) * mass * (3.0 * r2 + l2),
        (1.0 / 12.0) * mass * (3.0 * r2 + l2),
    )
}
