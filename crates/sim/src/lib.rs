//! SpaceY: Falcon 9-class first-stage 6DOF physics + CMA-ES trainer.
//!
//! Compiled to `wasm32-unknown-unknown` for in-browser training. Native
//! `cargo test` covers atmosphere, frames, and RTLS episode termination.

pub mod atmosphere;
pub mod cmaes;
pub mod constants;
pub mod earth;
pub mod guidance;
pub mod math;
pub mod policy;
pub mod scenario;
pub mod sim;
pub mod vehicle;
pub mod wind;

#[cfg(target_arch = "wasm32")]
mod wasm_api;
#[cfg(target_arch = "wasm32")]
pub use wasm_api::Engine;
