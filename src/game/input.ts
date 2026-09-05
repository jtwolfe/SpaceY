export type KeySet = Set<string>;

export function createInput() {
  const keys: KeySet = new Set();
  const onDown = (e: KeyboardEvent) => {
    if (e.repeat) return;
    keys.add(e.code);
    if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code)) {
      e.preventDefault();
    }
  };
  const onUp = (e: KeyboardEvent) => {
    keys.delete(e.code);
  };
  const onBlur = () => keys.clear();
  window.addEventListener("keydown", onDown);
  window.addEventListener("keyup", onUp);
  window.addEventListener("blur", onBlur);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) keys.clear();
  });
  return {
    keys,
    dispose() {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", onBlur);
    },
  };
}

export type Actions = {
  pitch: number;
  yaw: number;
  roll: number;
  throttleDelta: number;
  fireToggle: boolean;
  engines3: boolean;
};

export function readActions(keys: KeySet): { pitch: number; yaw: number; roll: number; throttleUp: boolean; throttleDown: boolean; engines3: boolean } {
  let pitch = 0;
  let yaw = 0;
  let roll = 0;
  if (keys.has("KeyW") || keys.has("ArrowUp")) pitch += 1;
  if (keys.has("KeyS") || keys.has("ArrowDown")) pitch -= 1;
  // A = left on chase cam = +yaw in this mapping (tilt top west).
  if (keys.has("KeyA") || keys.has("ArrowLeft")) yaw += 1;
  if (keys.has("KeyD") || keys.has("ArrowRight")) yaw -= 1;
  if (keys.has("KeyQ")) roll -= 1;
  if (keys.has("KeyE")) roll += 1;
  return {
    pitch,
    yaw,
    roll,
    throttleUp: keys.has("ShiftLeft") || keys.has("ShiftRight") || keys.has("Space"),
    throttleDown: keys.has("ControlLeft") || keys.has("ControlRight") || keys.has("KeyZ"),
    engines3: keys.has("Digit3"),
  };
}
