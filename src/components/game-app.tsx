import { useEffect, useRef } from "react";
import { useSpacey } from "@/game/store";
import { Dock } from "./dock";
import { Hud } from "./hud";
import { GymPanel } from "./gym-panel";
import { StartOverlay } from "./start-overlay";
import { TouchPad } from "./touch-pad";

export function GameApp() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const started = useSpacey((s) => s.started);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let dead = false;
    let dispose = () => {};
    void import("@/game/runtime").then(({ createRuntime }) => {
      if (dead || !canvasRef.current) return;
      const rt = createRuntime(canvasRef.current, {
        getPilot: () => useSpacey.getState().pilot,
        getMission: () => useSpacey.getState().mission,
        getCam: () => useSpacey.getState().cam,
        getWarp: () => useSpacey.getState().warp,
        getPaused: () => useSpacey.getState().paused,
        getStarted: () => useSpacey.getState().started,
        getSeed: () => useSpacey.getState().seed,
        getBrain: () => useSpacey.getState().brain,
        getBrainEpoch: () => useSpacey.getState().brainEpoch,
        onSnap: (snap) => useSpacey.getState().setSnap(snap),
        onBrain: (b) => useSpacey.getState().setBrain(b),
        onGym: (g) => useSpacey.getState().setGym(g),
        onNote: (n) => useSpacey.getState().setGenNote(n),
      });
      dispose = () => rt.dispose();
    });
    return () => {
      dead = true;
      dispose();
    };
  }, []);

  return (
    <div className="relative h-[100dvh] min-h-[100dvh] overflow-hidden bg-bg text-fg" style={{ touchAction: "none" }}>
      <canvas ref={canvasRef} className="block h-full w-full" />
      <Hud />
      <GymPanel />
      {started ? <Dock /> : null}
      <TouchPad />
      <StartOverlay />
    </div>
  );
}
