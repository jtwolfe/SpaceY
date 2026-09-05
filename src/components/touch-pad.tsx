import { useEffect, useRef } from "react";
import { useSpacey } from "@/game/store";

export function TouchPad() {
  const pilot = useSpacey((s) => s.pilot);
  const started = useSpacey((s) => s.started);
  const stick = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = stick.current;
    if (!el) return;
    const codes = new Set<string>();
    const apply = () => {
      const t = (window as unknown as { __controlsTest?: { setKeys?: (c: string[]) => void } }).__controlsTest;
      t?.setKeys?.(pilot === "manual" ? [...codes] : []);
    };
    const setFrom = (x: number, y: number, rect: DOMRect) => {
      codes.delete("KeyA");
      codes.delete("KeyD");
      codes.delete("KeyW");
      codes.delete("KeyS");
      const nx = (x - rect.left) / rect.width - 0.5;
      const ny = (y - rect.top) / rect.height - 0.5;
      if (nx < -0.12) codes.add("KeyA");
      if (nx > 0.12) codes.add("KeyD");
      if (ny < -0.12) codes.add("KeyW");
      if (ny > 0.12) codes.add("KeyS");
      apply();
    };
    const onDown = (e: PointerEvent) => {
      el.setPointerCapture(e.pointerId);
      setFrom(e.clientX, e.clientY, el.getBoundingClientRect());
    };
    const onMove = (e: PointerEvent) => {
      if (!el.hasPointerCapture(e.pointerId)) return;
      setFrom(e.clientX, e.clientY, el.getBoundingClientRect());
    };
    const onUp = () => {
      codes.clear();
      apply();
    };
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    return () => {
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
    };
  }, [pilot]);

  if (!started || pilot !== "manual") return null;

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-24 z-10 flex items-end justify-between px-4 md:hidden">
      <div
        ref={stick}
        className="pointer-events-auto size-28 rounded-lg bg-bg-elevated/80 ring-1 ring-border"
        style={{ touchAction: "none" }}
        aria-label="Attitude stick"
      />
      <div className="pointer-events-auto flex flex-col gap-2">
        <HoldButton label="Throttle" code="ShiftLeft" />
        <HoldButton label="3-eng" code="Digit3" />
      </div>
    </div>
  );
}

function HoldButton({ label, code }: { label: string; code: string }) {
  const down = (on: boolean) => {
    const t = (window as unknown as { __controlsTest?: { setKeys?: (c: string[]) => void } }).__controlsTest;
    t?.setKeys?.(on ? [code] : []);
  };
  return (
    <button
      type="button"
      className="min-h-11 min-w-24 rounded-sm bg-bg-elevated/90 px-3 font-mono text-xs text-steel ring-1 ring-border"
      style={{ touchAction: "none" }}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        down(true);
      }}
      onPointerUp={() => down(false)}
      onPointerCancel={() => down(false)}
    >
      {label}
    </button>
  );
}
