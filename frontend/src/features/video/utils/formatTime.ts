export function formatTime(t: number): string {
  if (!isFinite(t)) return "0:00";
  const s = Math.floor(t % 60).toString().padStart(2, "0");
  const m = Math.floor(t / 60);
  return `${m}:${s}`;
}
