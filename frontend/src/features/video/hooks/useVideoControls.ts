import React from "react";

export function useVideoControls(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [playing, setPlaying] = React.useState(false);
  const [current, setCurrent] = React.useState(0);
  const [duration, setDuration] = React.useState<number | null>(null);
  const [muted, setMuted] = React.useState(false);

  function togglePlay() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused || v.ended) {
      v.play().catch(() => {});
    } else {
      v.pause();
    }
  }

  function seekTo(seconds: number) {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(seconds, isFinite(v.duration) ? v.duration : seconds));
  }

  function toggleMute() {
    const v = videoRef.current;
    if (!v) return;
    const next = !muted;
    setMuted(next);
    v.muted = next;
  }

  return {
    playing,
    setPlaying,
    current,
    setCurrent,
    duration,
    setDuration,
    muted,
    setMuted,
    togglePlay,
    seekTo,
    toggleMute
  };
}
