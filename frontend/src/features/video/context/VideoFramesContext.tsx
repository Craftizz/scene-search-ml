"use client";

import { createContext, useCallback, useContext, useRef, useState } from "react";
import { Frame, Scene } from "@/types/types";

type VideoFramesContextValue = {
  frames: Frame[];
  scenes: Scene[];
  addFrame: (f: Frame) => void;
  addScene: (s: Scene) => void;
  clearScenes: () => void;
  setScenes: (list: Scene[]) => void;
  clearFrames: () => void;
  setFrames: (list: Frame[]) => void;
  updateFrame: (f: Frame) => void;
  registerVideo?: (el: HTMLVideoElement | null) => void;
  seekTo?: (seconds: number, play?: boolean) => void;
};

const VideoFramesContext = createContext<VideoFramesContextValue | undefined>(
  undefined
);

export const VideoFramesProvider: React.FC<React.PropsWithChildren<{}>> = ({
  children,
}) => {
  const [frames, setFramesState] = useState<Frame[]>([]);
  const [scenes, setScenesState] = useState<Scene[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const addFrame = useCallback((frame: Frame) => {
    setFramesState((prev) => [...prev, frame]);
  }, []);

    const addScene = useCallback((scene: Scene) => {
    setScenesState((prev) => [...prev, scene]);
  }, []);

  const clearFrames = useCallback(() => {
    setFramesState((prev) => {
      prev.forEach((p) => {
        try {
          URL.revokeObjectURL(p.url);
        } catch {}
      });
      return [];
    });
  }, []);

  const clearScenes = useCallback(() => {
    setScenesState(() => []);
  }, []);

  const setFrames = useCallback((list: Frame[]) => {
    setFramesState((prev) => {
      prev.forEach((p) => {
        try {
          URL.revokeObjectURL(p.url);
        } catch {}
      });
      return list;
    });
  }, []);

  const setScenes = useCallback((list: Scene[]) => {
    setScenesState(() => list);
  }, []);

  const updateFrame = useCallback((frame: Frame) => {
    setFramesState((prev) => prev.map((p) => (p.timestamp === frame.timestamp && p.url === frame.url ? frame : p)));
  }, []);

  const registerVideo = useCallback((el: HTMLVideoElement | null) => {
    videoRef.current = el;
  }, []);

  const seekTo = useCallback((seconds: number, play = false) => {
    try {
      const v = videoRef.current;
      if (!v) return;
      if (!isFinite(seconds) || seconds < 0) return;
      if (typeof v.duration === "number" && !isNaN(v.duration)) {
        v.currentTime = Math.min(seconds, v.duration);
      } else {
        v.currentTime = seconds;
      }

      if (play) {
        // attempt to play after seeking; ignore promise rejection
        try {
          const p = v.play();
          if (p && typeof p.catch === "function") p.catch(() => {});
        } catch {}
      }
    } catch (e) {
      console.warn("seekTo failed", e);
    }
  }, []);

  return (
    <VideoFramesContext.Provider
      value={{ frames, scenes, addFrame, addScene, clearFrames, clearScenes, setFrames, setScenes, updateFrame, registerVideo, seekTo }}
    >
      {children}
    </VideoFramesContext.Provider>
  );
};

export function useVideoFrames() {
  const ctx = useContext(VideoFramesContext);
  if (!ctx)
    throw new Error("useVideoFrames must be used within VideoFramesProvider");
  return ctx;
}
