"use client";

import React, { createContext, useCallback, useContext, useState } from "react";
import { CaptionedFrame } from "@/types/types";

type VideoFramesContextValue = {
  frames: CaptionedFrame[];
  addFrame: (f: CaptionedFrame) => void;
  clearFrames: () => void;
  setFrames: (list: CaptionedFrame[]) => void;
  updateFrame: (f: CaptionedFrame) => void;
  registerVideo?: (el: HTMLVideoElement | null) => void;
  seekTo?: (seconds: number, play?: boolean) => void;
};

const VideoFramesContext = createContext<VideoFramesContextValue | undefined>(
  undefined
);

export const VideoFramesProvider: React.FC<React.PropsWithChildren<{}>> = ({
  children,
}) => {
  const [frames, setFramesState] = useState<CaptionedFrame[]>([]);
  const videoRef = React.useRef<HTMLVideoElement | null>(null);

  const addFrame = useCallback((frame: CaptionedFrame) => {
    setFramesState((prev) => [...prev, frame]);
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

  const setFrames = useCallback((list: CaptionedFrame[]) => {
    setFramesState((prev) => {
      prev.forEach((p) => {
        try {
          URL.revokeObjectURL(p.url);
        } catch {}
      });
      return list;
    });
  }, []);

  const updateFrame = useCallback((frame: CaptionedFrame) => {
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
      value={{ frames, addFrame, clearFrames, setFrames, updateFrame, registerVideo, seekTo }}
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
