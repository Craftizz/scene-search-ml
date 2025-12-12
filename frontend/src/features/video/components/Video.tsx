"use client";

import React, { useCallback, useEffect, useState } from "react";
import styles from "./Video.module.css";
import VideoDrop from "./VideoDrop";
import VideoDisplay from "./VideoDisplay";
import { extractFramesStream } from "../utils/extractFramesStream";
import { useVideoFrames } from "../context/VideoFramesContext";

export default function Video() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [extractedCount, setExtractedCount] = useState(0);
  const { frames: thumbnails, addFrame, clearFrames, setFrames, updateFrame } = useVideoFrames();

  const handleFileSelected = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setVideoSrc(url);
  }, []);

  useEffect(() => {
    if (!videoSrc) {
      clearFrames();
      return;
    }

    let cancelled = false;

    async function extract() {
      try {
        setExtracting(true);
        setExtractedCount(0);
        clearFrames();

        await extractFramesStream(
          videoSrc as string,
          1,
          320,
          (frame) => {
            if (cancelled) {
              try {
                URL.revokeObjectURL(frame.url);
              } catch {}
              return;
            }
            addFrame(frame);
            setExtractedCount((c) => c + 1);
          },
          {
            // when captions arrive, update the frame in the context
            onCaption: (updated) => {
              if (cancelled) return;
              try {
                updateFrame(updated);
              } catch (e) {
                console.warn("Failed to update frame with caption", e);
              }
            },
          }
        );

        if (!cancelled) setExtracting(false);
      } catch (err) {
        setExtracting(false);
        console.error("Frame extraction failed", err);
      }
    }

    extract();

    return () => {
      cancelled = true;
    };
  }, [videoSrc, addFrame, clearFrames, setFrames]);

  return (
    <div className={styles.video}>
      {!videoSrc && <VideoDrop onFileSelected={handleFileSelected} />}
      {videoSrc && <VideoDisplay src={videoSrc} />}
    </div>
  );
}