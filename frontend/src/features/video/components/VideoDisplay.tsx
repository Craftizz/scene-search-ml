"use client";
import { useCallback, useEffect, useRef } from "react";
import { useVideoFrames } from "../context/VideoFramesContext";
import { useVideoControls } from "../hooks/useVideoControls";
import { useVideoThumbnails } from "../hooks/useVideoThumbnails";
import { useContainerWidth } from "../hooks/useContainerWidth";
import VideoControls from "./VideoControls";
import ThumbnailProgress from "./ThumbnailProgress";

import styles from "./Video.module.css";


export default function VideoDisplay({ 
  src 
}:{ 
  src: string | null 
}) {

  const { registerVideo, frames } = useVideoFrames();
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const {
    playing,
    setPlaying,
    current,
    setCurrent,
    duration,
    setDuration,
    muted,
    togglePlay,
    seekTo,
    toggleMute,
  } = useVideoControls(videoRef);

  const progressRef = useRef<HTMLDivElement | null>(null);
  const thumbRef = useRef<HTMLDivElement | null>(null);
  const containerWidth = useContainerWidth(thumbRef, progressRef);
  const thumbnails = useVideoThumbnails(frames, containerWidth, duration);

  useEffect(() => {
    registerVideo?.(videoRef.current ?? null);
  }, [videoRef.current, registerVideo]);

  const handleTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    setCurrent(e.currentTarget.currentTime);
  };

  const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const v = e.currentTarget;
    setDuration(isFinite(v.duration) ? v.duration : null);
  };

  const onProgressClick = useCallback(
    (e: React.MouseEvent) => {
      const el = thumbRef.current ?? progressRef.current;
      if (!el || !duration) return;
      const rect = el.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      const pct = rect.width > 0 ? x / rect.width : 0;
      const seconds = pct * duration;
      seekTo(seconds);
    },
    [duration, seekTo]
  );

  const handlePlay = () => setPlaying(true);
  const handlePause = () => setPlaying(false);

  return (
    <div className={styles.playerWrapper}>
      {src && (
        <div className={styles.videoArea}>
          <div className={styles.videoContainer}>
            <video
              ref={(el) => {
                videoRef.current = el;
                registerVideo?.(el ?? null);
              }}
              className={styles.videoplayer}
              src={src}
              preload="metadata"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onPlay={handlePlay}
              onPause={handlePause}
              onClick={togglePlay}
            />

            <VideoControls
              playing={playing}
              muted={muted}
              onTogglePlay={togglePlay}
              onToggleMute={toggleMute}
            />
          </div>

          <ThumbnailProgress
            thumbnails={thumbnails}
            current={current}
            duration={duration}
            thumbRef={thumbRef}
            progressRef={progressRef}
            onProgressClick={onProgressClick}
            onSeek={seekTo}
          />
        </div>
      )}
    </div>
  );
}
