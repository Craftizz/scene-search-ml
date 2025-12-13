import { useState, useCallback } from "react";
import styles from "./ThumbnaillProgress.module.css";
import { CaptionedFrame } from "@/types/types";
import { formatTimecode } from "@/utils/formatTimecode";

interface ThumbnailProgressProps {
  thumbnails: (CaptionedFrame | null)[];
  current: number;
  duration: number | null;
  thumbRef: React.RefObject<HTMLDivElement | null>;
  progressRef: React.RefObject<HTMLDivElement | null>;
  onProgressClick: (e: React.MouseEvent) => void;
  onSeek: (seconds: number) => void;
}

export default function ThumbnailProgress({
  thumbnails,
  current,
  duration,
  thumbRef,
  progressRef,
  onProgressClick,
  onSeek,
}: ThumbnailProgressProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [hoverPosition, setHoverPosition] = useState<number | null>(null);

  

  const getPositionFromEvent = (e: React.MouseEvent): number | null => {
    const el = thumbRef.current ?? progressRef.current;
    if (!el || !duration) return null;

    const rect = el.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const pct = rect.width > 0 ? x / rect.width : 0;
    return pct * duration;
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    const seconds = getPositionFromEvent(e);
    if (seconds !== null) {
      onSeek(seconds);
    }
  };

  const handleGlobalMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const seconds = getPositionFromEvent(e);
      if (seconds !== null) {
        onSeek(seconds);
      }
    },
    [duration, onSeek, thumbRef, progressRef]
  );

  const handleGlobalMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleMouseEnter = (e: React.MouseEvent) => {
    const seconds = getPositionFromEvent(e);
    if (seconds !== null) {
      setHoverPosition(seconds);
    }
  };

  const handleMouseLeave = () => {
    setHoverPosition(null);
  };

  const handleMouseMoveHover = (e: React.MouseEvent) => {
    if (!isDragging) {
      const seconds = getPositionFromEvent(e);
      if (seconds !== null) {
        setHoverPosition(seconds);
      }
    }
  };

  return (
    <div
      className={styles.progressWrap}
      ref={progressRef}
      onClick={onProgressClick}
      onMouseDown={handleMouseDown}
      onMouseMove={isDragging ? handleGlobalMouseMove : handleMouseMoveHover}
      onMouseUp={handleGlobalMouseUp}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >

      <div className={styles.progressNumbers}>
          <span className={styles.timeCurrent}>
            {formatTimecode(current)}
          </span>
          <span className={styles.timeDuration}>
            {duration ? formatTimecode(duration) : "00:00:00"}
          </span>
        </div>

      <div className={styles.thumbStripContainer}>
        
        {/* Thumbnail strip as progress - stable rendering */}
        <div className={styles.thumbStrip} ref={thumbRef}>
          {thumbnails?.map((frame, i) => {
            // Calculate boundaries: each thumb spans from midpoint to previous to midpoint to next
            const prevTimestamp = i > 0 && thumbnails[i - 1] ? thumbnails[i - 1]!.timestamp : 0;
            const currTimestamp = frame?.timestamp ?? ((i + 0.5) / thumbnails.length) * (duration ?? 0);
            const nextTimestamp = i < thumbnails.length - 1 && thumbnails[i + 1] ? thumbnails[i + 1]!.timestamp : (duration ?? 0);
            
            const leftEdge = i === 0 ? 0 : (prevTimestamp + currTimestamp) / 2;
            const rightEdge = i === thumbnails.length - 1 ? (duration ?? 0) : (currTimestamp + nextTimestamp) / 2;
            
            const leftPercent = duration ? (leftEdge / duration) * 100 : 0;
            const widthPercent = duration ? ((rightEdge - leftEdge) / duration) * 100 : 100 / thumbnails.length;
            
            return frame ? (
              <img
                key={`thumb-${frame.timestamp}-${i}`}
                src={frame.url}
                className={styles.thumbItem}
                style={{
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                }}
                alt={`Frame at ${Math.floor(frame.timestamp)}s`}
                draggable={false}
              />
            ) : (
              <div
                key={`placeholder-${i}`}
                className={styles.thumbPlaceholder}
                style={{
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                }}
              />
            );
          })}

          {hoverPosition !== null && !isDragging && (
            <div
              className={styles.progressHover}
              style={{
                left: duration ? `${(hoverPosition / duration) * 100}%` : "0%",
              }}
              aria-hidden="true"
            />
          )}

          <div
            className={styles.progressOverlay}
            style={{ left: duration ? `${(current / duration) * 100}%` : "0%" }}
            aria-hidden="true"
          />

          <div
            className={styles.progressMarker}
            style={{ left: duration ? `${(current / duration) * 100}%` : "0%" }}
            aria-hidden="true"
          />
        </div>
      </div>
    </div>
  );
}
