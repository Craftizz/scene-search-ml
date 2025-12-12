"use client";

import styles from "./Frame.module.css";
import { useVideoFrames } from "@/features/video/context/VideoFramesContext";
import type { CaptionedFrame } from "@/types/types";
import { formatTimecode } from "@/features/search/utils/formatTimecode";

type Props = {
  frame: CaptionedFrame;
  index?: number;
  onSelect?: (frame: CaptionedFrame, index?: number) => void;
};

export default function Frame({ frame, index = 0, onSelect }: Props) {
  if (!frame) return null;

  const { seekTo } = useVideoFrames();

  return (
    <div className={styles.thumbnailItem}>
      <img
        src={frame.url}
        alt={`frame-${index}`}
        className={styles.thumbnailImage}
        onClick={() => {
          try {
            seekTo?.(frame.timestamp, true);
          } catch {}
            onSelect?.(frame, index);
        }}
      />
      <div className={styles.thumbnailDescription}>
        <p className={styles.thumbnailTimestamp}>{formatTimecode(frame.timestamp)}</p>
        {frame.caption ? (
          <p className={styles.thumbnailCaption}>{frame.caption}</p>
        ) : (
          <SkeletonCaption />
        )}
      </div>
    </div>
  );
}

export function SkeletonFrame() {
  return (
    <div className={`${styles.thumbnailItem || ""} ${styles.skeletonItem || ""}`}>
      <div className={styles.skeletonBox} />
    </div>
  );
}

export function SkeletonCaption() {
  return <div className={styles.skeletonCaption} />;
}
