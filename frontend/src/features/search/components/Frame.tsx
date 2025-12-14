"use client";

import { useVideoFrames } from "@/features/video/context/VideoFramesContext";
import type { Frame } from "@/types/types";
import { formatTimecode } from "@/utils/formatTimecode";

import styles from "./Frame.module.css";

type Props = {
  frame: Frame;
  index?: number;
  onSelect?: (frame: Frame, index?: number) => void;
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
      </div>
    </div>
  );
}
