"use client";

import type { Scene } from "@/types/types";
import { useVideoFrames } from "@/features/video/context/VideoFramesContext";
import { formatTimecode } from "@/utils/formatTimecode";
import { SkeletonCaption } from "./Skeleton";

import styles from "./Scene.module.css";

type Props = {
  scene: Scene;
  onSelect?: (scene: Scene) => void;
};

export default function SceneComponent({ scene, onSelect }: Props) {
  if (!scene) return null;

  const { seekTo } = useVideoFrames();

  const start = Number(scene.timestamp || 0);
  const end = Number(scene.end_timestamp || 0);

  return (
    <div className={styles.thumbnailItem}>
      <img
        src={scene.url}
        alt={`scene-${scene.id}`}
        className={styles.thumbnailImage}
        onClick={() => {
          try {
            seekTo?.(start, true);
          } catch {}
          onSelect?.(scene);
        }}
      />
      <div className={styles.thumbnailDescription}>
        <p className={styles.thumbnailScene}>Scene {scene.id}</p>
        <p className={styles.thumbnailTimestamp}>
          {formatTimecode(start)}
          {end ? ` — ${formatTimecode(end)}` : ""}
        </p>
        {scene.caption ? (
          <p className={styles.thumbnailCaption}>{scene.caption}</p>
        ) : (
          <SkeletonCaption />
        )}
      </div>
    </div>
  );
}
