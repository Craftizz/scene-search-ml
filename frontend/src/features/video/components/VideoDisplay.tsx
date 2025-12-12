"use client";
import styles from "./Video.module.css";
import { useVideoFrames } from "../context/VideoFramesContext";

type Props = {
  src: string | null;
};

export default function VideoDisplay({ src }: Props) {
  const { registerVideo } = useVideoFrames();
  return (
    <div className={styles.playerWrapper}>
      {src && (
        <video
          ref={(el) => registerVideo?.(el)}
          className={styles.video}
          src={src}
          controls
          preload="metadata"
        />
      )}
    </div>
  );
}
