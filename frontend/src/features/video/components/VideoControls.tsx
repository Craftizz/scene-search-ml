import React from "react";
import styles from "./Video.module.css";
import PlayIcon from "@/components/icons/PlayIcon";
import PauseIcon from "@/components/icons/PauseIcon";
import VolumeOn from "@/components/icons/VolumeOn";
import VolumeOff from "@/components/icons/VolumeOff";

interface VideoControlsProps {
  playing: boolean;
  muted: boolean;
  onTogglePlay: () => void;
  onToggleMute: () => void;
}

export default function VideoControls({
  playing,
  muted,
  onTogglePlay,
  onToggleMute
}: VideoControlsProps) {
  return (
    <div className={styles.controlsOverlay}>
      <button onClick={onTogglePlay} className={styles.controlButton}>
        {playing ? <PauseIcon /> : <PlayIcon />}
      </button>

      <div className={styles.controlsRight}>
        <div className={styles.volumeWrap}>
          <button onClick={onToggleMute} className={styles.controlButton}>
            {muted ? <VolumeOff /> : <VolumeOn />}
          </button>
        </div>
      </div>
    </div>
  );
}
