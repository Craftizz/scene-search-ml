
"use client";
import styles from "./Results.module.css";
import Frame, { SkeletonFrame } from "./Frame";
import type { CaptionedFrame } from "@/types/types";

type Props = {
  frames?: CaptionedFrame[];
};

export default function Results({ frames = [] }: Props) {
  return (
    <div className={styles.results}>
      <div className={styles.resultsContent}>
        <p className={styles.resultsLabel}>Results</p>
        {frames.length > 0 && (
          <p className={styles.resultsCount}>Showing {frames.length} scenes</p>
        )}
        <div className={styles.frames}>
          {frames.length > 0 ? (
            frames.map((f, i) => <Frame key={i} frame={f} index={i} />)
          ) : (
            Array.from({ length: 6 }).map((_, i) => <SkeletonFrame key={i} />)
          )}
        </div>
      </div>
    </div>
  );
}