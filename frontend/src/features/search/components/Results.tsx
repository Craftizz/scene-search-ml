
"use client";
import styles from "./Results.module.css";
import FrameComponent from "./Frame";
import { SkeletonFrame } from "./Skeleton";
import SceneComponent from "./Scene";
import type { Frame, Scene } from "@/types/types";

type ResultsType = "scenes" | "frames";

function ResultsView({
  children,
  count,
  label,
}: {
  children: React.ReactNode;
  count: number;
  label: string;
}) {
  return (
    <div className={styles.results}>
      <div className={styles.resultsContent}>
        <p className={styles.resultsLabel}>Results</p>
        {count > 0 && <p className={styles.resultsCount}>Showing {count} {label}</p>}
        <div className={styles.frames}>
          {children}
        </div>
      </div>
    </div>
  );
}


export default function Results({
  items,
  type,
}: {
  items: Frame[] | Scene[];
  type: ResultsType;
}) {
  const count = items.length ?? 0;

  const renderItem = (item: Frame | Scene, index: number) => {
    return type === "scenes"
      ? <SceneComponent key={index} scene={item as Scene} />
      : <FrameComponent key={index} frame={item as Frame} />;
  };

  return (
    <ResultsView count={count} label={type}>
      {count > 0
        ? items.map(renderItem)
        : Array.from({ length: 6 }).map((_, i) => <SkeletonFrame />)}
    </ResultsView>
  );
}