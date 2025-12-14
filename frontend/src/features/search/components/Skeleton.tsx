"use client";

import styles from "./Frame.module.css";

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

export default {
  SkeletonFrame,
  SkeletonCaption,
};
