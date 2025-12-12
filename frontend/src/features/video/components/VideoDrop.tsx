
"use client";
import React, { useRef, useState } from "react";
import styles from "./VideoDrop.module.css";

type Props = {
	onFileSelected: (file: File) => void;
};

export default function VideoDrop({ onFileSelected }: Props) {
	
	const inputRef = useRef<HTMLInputElement | null>(null);
	const [isDragging, setIsDragging] = useState(false);

	const onDrop: React.DragEventHandler<HTMLDivElement> = (e) => {
		e.preventDefault();
		e.stopPropagation();
		setIsDragging(false);
		const file = e.dataTransfer?.files?.[0];
		if (file && file.type.startsWith("video/")) {
			onFileSelected(file);
		}
	};

	const onDragOver: React.DragEventHandler<HTMLDivElement> = (e) => {
		e.preventDefault();
		e.stopPropagation();
		setIsDragging(true);
	};

	const onDragLeave: React.DragEventHandler<HTMLDivElement> = (e) => {
		e.preventDefault();
		e.stopPropagation();
		setIsDragging(false);
	};

	const onFileChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
		const file = e.target.files?.[0];
		if (file && file.type.startsWith("video/")) {
			onFileSelected(file);

			if (inputRef.current) inputRef.current.value = "";
		}
	};

	return (
    <div
      className={styles.dropzone}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      role="button"
      aria-label="Drop a video file here"
      onClick={() => inputRef.current?.click()}
    >
      <div className={`${styles.dropzoneInner} ${isDragging ? styles.dropzoneInnerDragging : ""}`}>
        <p className={styles.dropzoneHeading}>Drop Video</p>
        <p className={styles.dropzoneHint}>MP4, WebM, or MOV</p>
		<p className={styles.dropzoneSubhint}>or click to select</p>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className={styles.hiddenInput}
          onChange={onFileChange}
        />
      </div>
    </div>
  );
}



