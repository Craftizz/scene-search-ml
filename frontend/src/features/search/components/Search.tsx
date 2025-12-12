"use client";
import Results from "./Results";
import SearchBar from "./SearchBar";
import styles from "./Search.module.css";
import { useVideoFrames } from "../../video/context/VideoFramesContext";

export default function Search() {
    
  const { frames } = useVideoFrames();

  return (
    <div className={styles.search}>
      <SearchBar />
      <Results frames={frames} />
    </div>
  );
}
