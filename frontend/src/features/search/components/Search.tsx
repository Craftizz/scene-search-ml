"use client";
import Results from "./Results";
import SearchBar from "./SearchBar";
import styles from "./Search.module.css";
import { useVideoFrames } from "../../video/context/VideoFramesContext";
import { useState } from "react";
import type { CaptionedFrame } from "@/types/types";

export default function Search() {
  const { frames } = useVideoFrames();
  const [results, setResults] = useState<CaptionedFrame[] | null>(null);
  // Use relative ranking: request top 10% of frames
  const TOP_PERCENT = 0.10;
  // Optional raw-cosine threshold (range -1..1). Use this to cut on raw similarity.
  const THRESHOLD_RAW = 0.23;

  async function handleSearch(query?: string) {
    if (!query) {
      setResults(null);
      return;
    }

    // send only frames that have embeddings
    const framesWithEmb = frames.filter((f) => Array.isArray(f.embedding) && f.embedding.length > 0);

    try {
      // Ensure frames are plain JSON (no methods, no typed arrays)
      const payloadFrames = framesWithEmb.map((f) => ({
        timestamp: Number(f.timestamp),
        url: String(f.url),
        caption: f.caption ?? "",
        embedding: Array.isArray(f.embedding) ? f.embedding.map((n) => Number(n)) : undefined,
      }));

      const resp = await fetch(`/api/similar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ string: query, frames: payloadFrames, top_percent: TOP_PERCENT, threshold_raw: THRESHOLD_RAW }),
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        console.error("Search failed", resp.status, text);
        throw new Error(`Search failed: ${resp.status}`);
      }

      const data = await resp.json();
      // backend returns { results: [{ frame: {...}, similarity, probability }] }
      const mapped: CaptionedFrame[] = (data.results || []).map((r: any) => r.frame as CaptionedFrame);
      setResults(mapped);
    } catch (e) {
      console.error("Search error", e);
      setResults([]);
    }
  }

  return (
    <div className={styles.search}>
      <SearchBar onSearch={handleSearch} />
      <Results frames={results ?? frames} />
    </div>
  );
}
