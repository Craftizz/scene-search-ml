"use client";
import Results from "./Results";
import SearchBar from "./SearchBar";
import styles from "./Search.module.css";
import { useVideoFrames } from "../../video/context/VideoFramesContext";
import { useState } from "react";
import type { Frame, Scene } from "@/types/types";

export default function Search() {

  const { frames, scenes } = useVideoFrames();

  const [query, setQuery] = useState<string>("");
  const [results, setResults] = useState<Frame[] | Scene[] | null>(null);

  async function handleSearch(query?: string) {
    if (!query) {
      setResults(null);
      return;
    }

    // Only frames with embeddings can be searched
    const embeddedFrames = frames.filter(
      (f) => Array.isArray(f.embedding) && f.embedding.length > 0
    );

    try {

      // Ensure frames are plain JSON (no methods, no typed arrays)
      const payloadFrames = embeddedFrames.map((embeddedFrame) => ({
        timestamp: Number(embeddedFrame.timestamp),
        url: String(embeddedFrame.url),
        embedding: Array.isArray(embeddedFrame.embedding)
          ? embeddedFrame.embedding.map((n) => Number(n))
          : undefined,
      }));

      const resp = await fetch(`/api/similar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ string: query, frames: payloadFrames }),
      });

      if (!resp.ok) {

          throw new Error(`Search failed: ${resp.status}`);
      }

      const data = await resp.json();
      const mapped: Frame[] = (data.results || []).map(
        (r: any) => r.frame as Frame
      );

      setResults(mapped);
      setQuery(query);

    } catch (e) {

      setResults([]);
    }
  }

  const resultDisplay: Frame[] | Scene[] = results ?? scenes ?? [];

  return (
    <div className={styles.search}>
      <SearchBar onSearch={handleSearch} />

      {query.length > 0 && results?.length === 0 ? (
        <p className={styles.noResults}>We found nothing for "{query}"</p>
      ) : (
        <Results
          items={resultDisplay}
          type={
            typeof query === "string" && query.trim().length > 0
              ? "frames"
              : "scenes"
          }
        />
      )}
    </div>
  );
}
