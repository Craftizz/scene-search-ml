"use client";

import { useCallback, useEffect, useState } from "react";
import styles from "./Video.module.css";
import VideoDrop from "./VideoDrop";
import VideoDisplay from "./VideoDisplay";
import { extractFramesStream } from "../utils/extractFramesStream";
import { AnalyzeClient, AnalyzeMessage } from "../utils/analyzeWebsocketClient";
import { useVideoFrames } from "../context/VideoFramesContext";

const apiKey = process.env.NEXT_PUBLIC_API_KEY || "";

export default function Video() {

  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const { 
    addFrame, 
    addScene, 
    clearFrames, 
    updateFrame,
    updateScene,
  } = useVideoFrames();

  const tsToUrlRef =
    typeof window !== "undefined"
      ? (window as any)._tsToUrlRef ??
        ((window as any)._tsToUrlRef = { map: new Map<number, string>() })
      : { map: new Map<number, string>() };

  const handleFileSelected = useCallback((file: File) => {
    setVideoSrc(URL.createObjectURL(file));
  }, []);

  function handleEmbedded(timestamps: number[], embeddings: number[][]) {
    for (let i = 0; i < timestamps.length; i++) {
      const ts = timestamps[i];
      const emb = embeddings[i];
      const url = tsToUrlRef.map.get(ts) || "";
      if (typeof ts === "number" && Array.isArray(emb) && url) {
        updateFrame({ timestamp: ts, url, embedding: emb });
      }
    }
  }

  function handleSceneMessage(msg: AnalyzeMessage) {

    const startTimestamp =
      typeof (msg as any).timestamp === "number"
        ? (msg as any).timestamp
        : undefined;

    const endTimestamp =
      typeof (msg as any).end_timestamp === "number"
        ? (msg as any).end_timestamp
        : undefined;

    let url: string | undefined = (msg as any).image;
    if (!url && typeof startTimestamp === "number") url = tsToUrlRef.map.get(startTimestamp);
    if (!url && typeof endTimestamp === "number") url = tsToUrlRef.map.get(endTimestamp);

    addScene({
      id: (msg as any).id ?? Date.now(),
      timestamp: startTimestamp ?? 0,
      url: url ?? "",
      caption: (msg as any).caption,
      request_id: (msg as any).request_id,
      end_timestamp: (msg as any).end_timestamp,
    });
    // map request_id -> scene id for later caption updates
    try {
      const rid = (msg as any).request_id;
      const sid = (msg as any).id ?? Date.now();
      if (rid && updateScene) updateScene(rid, { request_id: rid });
    } catch {}
  }

  useEffect(() => {
    if (!videoSrc) {
      clearFrames();
      return;
    }

    let cancelled = false;
    const client = new AnalyzeClient({ apiKey });

    async function run() {
      try {
        clearFrames();
        await client.connect();

        client.onMessage((msg: AnalyzeMessage) => {
          if (msg.type === "scene") {
            handleSceneMessage(msg);
            return;
          }
          if (msg.type === "caption_started") {
            // optionally mark scene as loading
            if (updateScene) updateScene((msg as any).request_id, { caption: "..." });
            return;
          }
          if (msg.type === "caption_chunk") {
            const rid = (msg as any).request_id;
            const chunk = (msg as any).chunk;
            if (rid && updateScene) {
              // append chunk to existing caption
              updateScene(rid, (s) => ({ caption: (s.caption ?? "") + String(chunk) } as any));
            }
            return;
          }
          if (msg.type === "caption_result") {
            const rid = (msg as any).request_id;
            const caption = (msg as any).caption;
            if (rid && updateScene) updateScene(rid, { caption });
            return;
          }
          if (
            msg.type === "embedded_batch" &&
            Array.isArray(msg.timestamps) &&
            Array.isArray(msg.embeddings)
          ) {
            handleEmbedded(msg.timestamps, msg.embeddings);
            return;
          }
          if (
            msg.type === "embedded_chunk" &&
            Array.isArray(msg.timestamps) &&
            Array.isArray(msg.embeddings)
          ) {
            handleEmbedded(msg.timestamps, msg.embeddings);
            return;
          }
        });

        await extractFramesStream(
          videoSrc as string,
          1,
          320,
          async (frame, blob) => {
            if (cancelled) {
              try {
                URL.revokeObjectURL(frame.url);
              } catch {}
              return;
            }
            addFrame(frame);
            tsToUrlRef.map.set(frame.timestamp, frame.url);
            if (!blob) return;
            try {
              await client.sendFrame(frame.timestamp, blob);
            } catch (e) {
              console.warn("failed to send frame to analyze ws", e);
            }
          }
        );
      } catch (e) {
        console.error("Video extraction failed", e);
      } finally {
        client.close();
      }
    }

    run();

    return () => {
      cancelled = true;
      client.close();
    };
  }, [videoSrc, addFrame, addScene, clearFrames, updateFrame, updateScene, apiKey]);

  return (
    <div className={styles.video}>
      {!videoSrc && <VideoDrop onFileSelected={handleFileSelected} />}
      {videoSrc && <VideoDisplay src={videoSrc} />}
    </div>
  );
}
