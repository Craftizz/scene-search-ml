/** Strategies moved to separate modules */
import { captureStreamStrategy } from "./strategies/captureStreamStrategy";
import { seekingStrategy } from "./strategies/seekingStrategy";
import { Frame } from "@/types/types";

type ExtractOptions = {
  apiKey?: string; // optional X-API-Key for backend
  batchSize?: number;
  batchTimeout?: number;
  endpoint?: string; // caption endpoint, default /v1/caption/
  // NOTE: extractFramesStream only extracts frames. Uploading/captioning/scene
  // detection should be handled by the caller. onCaption/onScene removed.
};

function awaitLoadedMetadata(video: HTMLVideoElement): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const onMeta = () => resolve();
    const onErr = (e: any) => reject(e);
    video.addEventListener("loadedmetadata", onMeta, { once: true });
    video.addEventListener("error", onErr, { once: true });
  });
}

export async function extractFramesStream(
  src: string,
  intervalSeconds = 1,
  thumbWidth = 320,
  onFrame: (frame: Frame, blob?: Blob) => void,
  options?: ExtractOptions
): Promise<void> {
  // extraction-only: do not create uploader or request captions/embeddings here
  const video = document.createElement("video");
  video.crossOrigin = "anonymous";
  video.preload = "metadata";
  video.muted = true;
  video.playsInline = true;
  video.src = src;

  await awaitLoadedMetadata(video);

  const totalDuration = Math.max(0, video.duration || 0);
  const intervalStep = Math.max(1, intervalSeconds);
  const timeStamps: number[] = [];

  let currentTime = 0;

  while (currentTime <= Math.floor(totalDuration)) {
    timeStamps.push(currentTime);
    currentTime += intervalStep;
  }

  const aspect =
    video.videoHeight && video.videoWidth
      ? video.videoHeight / video.videoWidth
      : 9 / 16;

  const canvas = document.createElement("canvas");
  canvas.width = thumbWidth;
  canvas.height = Math.max(1, Math.round(thumbWidth * aspect));
  const ctx = canvas.getContext("2d");

  if (!ctx) throw new Error("Canvas 2D not supported");

  const wrappedOnFrame = (frame: Frame, blob: Blob) => {
    onFrame(frame, blob);
  };

  // Primary: try captureStream strategy, otherwise fallback to seeking strategy
  if (
    typeof (video as any).captureStream === "function" &&
    (window as any).ImageCapture
  ) {
    try {
      await captureStreamStrategy(
        video,
        timeStamps,
        canvas,
        ctx,
        wrappedOnFrame
      );
      return;
    } catch (e) {
      console.warn("captureStream strategy failed, using a slower fallback", e);
    }
  }

  // Fallback
  await seekingStrategy(video, timeStamps, canvas, ctx, wrappedOnFrame);
}
