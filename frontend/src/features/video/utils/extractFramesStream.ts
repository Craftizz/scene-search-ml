/** Strategies moved to separate modules */
import { captureStreamStrategy } from "./strategies/captureStreamStrategy";
import { seekingStrategy } from "./strategies/seekingStrategy";
import { CaptionedFrame } from "@/types/types";
import { BatchUploader } from "./uploadBatch";

type ExtractOptions = {
  apiKey?: string; // optional X-API-Key for backend
  batchSize?: number;
  batchTimeout?: number;
  endpoint?: string; // caption endpoint, default /v1/caption/
  onCaption?: (frame: CaptionedFrame) => void; // called when caption for a frame is received
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
  onFrame: (frame: CaptionedFrame, blob?: Blob) => void,
  options?: ExtractOptions
): Promise<void> {
  const uploader = new BatchUploader({
    batchSize: options?.batchSize ?? 4,
    batchTimeout: options?.batchTimeout ?? 1500,
    apiKey: options?.apiKey,
    endpoint: options?.endpoint,
  });
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

  // Internal wrapper to receive possible blob and enqueue for upload
    const wrappedOnFrame = (frame: CaptionedFrame, blob?: Blob) => {
    // Immediately notify consumer so UI can show the thumbnail
    onFrame(frame);
    if (!blob) return;

    try {
      const file = new File([blob], `frame-${frame.timestamp}.jpg`, { type: "image/jpeg" });
        uploader.add(file).then(({ caption, embedding }) => {
          options?.onCaption?.({ ...frame, caption, embedding });
      }).catch((err) => {
        console.warn("upload failed for frame", frame.timestamp, err);
      });
    } catch (e) {
      console.warn("failed to create file for upload", e);
    }
  };

  // Primary: try captureStream strategy, otherwise fallback to seeking strategy
  if (
    typeof (video as any).captureStream === "function" &&
    (window as any).ImageCapture
  ) {
    try {
      await captureStreamStrategy(video, timeStamps, canvas, ctx, wrappedOnFrame);
      return;

    } catch (e) {
      console.warn("captureStream strategy failed, using a slower fallback", e);
    }
  }

  // Fallback
  await seekingStrategy(video, timeStamps, canvas, ctx, wrappedOnFrame);
  uploader.dispose();
}
