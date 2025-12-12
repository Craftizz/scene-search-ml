import { CaptionedFrame } from "@/types/types";

export async function captureStreamStrategy(
  video: HTMLVideoElement,
  times: number[],
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  onFrame: (frame: CaptionedFrame, blob?: Blob) => void
): Promise<void> {
  const toBlobUrl = (blob: Blob) => URL.createObjectURL(blob);

  const stream = (video as any).captureStream();
  const track = stream.getVideoTracks()[0];
  const imageCap = new (window as any).ImageCapture(track);

  await video.play().catch(() => {});
  try {
    video.playbackRate = 8;
  } catch {}

  let index = 0;
  const epsilon = 0.06;

  await new Promise<void>((resolve, reject) => {
    const pollMs = 50;
    const interval = setInterval(async () => {
      try {
        if (index >= times.length) {
          clearInterval(interval);
          resolve();
          return;
        }

        const current = video.currentTime;
        if (current + epsilon >= times[index]) {
          try {
            const bitmap: ImageBitmap = await imageCap.grabFrame();
            ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            bitmap.close();
            const blob = await new Promise<Blob | null>((res) => canvas.toBlob((b) => res(b), "image/jpeg", 0.85));
            if (blob) onFrame({ timestamp: times[index], url: toBlobUrl(blob) }, blob);
          } catch (e) {
            console.warn("grabFrame error", e);
          }
          index += 1;
        }
      } catch (e) {
        clearInterval(interval);
        reject(e);
      }
    }, pollMs);

    video.addEventListener("error", (e) => {
      clearInterval(interval);
      reject(e);
    }, { once: true });
  });

  try { track.stop(); } catch {}
  try { video.pause(); } catch {}
}
