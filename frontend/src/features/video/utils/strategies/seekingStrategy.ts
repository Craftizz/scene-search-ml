import { Frame } from "@/types/types";

export async function seekingStrategy(
  video: HTMLVideoElement,
  times: number[],
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  onFrame: (frame: Frame, blob: Blob) => void
): Promise<void> {
  const toBlobUrl = (blob: Blob) => URL.createObjectURL(blob);

  for (const t of times) {
    // eslint-disable-next-line no-await-in-loop
    await new Promise<void>((res, rej) => {
      const onseeked = async () => {
        try {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const blob = await new Promise<Blob | null>((r) => canvas.toBlob((b) => r(b), "image/jpeg", 0.85));
          if (blob) onFrame({ timestamp: t, url: toBlobUrl(blob) }, blob);
          res();
        } catch (err) {
          rej(err);
        }
      };
      const onerr = (e: any) => rej(e);
      video.currentTime = Math.min(t, video.duration);
      video.addEventListener("seeked", onseeked, { once: true });
      video.addEventListener("error", onerr, { once: true });
    });
  }
}
