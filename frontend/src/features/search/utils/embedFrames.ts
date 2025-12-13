import type { CaptionedFrame } from "@/types/types";

type EmbeddingResp = { vector: number[] }[];

export async function embedFrames(frames: CaptionedFrame[]): Promise<CaptionedFrame[]> {
  if (!frames || frames.length === 0) return frames;

  const form = new FormData();

  for (const f of frames) {
    // fetch the image data from the object URL (or remote URL)
    const resp = await fetch(f.url);
    if (!resp.ok) throw new Error(`Failed to fetch frame image: ${resp.status}`);
    const blob = await resp.blob();
    // backend expects `files` field (multipart)
    form.append("files", blob, "frame.png");
  }

  const apiUrl = "/api/embedding"; // proxied route in frontend
  const res = await fetch(apiUrl, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(`Embedding API error: ${res.status} ${res.statusText}`);
  }

  const data = (await res.json()) as EmbeddingResp;
  // response should be array of { vector: number[] } preserving order
  return frames.map((f, i) => ({ ...f, embedding: data[i]?.vector }));
}

export async function embedFrame(frame: CaptionedFrame): Promise<CaptionedFrame> {
  const [out] = await embedFrames([frame]);
  return out;
}
