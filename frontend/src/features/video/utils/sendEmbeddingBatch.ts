export default async function sendEmbeddingBatch(files: File[], apiKey?: string) {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);

  const headers: Record<string, string> = {};
  if (apiKey) headers["X-API-Key"] = apiKey;

  const res = await fetch("/api/embedding", {
    method: "POST",
    headers,
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Embedding upload failed: ${res.status} ${res.statusText} ${text}`);
  }

  const json = await res.json();
  // Expecting [{ vector: number[] }, ...]
  return (json as Array<{ vector: number[] }>).map((r) => r.vector);
}
