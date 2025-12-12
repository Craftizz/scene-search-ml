export async function sendBatch(files: File[], apiKey?: string, endpoint = "/api/caption") {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);

  const headers: Record<string, string> = {};
  if (apiKey) headers["X-API-Key"] = apiKey;

  const res = await fetch(endpoint, {
    method: "POST",
    headers,
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Upload failed: ${res.status} ${res.statusText} ${text}`);
  }

  const json = await res.json();
  // Expecting [{ caption: string }, ...]
  return (json as Array<{ caption: string }>).map((r) => r.caption);
}

export default sendBatch;
