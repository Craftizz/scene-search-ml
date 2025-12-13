import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const backend = process.env.BACKEND_URL || "http://localhost:8000";
  const apiKey = process.env.BACKEND_API_KEY;

  const target = `${backend.replace(/\/+$/, "")}/v1/embedding/`;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  if (apiKey) headers.set("X-API-Key", apiKey);

  const resp = await fetch(target, {
    method: "POST",
    headers,
    // @ts-ignore
    duplex: "half",
    body: request.body,
  });

  const respBody = await resp.arrayBuffer();
  const outHeaders: Record<string, string> = {};
  resp.headers.forEach((v, k) => (outHeaders[k] = v));

  return new NextResponse(respBody, { status: resp.status, headers: outHeaders });
}
