import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const backend = process.env.BACKEND_URL || "http://localhost:8000";
  const apiKey = process.env.BACKEND_API_KEY;

  // Forward original request query string (e.g. ?threshold=0.7)
  const reqUrl = new URL(request.url);
  const search = reqUrl.search || "";
  const target = `${backend.replace(/\/+$/, "")}/v1/similar/${search}`;

  const headers = new Headers();
  headers.set("content-type", "application/json");
  if (apiKey) headers.set("X-API-Key", apiKey);

  const body = await request.text();

  const resp = await fetch(target, {
    method: "POST",
    headers,
    body,
  });

  const respBody = await resp.arrayBuffer();
  const outHeaders: Record<string, string> = {};
  resp.headers.forEach((v, k) => (outHeaders[k] = v));

  return new NextResponse(respBody, { status: resp.status, headers: outHeaders });
}
