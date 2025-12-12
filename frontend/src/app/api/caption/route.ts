import { NextResponse } from "next/server";

export async function POST(request: Request) {
	// Proxy incoming multipart/form-data POST to the backend caption service.
	const backend = process.env.BACKEND_URL || "http://localhost:8000";
	const apiKey = process.env.BACKEND_API_KEY;

	const target = `${backend.replace(/\/+$/, "")}/v1/caption/`;

	const headers = new Headers();
	// preserve content-type (includes multipart boundary)
	const contentType = request.headers.get("content-type");
	if (contentType) headers.set("content-type", contentType);
	if (apiKey) headers.set("X-API-Key", apiKey);

	const resp = await fetch(target, {
		method: "POST",
		headers,
		// Required for forwarding a ReadableStream body in Node/Next
		// @ts-ignore -- `duplex` is supported by the runtime fetch but not always in types
		duplex: "half",
		// forward body stream directly
		body: request.body,
	});

	// forward status and headers
	const respBody = await resp.arrayBuffer();
	const outHeaders: Record<string, string> = {};
	resp.headers.forEach((v, k) => (outHeaders[k] = v));

	return new NextResponse(respBody, { status: resp.status, headers: outHeaders });
}
