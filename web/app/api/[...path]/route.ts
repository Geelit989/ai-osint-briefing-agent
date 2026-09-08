import http from "node:http";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };

async function proxy(request: Request, context: Context): Promise<Response> {
  const { path } = await context.params;
  const route = path.join("/");
  const supported = /^(health|corpus\/status|runs(?:\/[^/]+(?:\/sources(?:\/[^/]+)?)?)?|documents(?:\/[^/]+)?)$/.test(route);
  if (!supported || (request.method === "POST" && route !== "runs")) {
    return Response.json({ detail: "Unknown ARGUS endpoint." }, { status: 404 });
  }
  const uiPort = process.env.ARGUS_UI_PORT || "3000";
  const localHosts = new Set([`127.0.0.1:${uiPort}`, `localhost:${uiPort}`]);
  if (!localHosts.has(request.headers.get("host") || "")) {
    return Response.json({ detail: "The workspace is available only through its local address." }, { status: 403 });
  }
  const origin = request.headers.get("origin");
  if (request.method === "POST" && origin && ![...localHosts].some((host) => origin === `http://${host}`)) {
    return Response.json({ detail: "Requests must originate from this local workspace." }, { status: 403 });
  }
  const port = Number(process.env.ARGUS_API_PORT || "8000");
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    return Response.json({ detail: "The local API port is invalid." }, { status: 503 });
  }
  const body = request.method === "POST" ? Buffer.from(await request.arrayBuffer()) : undefined;
  if (body && body.byteLength > 32_768) {
    return Response.json({ detail: "The request is too large." }, { status: 413 });
  }

  // Node's HTTP client has no response timeout by default. Local model inference
  // can take many minutes; do not apply a short proxy or fetch headers timeout.
  return new Promise<Response>((resolve) => {
    const upstream = http.request({
      hostname: "127.0.0.1",
      port,
      path: `/api/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`,
      method: request.method,
      signal: request.signal,
      headers: {
        accept: "application/json",
        ...(body ? { "content-type": "application/json", "content-length": body.byteLength } : {}),
      },
    }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => chunks.push(chunk));
      response.on("end", () => resolve(new Response(Buffer.concat(chunks), {
        status: response.statusCode || 502,
        headers: {
          "content-type": response.headers["content-type"] || "application/json",
          "cache-control": "no-store",
        },
      })));
      response.on("error", () => resolve(Response.json({ detail: "The local ARGUS API connection was interrupted." }, { status: 502 })));
    });
    upstream.on("error", () => resolve(Response.json({
      detail: "The local ARGUS API is unavailable. Start the API service, then retry.",
    }, { status: 503 })));
    upstream.end(body);
  });
}

export const GET = proxy;
export const POST = proxy;
