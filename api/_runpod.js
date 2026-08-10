/**
 * _runpod.js — shared helper used by every /api/*.js proxy route.
 *
 * All of these proxy routes run on Vercel and talk to a RunPod Serverless
 * Endpoint (auto-scaling GPU workers running handler.py from this repo).
 * The RunPod API key lives ONLY in a Vercel environment variable — it is
 * never sent to, or reachable from, the browser.
 *
 * Files never pass through these routes as raw bytes — Vercel Functions
 * reject any request body over 4.5 MB outright, which every real video
 * blows past. Instead the browser uploads directly to Vercel Blob (see
 * api/blob-upload.js + the uploadToBlob() helper in app.js) and these
 * routes just forward the resulting file_url to RunPod as a small JSON
 * field.
 *
 * Required Vercel env vars (Project → Settings → Environment Variables):
 *   RUNPOD_API_KEY     - your RunPod API key
 *   RUNPOD_ENDPOINT_ID - the Serverless Endpoint ID created from this repo
 *   BLOB_READ_WRITE_TOKEN - auto-added when you create a Blob store
 */

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const RUNPOD_BASE = "https://api.runpod.ai/v2";

function assertConfigured() {
  if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
    const err = new Error(
      "RunPod is not configured on this deployment yet. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in Vercel → Project → Settings → Environment Variables, then redeploy."
    );
    err.isConfigError = true;
    throw err;
  }
}

/** Submits a job to RunPod. Returns the RunPod job id immediately (async /run — does not block). */
export async function runpodRun(input) {
  assertConfigured();
  const resp = await fetch(`${RUNPOD_BASE}/${RUNPOD_ENDPOINT_ID}/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({ input }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data?.error || `RunPod /run failed (${resp.status})`);
  }
  return data; // { id, status }
}

/** Polls a RunPod job's current status/output. */
export async function runpodStatus(jobId) {
  assertConfigured();
  const resp = await fetch(`${RUNPOD_BASE}/${RUNPOD_ENDPOINT_ID}/status/${jobId}`, {
    headers: { Authorization: `Bearer ${RUNPOD_API_KEY}` },
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data?.error || `RunPod /status failed (${resp.status})`);
  }
  return data; // { id, status: IN_QUEUE|IN_PROGRESS|COMPLETED|FAILED|CANCELLED, output, error }
}

export function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "*");
}

export function sendError(res, err, status = 500) {
  console.error("Proxy error:", err);
  return res.status(status).json({ error: err.message || String(err) });
}
