// Deprecated: this used to proxy to a long-dead local tunnel
// (cinecut-gpu-v42.loca.lt) and hasn't worked in a while. File results are
// now served via /api/result/{jobId}?field=... URLs returned directly by
// /api/job-status/{jobId} (see api/job-status/[id].js) — nothing in app.js
// calls this path anymore. Kept only so a stale bookmark/link fails loudly
// instead of hanging on a dead host.
export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  return res.status(410).json({
    error: 'This endpoint was retired. Results are now served via /api/result/{jobId}.',
  });
}
