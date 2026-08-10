// Same handler as /api/job-status/[id].js — kept as a separate path only
// because app.js's upscale poller calls this exact URL (mirrors the old
// server.py, which registered /api/upscale-status/{id} and
// /api/job-status/{id} on the very same function).
export { default } from "../job-status/[id].js";
