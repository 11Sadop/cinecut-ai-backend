/**
 * GET /api/job-status/:id  (also served as /api/upscale-status/:id — see
 * api/upscale-status/[id].js, which just re-exports this same handler,
 * exactly mirroring the old server.py where both paths pointed at one
 * function reading the same _jobs dict).
 *
 * Translates a RunPod job status into the shape app.js already expects:
 *   { status: 'processing' | 'done' | 'error', error?,
 *     clean_media_url?, vocals_url?, guitar_url?, piano_url?, drums_url?,
 *     bass_url?, other_url?, upscale_url?, result_url?, file_url?,
 *     session_id?, transcript?, detected_language? }
 *
 * handler.py already uploads every output file straight to Vercel Blob and
 * returns real, directly-fetchable HTTPS URLs in its job output — so this
 * route just passes those fields through as-is (no re-fetching/decoding
 * needed here).
 */
import { runpodStatus, runpodCancel, setCors, sendError } from "../_runpod.js";

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();

  const jobId = req.query.id;
  if (!jobId) return res.status(400).json({ error: "Missing job id" });

  // DELETE /api/job-status/:id — cancel-operation support. Actually tells
  // RunPod to stop the job (frees the GPU worker immediately) instead of
  // just having the browser stop polling while the job keeps running.
  if (req.method === "DELETE") {
    try {
      await runpodCancel(jobId);
      return res.status(200).json({ status: "cancelled" });
    } catch (err) {
      if (err.isConfigError) return res.status(200).json({ status: "error", error: err.message });
      return sendError(res, err);
    }
  }

  try {
    const s = await runpodStatus(jobId);

    if (s.status === "IN_QUEUE" || s.status === "IN_PROGRESS") {
      return res.status(200).json({ status: "processing" });
    }
    if (s.status === "FAILED" || s.status === "CANCELLED") {
      return res.status(200).json({ status: "error", error: s.error || "فشلت المعالجة على GPU" });
    }
    if (s.status !== "COMPLETED") {
      // Unknown/transient RunPod state — treat as still processing.
      return res.status(200).json({ status: "processing" });
    }

    const out = s.output || {};
    if (out.status === "error") {
      return res.status(200).json({ status: "error", error: out.error || "فشلت المعالجة" });
    }

    // Everything handler.py returns is already a real HTTPS Blob URL (or
    // plain JSON like transcript/detected_language) — pass it through, but
    // status must end up "done" regardless of whatever inner "status" value
    // the wrapped server.py function returned (e.g. _sync_transcribe
    // returns status:"success") — spread first, then force it last so it
    // always wins.
    return res.status(200).json({ ...out, status: "done" });
  } catch (err) {
    if (err.isConfigError) return res.status(200).json({ status: "error", error: err.message });
    return sendError(res, err);
  }
}
