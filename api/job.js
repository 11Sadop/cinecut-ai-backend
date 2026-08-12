/**
 * POST /api/job
 *
 * Single consolidated endpoint for every RunPod operation (Vercel's Hobby
 * plan caps a deployment at 12 serverless functions — this project's
 * previous one-file-per-operation layout blew past that immediately, which
 * is why the last deploy failed with "No more than 12 Serverless Functions
 * can be added..."). Instead of one Vercel function per operation, the
 * frontend now POSTs { operation: "...", ...params } here and this single
 * function forwards it to RunPod.
 *
 * Body: { operation, file_url?, filename?, url?, ...operation params }
 * See handler.py's _OPERATIONS dict for the full list of valid operations
 * and their expected params — this route does no validation of its own,
 * it just forwards the body as RunPod job input.
 */
import { runpodRun, setCors, sendError } from "./_runpod.js";

const VALID_OPERATIONS = new Set([
  "separate_audio",
  "stem_from_url",
  "transcribe",
  "transcribe_url",
  "upscale",
  "upscale_url",
  "download_url",
  "remove_background_image",
  "remove_background_video",
  "burn_subtitles",
  "tts",
]);

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();

  try {
    const body = req.body || {};
    const { operation, ...params } = body;

    if (!VALID_OPERATIONS.has(operation)) {
      return res.status(400).json({
        error: `Unknown or missing 'operation'. Expected one of: ${[...VALID_OPERATIONS].join(", ")}`,
      });
    }

    const { id } = await runpodRun({ operation, ...params });
    return res.status(200).json({ status: "processing", job_id: id });
  } catch (err) {
    return sendError(res, err);
  }
}
