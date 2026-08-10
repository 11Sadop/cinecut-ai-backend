import { runpodRun, setCors, sendError } from "./_runpod.js";

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();

  try {
    const body = req.body || {};
    if (!body.file_url) return res.status(400).json({ error: "الملف مطلوب (file_url)" });

    const { id } = await runpodRun({
      operation: "transcribe",
      file_url: body.file_url,
      filename: body.filename || "audio.wav",
      language: body.language || "ar",
    });

    return res.status(200).json({ status: "processing", job_id: id });
  } catch (err) {
    return sendError(res, err);
  }
}
