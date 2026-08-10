import { runpodRun, setCors, sendError } from "./_runpod.js";

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();

  try {
    const body = req.body || {};
    if (!body.file_url) return res.status(400).json({ error: "الملف غير موجود (file_url)" });

    const { id } = await runpodRun({
      operation: "upscale",
      file_url: body.file_url,
      filename: body.filename || "video.mp4",
      resolution: body.resolution || "4k",
      fps: body.fps || "120",
      color_mode: body.color_mode || "face",
      speed: body.speed || "ai",
    });

    return res.status(200).json({ status: "processing", job_id: id });
  } catch (err) {
    return sendError(res, err);
  }
}
