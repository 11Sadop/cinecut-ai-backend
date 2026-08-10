import { runpodRun, setCors, sendError } from "./_runpod.js";

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();

  try {
    const body = req.body || {};
    if (!body.file_url) return res.status(400).json({ error: "الملف مطلوب (file_url)" });

    const { id } = await runpodRun({
      operation: "burn_subtitles",
      file_url: body.file_url,
      filename: body.filename || "video.mp4",
      text: body.text || "",
      style_mode: body.style_mode || "credits",
      font_size: body.font_size || "28",
      font_color: body.font_color || "#ffc800",
      font_name: body.font_name || "Arial",
      segments_json: body.segments_json || "",
    });

    return res.status(200).json({ status: "processing", job_id: id });
  } catch (err) {
    return sendError(res, err);
  }
}
