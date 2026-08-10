import { runpodRun, setCors, sendError } from "./_runpod.js";

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();

  try {
    const body = req.body || {};
    if (!body.file_url) return res.status(400).json({ error: "الملف مطلوب (file_url)" });

    const input = {
      operation: "remove_background_video",
      file_url: body.file_url,
      filename: body.filename || "video.mp4",
      mode: body.mode || "color",
      color: body.color || "#00ff00",
      blur_amount: body.blur_amount || "25",
    };
    if (body.custom_bg_url) input.custom_bg_url = body.custom_bg_url;

    const { id } = await runpodRun(input);
    return res.status(200).json({ status: "processing", job_id: id });
  } catch (err) {
    return sendError(res, err);
  }
}
