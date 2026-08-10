import { runpodRun, setCors, sendError } from "./_runpod.js";

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();

  try {
    const body = req.body || {};
    const url = body.url || "";
    if (!url) return res.status(400).json({ error: "الرابط مطلوب" });

    const { id } = await runpodRun({
      operation: "transcribe_url",
      url,
      language: body.language || "ar",
    });

    return res.status(200).json({ status: "processing", job_id: id });
  } catch (err) {
    return sendError(res, err);
  }
}
