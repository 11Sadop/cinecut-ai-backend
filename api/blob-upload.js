/**
 * POST /api/blob-upload
 *
 * Server side of the Vercel Blob "client upload" handshake (see
 * https://vercel.com/docs/vercel-blob/client-upload). The browser's
 * `upload()` call (in app.js's uploadToBlob helper) hits this route first
 * to get a short-lived, scoped token, then PUTs the file bytes straight to
 * Vercel Blob — the file never passes through this (or any) Vercel
 * Function, so the platform's 4.5 MB function body limit never applies.
 *
 * There's no login system in this app, so this route can't check a user
 * session the way Vercel's docs example does. As light abuse-deterrence,
 * requests must include the header `x-cinecut-key` matching CINECUT_UPLOAD_KEY
 * (set that same value in app.js's UPLOAD_KEY constant and in this Vercel
 * project's env vars — optional but recommended if this deployment is
 * public).
 */
import { handleUpload } from "@vercel/blob/client";
import { setCors, sendError } from "./_runpod.js";

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();

  const requiredKey = process.env.CINECUT_UPLOAD_KEY;
  if (requiredKey && req.headers["x-cinecut-key"] !== requiredKey) {
    return res.status(401).json({ error: "Unauthorized upload request" });
  }

  try {
    const body = req.body;
    const jsonResponse = await handleUpload({
      body,
      request: req,
      onBeforeGenerateToken: async () => ({
        allowedContentTypes: ["video/*", "audio/*", "image/*"],
        addRandomSuffix: true,
        maximumSizeInBytes: 5 * 1024 * 1024 * 1024, // 5GB
      }),
      onUploadCompleted: async () => {},
    });
    return res.status(200).json(jsonResponse);
  } catch (err) {
    return sendError(res, err, 400);
  }
}
