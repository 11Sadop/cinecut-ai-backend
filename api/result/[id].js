// Deprecated: results are now real Vercel Blob URLs returned directly in
// the job-status response (handler.py uploads outputs straight to Blob),
// so there's no more need to re-fetch/decode a base64 field through a
// second Vercel Function call. Kept only so a stale reference fails loudly.
export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  return res.status(410).json({
    error: 'This endpoint was retired. Job results now include direct Blob URLs.',
  });
}
