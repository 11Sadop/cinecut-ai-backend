// Deprecated: superseded by /api/separate-audio.js, which submits jobs to
// RunPod Serverless instead of a fragile Cloudflare quick tunnel. This file
// is kept only in case anything external still links to /api/separate, and
// simply delegates to the real implementation so it never points at a dead
// tunnel URL again.
export { default } from "./separate-audio.js";
