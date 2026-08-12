/**
 * GET /api/ops-log
 *
 * Owner-only aggregation endpoint for the separate ops/cost dashboard.
 *
 * Every job handler.py runs writes one small JSON record to Vercel Blob
 * under logs/*.json (see handler.py's _log_operation) -- that happens
 * server-side, on the RunPod worker, for EVERY job regardless of which
 * browser/device triggered it, which is what makes this data genuinely
 * global (unlike the old localStorage-based in-app stats page, which only
 * ever saw whatever happened in one specific browser). This route reads
 * all of those records back, aggregates them, and estimates the RunPod GPU
 * cost of each in Saudi Riyal.
 *
 * Access note: the only intended caller is the separate,
 * Vercel-password-protected dashboard project -- that deployment password
 * already keeps random visitors off the page entirely. A small shared
 * value below is a second layer so this JSON endpoint isn't a bare,
 * guessable URL sitting on the public main site.
 */
import { list } from "@vercel/blob";

const OPS_KEY = "cc_ops_7f3a9d2e1b6c4f58a0d3e9b7c2f14a6d8e5b3c9f1a7d4e2b8c6f0a3d9e5b1c7f4a";

const GPU_RATE_USD_PER_HOUR = 0.69; // RunPod "24 GB" primary tier
const USD_TO_SAR = 3.75; // fixed peg since 1986

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, x-cinecut-ops-key");
}

function costSar(durationSec) {
  const hours = (durationSec || 0) / 3600;
  return hours * GPU_RATE_USD_PER_HOUR * USD_TO_SAR;
}

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();

  if (req.headers["x-cinecut-ops-key"] !== OPS_KEY) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  try {
    const records = [];
    let cursor;
    let hasMore = true;
    while (hasMore) {
      const result = await list({ prefix: "logs/", cursor, limit: 1000 });
      const chunk = await Promise.all(
        result.blobs.map(async (b) => {
          try {
            const r = await fetch(b.url);
            if (!r.ok) return null;
            return await r.json();
          } catch {
            return null;
          }
        })
      );
      records.push(...chunk.filter(Boolean));
      cursor = result.cursor;
      hasMore = result.hasMore;
    }

    records.sort((a, b) => (b.started_at || 0) - (a.started_at || 0));

    const done = records.filter((r) => r.status === "done");
    const errored = records.filter((r) => r.status === "error");
    const totalOps = records.length;
    const totalClipMinutes = done.reduce((s, r) => s + (r.clip_duration_sec || 0), 0) / 60;
    const totalCostSar = done.reduce((s, r) => s + costSar(r.duration_sec), 0);
    const successRate = totalOps > 0 ? Math.round((done.length / totalOps) * 100) : 0;
    const avgProcSec = done.length
      ? done.reduce((s, r) => s + (r.duration_sec || 0), 0) / done.length
      : 0;

    const perOperation = {};
    for (const r of records) {
      const op = r.operation || "unknown";
      if (!perOperation[op]) {
        perOperation[op] = { count: 0, done: 0, error: 0, costSar: 0 };
      }
      perOperation[op].count++;
      if (r.status === "done") {
        perOperation[op].done++;
        perOperation[op].costSar += costSar(r.duration_sec);
      }
      if (r.status === "error") perOperation[op].error++;
    }

      return res.status(200).json({
              generatedAt: Date.now(),
              summary: {
                        totalOps,
                        doneCount: done.length,
                        errorCount: errored.length,
                        successRate,
                        totalClipMinutes,
                        avgProcSec,
                        totalCostSar,
              },
              perOperation,
              history: records.slice(0, 500).map((r) => ({
                        operation: r.operation,
                        status: r.status,
                        started_at: r.started_at,
                        ended_at: r.ended_at,
                        duration_sec: r.duration_sec,
                        clip_duration_sec: r.clip_duration_sec,
                        cost_sar: r.status === "done" ? costSar(r.duration_sec) : null,
                        error: r.error || null,
              })),
      });
  } catch (err) {
        return res.status(500).json({ error: err.message || String(err) });
  }
}
