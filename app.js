/**
 * CineCut AI Pro Ultimate Suite V124
 * Full GPU Tunnel Engine with Mobile Background Safari/Chrome Support
 */

// GPU compute now runs on RunPod Serverless (auto-scaling workers, scale-to-
// -zero when idle) behind this same Vercel deployment's own /api/* routes —
// no more fragile local Cloudflare tunnel URL to keep updating. Every fetch
// below goes through resolveMediaUrl() so both relative Vercel paths and
// absolute/data URLs work unchanged.
const GPU_TUNNEL = "";
const TUNNEL_HEADERS = {};

/** Resolves a path returned by our /api/* routes into a fetchable URL.
 * Job results are now real Vercel Blob URLs (already absolute), so this
 * mostly just passes those through; it also still handles the few
 * same-origin relative /api/* paths (e.g. job-status) unchanged. */
function resolveMediaUrl(path) {
  if (!path) return path;
  if (/^(https?:|blob:|data:)/i.test(path)) return path;
  return `${GPU_TUNNEL}${path}`;
}

let _blobUploadFn = null;
/** Uploads a File/Blob straight to Vercel Blob from the browser (bypasses
 * Vercel's 4.5MB serverless function body limit entirely — the bytes never
 * pass through any of our /api/* routes). Returns the resulting public URL,
 * which is then sent to the job-submission routes as a small JSON string
 * instead of the raw file.
 *
 * Retries the WHOLE upload up to 3 times with a short backoff before giving
 * up. The @vercel/blob client already retries individual chunk requests
 * internally, but on a flaky connection (common with large 4K source
 * videos for the upscale tool) it can still exhaust its own retries and
 * throw "Request failed after retries. Please try again." — previously
 * that bubbled straight up as a hard failure on the very first hiccup, with
 * no second chance. This wraps the same call in an outer retry loop so a
 * single dropped connection doesn't kill the whole job. */
async function uploadToBlob(file, filename, _onRetry) {
  if (!_blobUploadFn) {
    const mod = await import('https://esm.sh/@vercel/blob@0.27.1/client');
    _blobUploadFn = mod.upload;
  }

  const maxAttempts = 3;
  let lastErr = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const blob = await _blobUploadFn(filename || file.name || 'upload.bin', file, {
        access: 'public',
        handleUploadUrl: '/api/blob-upload',
      });
      return blob.url;
    } catch (e) {
      lastErr = e;
      console.warn(`uploadToBlob attempt ${attempt}/${maxAttempts} failed:`, e);
      if (attempt < maxAttempts) {
        if (typeof _onRetry === 'function') {
          try { _onRetry(attempt, maxAttempts); } catch (_) {}
        }
        await new Promise(r => setTimeout(r, attempt * 1500));
      }
    }
  }
  const friendly = new Error(
    `تعذر رفع الملف إلى السيرفر بعد ${maxAttempts} محاولات (${lastErr?.message || 'مشكلة اتصال'}). ` +
    `يرجى التأكد من ثبات الإنترنت والمحاولة مرة أخرى، أو تجربة ملف أصغر.`
  );
  friendly.cause = lastErr;
  throw friendly;
}

/** Polls /api/job-status/{jobId} until RunPod finishes the job (or errors/
 * times out), resolving with the final status payload. Used by any /api/*
 * route that returns { job_id } for async GPU processing. */
function pollJobStatus(jobId, maxWaitMs = 5 * 60 * 1000, pollIntervalMs = 1500, opName = '') {
  localStorage.setItem('cinecut_active_job_id', jobId);
  // Records WHICH operation this job is, so that if the tab gets backgrounded
  // (very likely on mobile during a multi-minute job) and later resumed by
  // checkAndResumePendingMobileJob(), that function knows which result shape
  // to expect (upscale_url vs result_url vs transcript, etc.) instead of
  // only ever handling the audio-separation shape and silently dropping
  // every other completed job when the user comes back to the tab.
  if (opName) localStorage.setItem('cinecut_active_job_op', opName);
  state.activeJobId = jobId;
  let elapsed = 0;
  return new Promise((resolve, reject) => {
    const poller = setInterval(async () => {
      elapsed += pollIntervalMs;
      if (elapsed > maxWaitMs) {
        clearInterval(poller);
        reject(new Error('انتهت مهلة المعالجة'));
        return;
      }
      try {
        const statusRes = await fetch(`${GPU_TUNNEL}/api/job-status/${jobId}`, { headers: TUNNEL_HEADERS });
        if (!statusRes.ok) return;
        const statusData = await statusRes.json();
        if (statusData.status === 'done') {
          clearInterval(poller);
          localStorage.removeItem('cinecut_active_job_id');
          resolve(statusData);
        } else if (statusData.status === 'error') {
          clearInterval(poller);
          localStorage.removeItem('cinecut_active_job_id');
          reject(new Error(statusData.error || 'حدث خطأ أثناء المعالجة'));
        }
      } catch (e) {}
    }, pollIntervalMs);
    state.activeJobPoller = poller;
  });
}

// ─── OPERATIONS STATS / LIVE DASHBOARD ────────────────────────────────
// Persists a rolling log of every GPU operation (separation, upscale, STT,
// TTS, background removal) to localStorage so the stats page can show what
// is running right now (with elapsed time) plus a detailed history: clip
// duration processed, real processing time, success/failure, and aggregate
// stats (total clip-minutes, success rate, average processing time).
// Client-side only (localStorage) since this is a single-user app.
const OPS_LOG_KEY = "cinecut_ops_log";
const OPS_LOG_MAX = 300;

const OP_LABELS = {
  separate_audio: 'فصل الموسيقى عن الصوت',
  stem_from_url:  'فصل الموسيقى (من رابط)',
  upscale:        'رفع الجودة 4K',
  upscale_url:    'رفع الجودة 4K (من رابط)',
  transcribe:     'استخراج النص',
  transcribe_url: 'استخراج النص (من رابط)',
  tts:            'تحويل النص إلى صوت',
  remove_background_image: 'إزالة خلفية (صورة)',
  remove_background_video: 'إزالة خلفية (فيديو)',
};

function _loadOpsLog() {
  try {
    const raw = localStorage.getItem(OPS_LOG_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch (e) { return []; }
}

function _saveOpsLog(log) {
  try {
    if (log.length > OPS_LOG_MAX) log = log.slice(log.length - OPS_LOG_MAX);
    localStorage.setItem(OPS_LOG_KEY, JSON.stringify(log));
  } catch (e) {}
}

function getMediaDurationSeconds(fileOrUrl) {
  return new Promise((resolve) => {
    try {
      if (!fileOrUrl) { resolve(null); return; }
      const el = document.createElement('video');
      el.preload = 'metadata';
      el.muted = true;
      let objectUrlToRevoke = null;
      let settled = false;
      const done = (val) => {
        if (settled) return;
        settled = true;
        if (objectUrlToRevoke) { try { URL.revokeObjectURL(objectUrlToRevoke); } catch (e) {} }
        resolve(val);
      };
      el.onloadedmetadata = () => done(isFinite(el.duration) ? el.duration : null);
      el.onerror = () => done(null);
      setTimeout(() => done(null), 8000);

      if (fileOrUrl instanceof Blob) {
        const u = URL.createObjectURL(fileOrUrl);
        objectUrlToRevoke = u;
        el.src = u;
      } else if (typeof fileOrUrl === 'string') {
        el.crossOrigin = 'anonymous';
        el.src = fileOrUrl;
      } else {
        done(null);
      }
    } catch (e) { resolve(null); }
  });
}

function opLogStart(operation, clipDurationSec, extra) {
  const log = _loadOpsLog();
  const id = `op_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  log.push(Object.assign({
    id,
    operation,
    label: OP_LABELS[operation] || operation,
    startedAt: Date.now(),
    endedAt: null,
    status: 'running',
    clipDurationSec: (typeof clipDurationSec === 'number' && isFinite(clipDurationSec)) ? clipDurationSec : null,
    error: null,
  }, extra || {}));
  _saveOpsLog(log);
  _refreshStatsIfOpen();
  return id;
}

function opLogEnd(id, status, extra) {
  if (!id) return;
  const log = _loadOpsLog();
  const idx = log.findIndex(e => e.id === id);
  if (idx === -1) return;
  log[idx] = Object.assign(log[idx], { status: status || 'done', endedAt: Date.now() }, extra || {});
  _saveOpsLog(log);
  _refreshStatsIfOpen();
}

let statsRefreshInterval = null;
function _refreshStatsIfOpen() {
  const overlay = document.getElementById('stats-overlay');
  if (overlay && overlay.style.display !== 'none') renderStatsPage();
}

function fmtOpDuration(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return '—';
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function fmtOpDate(ts) {
  try {
    return new Date(ts).toLocaleString('ar-SA', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch (e) { return new Date(ts).toLocaleString(); }
}

function renderStatsPage() {
  const log = _loadOpsLog().slice().reverse();
  const running = log.filter(e => e.status === 'running');
  const finished = log.filter(e => e.status !== 'running');

  const runWrap = document.getElementById('stats-running-list');
  if (runWrap) {
    if (running.length === 0) {
      runWrap.innerHTML = '<div class="stats-empty-msg">لا توجد عمليات جارية حالياً</div>';
    } else {
      runWrap.innerHTML = running.map(e => {
        const elapsedSec = (Date.now() - e.startedAt) / 1000;
        const durLabel = e.clipDurationSec != null ? `مدة المقطع: ${fmtOpDuration(e.clipDurationSec)} · ` : '';
        return `
          <div class="stats-running-item">
            <span class="stats-pulse-dot"></span>
            <div class="stats-running-info">
              <div class="stats-running-label">${e.label}</div>
              <div class="stats-running-sub">${durLabel}جارية منذ ${fmtOpDuration(elapsedSec)}</div>
            </div>
          </div>`;
      }).join('');
    }
  }

  const doneEntries = finished.filter(e => e.status === 'done');
  const errorEntries = finished.filter(e => e.status === 'error');
  const totalOps = finished.length;
  const totalClipMinutes = doneEntries.reduce((sum, e) => sum + (e.clipDurationSec || 0), 0) / 60;
  const successRate = totalOps > 0 ? Math.round((doneEntries.length / totalOps) * 100) : 0;
  const avgProcSec = doneEntries.length
    ? doneEntries.reduce((sum, e) => sum + (((e.endedAt || e.startedAt) - e.startedAt) / 1000), 0) / doneEntries.length
    : 0;

  const cardsWrap = document.getElementById('stats-summary-cards');
  if (cardsWrap) {
    cardsWrap.innerHTML = `
      <div class="stats-card"><div class="stats-card-icon text-cyan"><i class="fa-solid fa-layer-group"></i></div><div class="stats-card-value">${totalOps}</div><div class="stats-card-label">إجمالي العمليات</div></div>
      <div class="stats-card"><div class="stats-card-icon text-purple"><i class="fa-solid fa-clock"></i></div><div class="stats-card-value">${totalClipMinutes.toFixed(1)}</div><div class="stats-card-label">دقائق مقاطع تمت معالجتها</div></div>
      <div class="stats-card"><div class="stats-card-icon text-green"><i class="fa-solid fa-circle-check"></i></div><div class="stats-card-value">${successRate}%</div><div class="stats-card-label">نسبة نجاح العمليات</div></div>
      <div class="stats-card"><div class="stats-card-icon text-gold"><i class="fa-solid fa-stopwatch"></i></div><div class="stats-card-value">${fmtOpDuration(avgProcSec)}</div><div class="stats-card-label">متوسط وقت المعالجة</div></div>
      <div class="stats-card"><div class="stats-card-icon text-pink"><i class="fa-solid fa-triangle-exclamation"></i></div><div class="stats-card-value">${errorEntries.length}</div><div class="stats-card-label">عمليات فاشلة</div></div>
    `;
  }

  const tbody = document.getElementById('stats-history-tbody');
  if (tbody) {
    if (finished.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="stats-empty-msg">لا يوجد سجل عمليات بعد</td></tr>`;
    } else {
      tbody.innerHTML = finished.slice(0, 100).map(e => {
        const procSec = e.endedAt ? (e.endedAt - e.startedAt) / 1000 : null;
        const statusBadge = e.status === 'done'
          ? '<span class="stats-badge stats-badge-ok">✅ تمت</span>'
          : `<span class="stats-badge stats-badge-fail" title="${(e.error || '').toString().replace(/"/g, '')}">❌ فشلت</span>`;
        return `
          <tr>
            <td>${e.label}</td>
            <td>${fmtOpDuration(e.clipDurationSec)}</td>
            <td>${fmtOpDuration(procSec)}</td>
            <td>${statusBadge}</td>
            <td>${fmtOpDate(e.startedAt)}</td>
          </tr>`;
      }).join('');
    }
  }
}

window.openStatsPage = function () {
  const overlay = document.getElementById('stats-overlay');
  if (!overlay) return;
  overlay.style.display = 'flex';
  overlay.classList.add('active');
  renderStatsPage();
  if (statsRefreshInterval) clearInterval(statsRefreshInterval);
  statsRefreshInterval = setInterval(renderStatsPage, 1000);
};

window.closeStatsPage = function () {
  const overlay = document.getElementById('stats-overlay');
  if (overlay) {
    overlay.style.display = 'none';
    overlay.classList.remove('active');
  }
  if (statsRefreshInterval) { clearInterval(statsRefreshInterval); statsRefreshInterval = null; }
};

window.clearOpsLog = function () {
  if (!confirm('هل تريد مسح كامل سجل العمليات؟')) return;
  localStorage.removeItem(OPS_LOG_KEY);
  renderStatsPage();
};

const state = {
  currentTool: 'stem',
  selectedFile: null,
  previewUrl: null,
  originalInputUrl: null,
  processedMediaUrl: null,
  processedVocalsUrl: null,
  processedMusicUrl: null,
  processedCleanVideoUrl: null,
  cleanMediaDirectUrl: null,
  isProcessing: false,
  progressInterval: null,
  timerInterval: null,
  activeJobPoller: null,
  activeJobId: null,
  startTimeMs: 0,
  lastSessionId: null,
  transcriptSegments: [],
  bgRemoveMode: 'color',
  bgRemoveModeExplicit: false,
  bgRemoveColor: '#00ff00',
  bgRemoveCustomBgFile: null,
  bgRemoveResultUrl: null,
  bgRemoveResultBlobUrl: null,
  bgRemoveResultKind: null // 'image' | 'video'
};

// ─── UTILS & AUDIO STOP ─────────────────────────────────────────────────────
function stopAllActiveAudio() {
  document.querySelectorAll('audio, video').forEach(el => {
    try { el.pause(); } catch(e){}
  });
  if (window.speechSynthesis) {
    try { window.speechSynthesis.cancel(); } catch(e){}
  }
}

function triggerFileInput() {
  const el = document.getElementById('media-file-input');
  if (el) el.click();
}
window.triggerFileInput = triggerFileInput;

// ─── FILE INPUT CHANGE HANDLER ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('media-file-input');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;

      state.selectedFile = file;
      state.previewUrl = URL.createObjectURL(file);
      state.originalInputUrl = null;

      const filePill = document.getElementById('selected-file-name');
      if (filePill) {
        filePill.innerText = `📄 الملف المختار: ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
        filePill.style.display = 'inline-block';
      }

      const isImage = file.type.includes('image');
      const isAudio = file.type.includes('audio');
      renderLiveMediaPreview(state.previewUrl, isImage ? 'image' : (isAudio ? 'audio' : 'video'));

      // ROOT CAUSE of "إزالة الخلفية تطلع خضراء دائمًا، أبيها شفافة": the
      // tool defaulted EVERY output (images included) to solid-color mode
      // because alpha-channel WebM genuinely doesn't render in a plain
      // <video> tag — but that limitation only applies to VIDEO. A
      // transparent PNG (image mode) displays correctly everywhere (an
      // <img> tag, Photoshop, PowerPoint, any editor) exactly like every
      // other background-removal tool (remove.bg, Canva, etc.) — there was
      // never a real reason to force color mode for images too. Now: once
      // we know the selected file is an image, auto-switch to transparent
      // UNLESS the user already explicitly picked a different mode
      // themselves (that choice always wins).
      if (state.currentTool === 'bgremove' && isImage && !state.bgRemoveModeExplicit && typeof window.setBgRemoveMode === 'function') {
        window.setBgRemoveMode('transparent', true);
      }

      // The audio/video combo checkboxes (stem/upscale/denoise/stt) are
      // irrelevant for the background-removal tool — keep them hidden there.
      showMultiOpsCheckboxes(state.currentTool !== 'bgremove');
    });
  }

  // Mobile background job resume check on Safari / Chrome switch
  checkAndResumePendingMobileJob();
});

// ─── RENDER MEDIA PREVIEW ──────────────────────────────────────────────────
function renderLiveMediaPreview(url, type = 'video') {
  const wrap = document.getElementById('modal-preview-player-wrap');
  const vPlayer = document.getElementById('tool-video-preview');
  const aPlayer = document.getElementById('tool-audio-preview');
  const iPlayer = document.getElementById('tool-image-preview');

  if (wrap) wrap.style.display = 'block';

  if (type === 'image') {
    if (vPlayer) vPlayer.style.display = 'none';
    if (aPlayer) aPlayer.style.display = 'none';
    if (iPlayer) {
      iPlayer.src = url;
      iPlayer.style.display = 'block';
    }
  } else if (type === 'audio') {
    if (vPlayer) vPlayer.style.display = 'none';
    if (iPlayer) iPlayer.style.display = 'none';
    if (aPlayer) {
      aPlayer.src = url;
      aPlayer.style.display = 'block';
    }
  } else {
    if (aPlayer) aPlayer.style.display = 'none';
    if (iPlayer) iPlayer.style.display = 'none';
    if (vPlayer) {
      vPlayer.src = url;
      vPlayer.style.display = 'block';
    }
  }
}

function showMultiOpsCheckboxes(show = true) {
  const wrap = document.getElementById('multi-operations-checkbox-wrap');
  if (wrap) wrap.style.display = show ? 'block' : 'none';
}

// ─── MODAL CONTROLLER ───────────────────────────────────────────────────────
window.openToolModal = function(toolName) {
  stopAllActiveAudio();
  state.isProcessing = false;

  state.currentTool = toolName;
  state.selectedFile = null;
  state.previewUrl = null;
  state.originalInputUrl = null;
  state.processedMediaUrl = null;
  state.processedVocalsUrl = null;
  state.processedMusicUrl = null;
  state.processedCleanVideoUrl = null;
  state.cleanMediaDirectUrl = null;

  // Clear previous dynamic result cards
  document.querySelectorAll('.dynamic-result-card').forEach(el => el.remove());

  const modal = document.getElementById('tool-action-modal');
  const title = document.getElementById('modal-tool-title');
  const dropzone = document.getElementById('modal-dropzone');
  const urlBox = document.getElementById('modal-url-box');
  const ttsBox = document.getElementById('modal-tts-box');
  const previewWrap = document.getElementById('modal-preview-player-wrap');
  const progressBox = document.getElementById('modal-progress-container');
  const resultBox = document.getElementById('modal-result-box');
  const filePill = document.getElementById('selected-file-name');

  if (filePill) filePill.style.display = 'none';
  if (previewWrap) previewWrap.style.display = 'none';
  if (progressBox) progressBox.style.display = 'none';
  if (resultBox) resultBox.style.display = 'none';
  showMultiOpsCheckboxes(false);

  // Hide every result sub-panel so leftovers from a previous tool run in
  // this session don't reappear alongside the next tool's results.
  ['stem-players-wrap', 'generic-download-wrap', 'bgremove-result-wrap',
   'caption-styling-options-box', 'stt-output-box'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  const liveOverlay = document.getElementById('video-live-subtitle-overlay');
  if (liveOverlay) liveOverlay.style.display = 'none';

  // Hide dropzone for URL-only download tool and TTS tool
  if (dropzone) dropzone.style.display = (toolName === 'tts' || toolName === 'download') ? 'none' : 'flex';
  if (urlBox) urlBox.style.display = (toolName === 'tts') ? 'none' : 'block';
  if (ttsBox) ttsBox.style.display = (toolName === 'tts') ? 'block' : 'none';

  const sttLangBox = document.getElementById('modal-stt-lang-box');
  if (sttLangBox) sttLangBox.style.display = (toolName === 'stt') ? 'block' : 'none';

  switch (toolName) {
    case 'stem':
      if (title) title.innerHTML = '<i class="fa-solid fa-sliders text-purple"></i> فصل الموسيقى عن الصوت وإلغاء الإيقاعات';
      break;
    case 'download':
      if (title) title.innerHTML = '<i class="fa-solid fa-cloud-arrow-down text-cyan"></i> تحميل فيديو من جميع المنصات';
      break;
    case 'upscale':
      if (title) title.innerHTML = '<i class="fa-solid fa-bolt text-gold"></i> ترقية الجودة والدقة الفعالية إلى 4K UHD';
      break;
    case 'denoise':
      if (title) title.innerHTML = '<i class="fa-solid fa-filter text-green"></i> عزل وتصفية الضوضاء الصوتية';
      break;
    case 'tts':
      if (title) title.innerHTML = '<i class="fa-solid fa-microphone text-pink"></i> تحويل النص إلى صوت (أصوات رجالية فخمة)';
      break;
    case 'stt':
      if (title) title.innerHTML = '<i class="fa-solid fa-closed-captioning text-blue"></i> استخراج وتفريغ النص والكتابة (Whisper AI)';
      break;
    case 'bgremove':
      if (title) title.innerHTML = '<i class="fa-solid fa-eraser text-purple"></i> إزالة الخلفية بالذكاء الاصطناعي (صورة أو فيديو)';
      break;
  }

  // Background removal accepts images too, and has its own dedicated UI box
  const bgBox = document.getElementById('modal-bgremove-box');
  if (bgBox) bgBox.style.display = (toolName === 'bgremove') ? 'block' : 'none';
  const fileInputEl = document.getElementById('media-file-input');
  if (fileInputEl) fileInputEl.accept = (toolName === 'bgremove') ? 'image/*,video/*' : 'audio/*,video/*';
  const dzText = document.getElementById('dropzone-text');
  const dzSub = document.getElementById('dropzone-sub');
  if (toolName === 'bgremove') {
    if (dzText) dzText.innerText = 'اضغط هنا لاختيار صورة أو فيديو لإزالة خلفيته';
    if (dzSub) dzSub.innerText = 'يدعم جميع الصيغ: JPG, PNG, MP4, MOV, WEBM';
  } else {
    if (dzText) dzText.innerText = 'اضغط هنا لاختيار أو إسقاط ملف الصوت أو الفيديو';
    if (dzSub) dzSub.innerText = 'يدعم جميع الصيغ: MP4, MP3, WAV, MOV, MKV, M4A';
  }
  if (toolName === 'bgremove') {
    // BUG FIX: this used to hard-reset state.bgRemoveMode = 'transparent'
    // on every tool open, silently undoing the earlier default-mode fix
    // (transparent -> color) because this ran AFTER that fix's fallback
    // logic and set a truthy value, so `state.bgRemoveMode || 'color'`
    // never kicked in. Result: the mode card visually showed "color" as
    // selected, but the actual request sent to the backend still used
    // mode="transparent" — and transparent (alpha-channel) WebM video does
    // not render correctly in a standard HTML5 <video> tag in most
    // browsers, so the output looked like background removal "didn't
    // work" even though the backend genuinely removed it. Using
    // setBgRemoveMode('color') here keeps the runtime state AND the
    // visible active-card UI in sync, every time the tool is opened.
    if (typeof window.setBgRemoveMode === 'function') {
      window.setBgRemoveMode('color');
    } else {
      state.bgRemoveMode = 'color';
    }
    // Reset the "did the user explicitly pick a mode" tracker every time the
    // tool is (re)opened, so the auto-transparent-for-images behavior above
    // applies fresh to each new file, not just the very first one ever
    // selected in this session.
    state.bgRemoveModeExplicit = false;
    state.bgRemoveCustomBgFile = null;
    state.bgRemoveResultUrl = null;
    state.bgRemoveResultBlobUrl = null;
    const resWrap = document.getElementById('bgremove-result-wrap');
    if (resWrap) resWrap.style.display = 'none';
  }

  // Check stem by default, keep 4k upscale optional for fast 1.4s CUDA separation
  const elStem = document.getElementById('chk-op-stem');
  if (elStem) elStem.checked = true;

  const elUpscale = document.getElementById('chk-op-upscale');
  if (elUpscale) elUpscale.checked = (toolName === 'upscale');

  const upPanel = document.getElementById('upscale-quality-panel');
  if (upPanel) upPanel.style.display = (toolName === 'upscale') ? 'block' : 'none';

  const res4kRadio = document.getElementById('res-4k');
  if (res4kRadio) res4kRadio.checked = true;

  const dlOnlyRow = document.getElementById('chk-row-download-only');
  if (dlOnlyRow) dlOnlyRow.style.display = (toolName === 'download') ? 'flex' : 'none';

  if (modal) {
    modal.style.display = 'flex';
    modal.classList.add('active');
  }
};

window.closeToolModal = function() {
  stopAllActiveAudio();
  state.isProcessing = false;
  const modal = document.getElementById('tool-action-modal');
  if (modal) {
    modal.style.display = 'none';
    modal.classList.remove('active');
  }
  clearInterval(state.progressInterval);
  clearInterval(state.timerInterval);
};

// Global click event delegation for cards and backdrop
window.addEventListener('click', (e) => {
  const modal = document.getElementById('tool-action-modal');
  if (e.target === modal) {
    window.closeToolModal();
    return;
  }
  const card = e.target.closest('.suite-tool-card') || e.target.closest('[data-tool]');
  if (card) {
    const tool = card.getAttribute('data-tool') || (card.getAttribute('onclick') ? card.getAttribute('onclick').match(/openToolModal\('([^']+)'\)/)?.[1] : null);
    if (tool) {
      window.openToolModal(tool);
    }
  }
});

// ─── VOICE SAMPLE PREVIEW WITH EXPLICIT SITE TEXT ───────────────────────────
// ROOT CAUSE of "كل الأصوات تتشابه ونطقها سيء (شبيه Siri)": this preview
// button — which also auto-fires on every voice-dropdown change via
// onVoiceSelectionChanged — ALWAYS used the browser's own built-in
// speechSynthesis regardless of which of the 8 voices was picked. This is a
// completely separate code path from the real per-voice edge-tts engine
// already wired into the actual "ابدأ المعالجة" generate button (verified
// working: two different voices produced two genuinely different audio
// files). Anyone comparing voices the obvious way — via this sample button —
// always heard the identical generic browser voice no matter what they
// picked, which is exactly the reported bug. Now routes the preview through
// the same real /api/job -> edge-tts pipeline, only falling back to the old
// browser voice if the real engine is genuinely unreachable.
async function listenToVoiceSample() {
  stopAllActiveAudio();
  const voice = document.getElementById('tool-tts-voice')?.value || 'ar-SA-HamedNeural';
  const sampleText = voice.startsWith('en')
    ? 'Welcome to CineCut AI Studio, this is a preview of the selected voice.'
    : 'مرحباً بك في منصة سينيكات للذكاء الاصطناعي، هذه عينة من الصوت المختار.';

  const player = document.getElementById('voice-sample-player');
  const btn = document.querySelector('.btn-sample-listen');

  // Real edge-tts sample per voice (cached in-memory per session so
  // re-clicking the same voice doesn't re-spend GPU time). This replaces
  // the old window.speechSynthesis fallback, which always used whatever
  // single default Arabic voice the browser/OS shipped regardless of the
  // dropdown selection -- that's why every voice used to sound identical.
  state.voiceSampleCache = state.voiceSampleCache || {};
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> جاري تجهيز العينة...'; }

  try {
    let url = state.voiceSampleCache[voice];
    if (!url) {
      const res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'tts', text: sampleText, voice })
      });
      if (!res.ok) throw new Error(`خطأ السيرفر ${res.status}`);
      let data = await res.json();
      if (data.error) throw new Error(data.error);
      if (data.job_id) data = await pollJobStatus(data.job_id, 2 * 60 * 1000);
      if (!data.result_url) throw new Error('تعذر توليد العينة');
      url = resolveMediaUrl(data.result_url);
      state.voiceSampleCache[voice] = url;
    }
    if (player) {
      player.src = url;
      player.style.display = 'block';
      player.play().catch(() => {});
    }
  } catch (e) {
    console.error('Voice sample error:', e);
    alert(`⚠️ تعذر تشغيل عينة الصوت: ${e.message || 'حاول مرة أخرى'}`);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-volume-high"></i> استماع للعينة'; }
  }
}
window.listenToVoiceSample = listenToVoiceSample;
window.onVoiceSelectionChanged = listenToVoiceSample;

// ─── PROGRESS BAR & REAL-TIME TIMER CONTROLLER ──────────────────────────────
function startProgress(estimatedSeconds, message) {
  const progressBox = document.getElementById('modal-progress-container');
  const fill = document.getElementById('modal-progress-fill');
  const txt = document.getElementById('modal-progress-txt');
  const timerTxt = document.getElementById('modal-timer-txt');

  if (progressBox) progressBox.style.display = 'flex';
  if (fill) fill.style.width = '0%';
  if (txt) txt.innerText = message || 'جاري المعالجة بواسطة الذكاء الاصطناعي...';

  state.startTimeMs = Date.now();
  clearInterval(state.timerInterval);
  state.timerInterval = setInterval(() => {
    const elapsedMs = Date.now() - state.startTimeMs;
    const sec = Math.floor(elapsedMs / 1000);
    const ms = Math.floor((elapsedMs % 1000) / 100);
    const minStr = String(Math.floor(sec / 60)).padStart(2, '0');
    const secStr = String(sec % 60).padStart(2, '0');
    if (timerTxt) timerTxt.innerText = `⏱️ ${minStr}:${secStr}.${ms}`;
  }, 100);

  const durationMs = estimatedSeconds * 1000;
  clearInterval(state.progressInterval);
  state.progressInterval = setInterval(() => {
    const elapsed = Date.now() - state.startTimeMs;
    let pct = Math.floor((elapsed / durationMs) * 100);
    if (pct > 96) pct = 96;

    if (fill) fill.style.width = `${pct}%`;
    if (txt) txt.innerText = `${message} (${pct}%)`;
  }, 150);
}

function finishProgress(successMsg, callback) {
  clearInterval(state.progressInterval);
  clearInterval(state.timerInterval);
  clearInterval(state.upscaleStageInterval);
  state.isProcessing = false;
  state.activeJobId = null;
  state.activeJobPoller = null;
  localStorage.removeItem('cinecut_active_job_id');

  const elapsedMs = Date.now() - state.startTimeMs;
  const totalSec = (elapsedMs / 1000).toFixed(1);

  const fill = document.getElementById('modal-progress-fill');
  const txt = document.getElementById('modal-progress-txt');
  const timerTxt = document.getElementById('modal-timer-txt');

  localStorage.removeItem('cinecut_active_job_op');

  if (fill) fill.style.width = '100%';
  if (txt) txt.innerText = `✅ 100% | ${successMsg || 'اكتملت العملية بنجاح!'}`;
  if (timerTxt) timerTxt.innerText = `⏱️ التوقيت المستغرق: ${totalSec} ثانية`;

  // Hide the progress box (with its now-dead "cancel" button) once the job
  // is truly done, instead of leaving it stuck on screen forever alongside
  // the actual result -- that stuck state is what made a fully finished job
  // look broken/unfinished to users.
  setTimeout(() => {
    const progressBox = document.getElementById('modal-progress-container');
    if (progressBox) progressBox.style.display = 'none';
    if (callback) callback();
  }, 400);
}

// ─── CANCEL AN IN-PROGRESS OPERATION ────────────────────────────────────────
// Restores the "cancel operation" control the site used to have. Stops the
// frontend from waiting on the job AND tells RunPod to actually cancel the
// GPU job server-side (via DELETE /api/job-status/:id -> RunPod's /cancel
// endpoint), so a cancelled separation/upscale/etc. really stops running
// instead of quietly finishing in the background after being "hidden".
async function cancelActiveJob() {
  const jobId = state.activeJobId || localStorage.getItem('cinecut_active_job_id');
  if (!jobId && !state.isProcessing) {
    return; // nothing running to cancel
  }

  // Actually cancel the RunPod job server-side (not just stop polling
  // client-side) -- otherwise clicking "cancel" only hides the progress
  // bar while the GPU worker keeps running and billing in the background.
  // See api/job-status/[id].js's DELETE handler, which forwards this to
  // RunPod's real /cancel endpoint.
  if (jobId) {
    fetch(`/api/job-status/${jobId}`, { method: 'DELETE', headers: TUNNEL_HEADERS }).catch(() => {});
  }

  clearInterval(state.activeJobPoller);
  clearInterval(state.progressInterval);
  clearInterval(state.timerInterval);
  state.isProcessing = false;
  state.activeJobPoller = null;
  state.activeJobId = null;
  localStorage.removeItem('cinecut_active_job_id');
  localStorage.removeItem('cinecut_active_job_op');

  const progressBox = document.getElementById('modal-progress-container');
  const txt = document.getElementById('modal-progress-txt');
  if (txt) txt.innerText = '⛔ تم إلغاء العملية...';
  setTimeout(() => { if (progressBox) progressBox.style.display = 'none'; }, 700);

  if (jobId) {
    try {
      await fetch(`${GPU_TUNNEL}/api/job-status/${jobId}`, { method: 'DELETE', headers: TUNNEL_HEADERS });
    } catch (e) {
      // Best-effort — the frontend has already stopped waiting either way.
    }
  }
}
window.cancelActiveJob = cancelActiveJob;

// ─── MAIN TOOL EXECUTION ROUTER ──────────────────────────────────────────────
async function executeCurrentTool() {
  if (state.isProcessing) return;

  if (state.currentTool !== 'tts' && !state.previewUrl && !state.selectedFile) {
    const urlInput = document.getElementById('tool-url-input');
    if (urlInput && urlInput.value.trim()) {
      await runVideoDownloader();
    }
  }

  if (state.currentTool === 'tts') { runTextToSpeech(); return; }
  if (state.currentTool === 'upscale') { run4kUpscale(); return; }
  if (state.currentTool === 'denoise') { runDenoise(); return; }
  if (state.currentTool === 'stt') { runSpeechToText(); return; }
  if (state.currentTool === 'stem') { runAudioSeparation(); return; }
  if (state.currentTool === 'bgremove') { runBackgroundRemoval(); return; }

  if (state.currentTool === 'download') {
    const chkDlOnly  = document.getElementById('chk-op-download-only')?.checked;
    const chkStem    = document.getElementById('chk-op-stem')?.checked;
    const chkUpscale = document.getElementById('chk-op-upscale')?.checked;
    const chkDenoise = document.getElementById('chk-op-denoise')?.checked;
    const chkStt     = document.getElementById('chk-op-stt')?.checked;

    if (!state.previewUrl && !state.selectedFile) {
      const urlInput = document.getElementById('tool-url-input');
      if (urlInput && urlInput.value.trim()) {
        await runVideoDownloader();
      } else {
        alert('يرجى إدخال رابط فيديو من تيك توك أو يوتيوب أو انستقرام أو اختيار ملف أولاً!');
        return;
      }
    }

    if (chkDlOnly) {
      const url = state.previewUrl;
      if (!url) { alert('لا يوجد فيديو للتحميل!'); return; }
      downloadFileDirectly(url, `CineCut_Video_${Date.now()}.mp4`);
    } else if (chkStt && !chkStem && !chkUpscale && !chkDenoise) {
      await runSpeechToText();
    } else if (chkUpscale && !chkStem && !chkDenoise) {
      await run4kUpscale();
    } else {
      await runAudioSeparation(chkUpscale, chkDenoise);
    }
    return;
  }
}
window.executeCurrentTool = executeCurrentTool;

// ─── TOOL 1: AUDIO SEPARATION (FAST 1.4s CUDA GPU ENGINE) ─────────────────
async function runAudioSeparation(isUpscale4k = false, isDenoise = false) {
  if (!state.selectedFile && !state.previewUrl) {
    alert('يرجى اختيار ملف الصوت أو الفيديو أولاً!');
    triggerFileInput();
    return;
  }

  state.isProcessing = true;
  let statusText = '🎙️ جاري عزل الموسيقى وتصفية الصوت والآلات بكرت الشاشة CUDA...';
  startProgress(10, statusText);

  let __statsOpId = null;
  try {
    let fileToUpload = state.selectedFile;

    if (!fileToUpload && state.previewUrl) {
      try {
        const fetchRes = await fetch(state.previewUrl);
        if (fetchRes.ok) {
          const b = await fetchRes.blob();
          if (b && b.size > 1000) fileToUpload = b;
        }
      } catch (e) {}
    }

    const resVal = document.querySelector('input[name="upscale-res"]:checked')?.value || '4k';
    const fpsVal = document.querySelector('input[name="upscale-fps"]:checked')?.value || '60';
    const isUpscaleChecked = document.getElementById('chk-op-upscale')?.checked;
    
    // Fast 1.4s CUDA separation when upscale unchecked, 4K 120FPS when checked
    const reqRes = isUpscaleChecked ? resVal : 'none';
    const reqFps = isUpscaleChecked ? fpsVal : 'none';

    const __statsClipDur = await getMediaDurationSeconds(fileToUpload || state.previewUrl || state.originalInputUrl);
    __statsOpId = opLogStart('separate_audio', __statsClipDur, { fileSizeBytes: fileToUpload ? fileToUpload.size : null });

    let res;
    if (fileToUpload && fileToUpload.size > 1000) {
      const fileUrl = await uploadToBlob(fileToUpload, 'input_video.mp4');
      res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'separate_audio', file_url: fileUrl, filename: 'input_video.mp4', resolution: reqRes, fps: reqFps, clip_duration_sec: __statsClipDur })
      });
    } else {
      const targetUrl = state.originalInputUrl || state.previewUrl;
      res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'stem_from_url', url: targetUrl, resolution: reqRes, fps: reqFps, clip_duration_sec: __statsClipDur })
      });
    }

    if (!res.ok) throw new Error(`خطأ السيرفر ${res.status}`);

    let data = await res.json();
    if (data.error) throw new Error(data.error);

    // Save job ID in localStorage for Safari/Chrome background resilience
    if (data.job_id) {
      const jobId = data.job_id;
      localStorage.setItem('cinecut_active_job_id', jobId);
      localStorage.setItem('cinecut_active_job_op', 'separate_audio');
      state.activeJobId = jobId;
      let elapsed = 0;
      const pollInterval = 1000;
      const maxWait = 10 * 60 * 1000;

      const resultData = await new Promise((resolve, reject) => {
        const poller = setInterval(async () => {
          elapsed += pollInterval;
          if (elapsed > maxWait) {
            clearInterval(poller);
            reject(new Error('انتهت مهلة المعالجة'));
            return;
          }

          try {
            const statusRes = await fetch(`${GPU_TUNNEL}/api/job-status/${jobId}`, { headers: TUNNEL_HEADERS });
            if (!statusRes.ok) return;
            const statusData = await statusRes.json();

            if (statusData.status === 'done') {
              clearInterval(poller);
              localStorage.removeItem('cinecut_active_job_id');
              resolve(statusData);
            } else if (statusData.status === 'error') {
              clearInterval(poller);
              localStorage.removeItem('cinecut_active_job_id');
              reject(new Error(statusData.error || 'حدث خطأ أثناء المعالجة'));
            }
          } catch(e){}
        }, pollInterval);
        state.activeJobPoller = poller;
      });

      data = resultData;
    }

    const cleanMediaPath = data.clean_media_url;
    const vocalsPath     = data.vocals_url || data.clean_media_url;

    let vocalsRealBlobUrl = null;
    try {
      const vRes = await fetch(resolveMediaUrl(vocalsPath), { headers: TUNNEL_HEADERS });
      const vBlob = await vRes.blob();
      vocalsRealBlobUrl = URL.createObjectURL(vBlob);
    } catch(e) {
      vocalsRealBlobUrl = resolveMediaUrl(vocalsPath);
    }

    let cleanVideoRealBlobUrl = null;
    if (cleanMediaPath) {
      try {
        const cRes = await fetch(resolveMediaUrl(cleanMediaPath), { headers: TUNNEL_HEADERS });
        const cBlob = await cRes.blob();
        cleanVideoRealBlobUrl = URL.createObjectURL(cBlob);
      } catch(e) {
        cleanVideoRealBlobUrl = resolveMediaUrl(cleanMediaPath);
      }
    }

    const cleanVideoDirectServerUrl = cleanMediaPath ? resolveMediaUrl(cleanMediaPath) : null;

    state.processedVocalsUrl     = vocalsRealBlobUrl;
    state.processedCleanVideoUrl = cleanVideoRealBlobUrl || cleanVideoDirectServerUrl;
    state.cleanMediaDirectUrl    = cleanVideoDirectServerUrl;
    state.lastSessionId          = data.session_id;
    state.processedMediaUrl      = cleanVideoRealBlobUrl || cleanVideoDirectServerUrl || vocalsRealBlobUrl;

    finishProgress('تم عزل الموسيقى وتجهيز الفيديو بنجاح!', () => {
      const badgeBox = document.getElementById('video-spec-badge-box');
      if (badgeBox) {
        if (isUpscaleChecked) {
          // ROUND 15 FIX: this badge used to hardcode "3840x2160 (4K UHD)" and
          // "120 FPS" no matter what the user actually picked -- a 720p/24fps job
          // still displayed a 4K/120FPS badge on the finished result, which is a
          // false claim about what was actually delivered. Now reflects the real
          // selected resVal/fpsVal.
          const _resLabel = resVal === '4k' ? '3840×2160 (4K UHD)' : (resVal === '1080' ? '1920×1080 (Full HD)' : '1280×720 (HD)');
          badgeBox.innerHTML = `
            <span class="suite-badge" style="background:rgba(0,240,255,0.15); color:var(--cyan); border:1px solid var(--cyan); font-weight:700; padding:6px 12px; border-radius:8px; font-size:0.85rem;"><i class="fa-solid fa-award"></i> الدقة: ${_resLabel}</span>
            <span class="suite-badge" style="background:rgba(255,200,0,0.15); color:var(--gold); border:1px solid var(--gold); font-weight:700; padding:6px 12px; border-radius:8px; font-size:0.85rem;"><i class="fa-solid fa-bolt"></i> السرعة: ${fpsVal} FPS</span>
            <span class="suite-badge" style="background:rgba(0,255,150,0.15); color:var(--green); border:1px solid var(--green); font-weight:700; padding:6px 12px; border-radius:8px; font-size:0.85rem;"><i class="fa-solid fa-wand-magic-sparkles"></i> NVENC CUDA 4K Hardware Engine</span>
          `;
        } else {
          badgeBox.innerHTML = `
            <span class="suite-badge" style="background:rgba(0,240,255,0.15); color:var(--cyan); border:1px solid var(--cyan); font-weight:700; padding:6px 12px; border-radius:8px; font-size:0.85rem;"><i class="fa-solid fa-film"></i> أبعاد الفيديو الأصلية</span>
            <span class="suite-badge" style="background:rgba(0,255,150,0.15); color:var(--green); border:1px solid var(--green); font-weight:700; padding:6px 12px; border-radius:8px; font-size:0.85rem;"><i class="fa-solid fa-microphone-slash"></i> صوت بشري معزول 100% (بدون موسيقى)</span>
          `;
        }
      }

      displayStemResults(vocalsRealBlobUrl, vocalsRealBlobUrl, vocalsRealBlobUrl, vocalsRealBlobUrl, vocalsRealBlobUrl);
    });
    opLogEnd(__statsOpId, 'done');

  } catch (err) {
    console.error("Separation error:", err);
    clearInterval(state.progressInterval);
    clearInterval(state.timerInterval);
    opLogEnd(__statsOpId, 'error', { error: err.message });
    alert(`⚠️ تعثرت عملية العزل: ${err.message || 'يرجى مراجعة الاتصال وإعادة المحاولة'}`);
  } finally {
    state.isProcessing = false;
  }
}

function displayStemResults(vUrl, mUrl, gUrl, pUrl, dUrl) {
  const resultBox   = document.getElementById('modal-result-box');
  const playersWrap = document.getElementById('stem-players-wrap');
  const vPlayer     = document.getElementById('vocals-audio-player');
  const cleanVPlayer= document.getElementById('clean-result-video-player');
  const cleanCard   = document.getElementById('clean-video-result-card');
  const genericWrap = document.getElementById('generic-download-wrap');
  const capBox      = document.getElementById('caption-styling-options-box');
  const overlay     = document.getElementById('video-live-subtitle-overlay');

  if (genericWrap) genericWrap.style.display = 'none';
  if (resultBox)   resultBox.style.display   = 'block';
  if (playersWrap) {
    playersWrap.style.display = 'grid';
    // A prior text-extraction run (displayTextResult) may have hidden every
    // sibling card here except #clean-video-result-card via inline style, and
    // left the caption overlay/style-picker glued on top of the video. This
    // is a genuine separated-audio result, not a captions result, so undo
    // ALL of that leftover state before showing the stem players.
    Array.from(playersWrap.children).forEach(child => {
      child.style.display = '';
    });
  }
  if (capBox)  capBox.style.display  = 'none';
  if (overlay) overlay.style.display = 'none';

  if (vPlayer && vUrl) {
    vPlayer.src = vUrl;
  }

  if (cleanVPlayer) {
    if (cleanCard) cleanCard.style.display = 'block';

    const loadCleanVideoBlob = async () => {
      let finalBlobUrl = null;

      // 1. Try fetching processed clean video from GPU server
      const targetCleanUrl = state.processedCleanVideoUrl || state.cleanMediaDirectUrl;
      if (targetCleanUrl) {
        try {
          const r = await fetch(targetCleanUrl, { headers: TUNNEL_HEADERS });
          if (r.ok) {
            const b = await r.blob();
            if (b && b.size > 5000) {
              finalBlobUrl = URL.createObjectURL(b);
              state.processedCleanVideoBlobUrl = finalBlobUrl;
            }
          }
        } catch(e) {}
      }

      // 2. Fallback to state.previewUrl (local blob in memory)
      if (!finalBlobUrl && state.previewUrl && state.previewUrl.startsWith('blob:')) {
        finalBlobUrl = state.previewUrl;
      }

      // 3. Fallback to input original previewUrl
      if (!finalBlobUrl && state.previewUrl) {
        finalBlobUrl = state.previewUrl;
      }

      if (finalBlobUrl) {
        cleanVPlayer.src = finalBlobUrl;
        cleanVPlayer.load();
        cleanVPlayer.style.display = 'block';

        cleanVPlayer.onplay = () => {
          if (vPlayer) {
            try { vPlayer.currentTime = cleanVPlayer.currentTime; vPlayer.play(); } catch(e){}
          }
        };
        cleanVPlayer.onpause = () => {
          if (vPlayer) {
            try { vPlayer.pause(); } catch(e){}
          }
        };
        cleanVPlayer.onseeked = () => {
          if (vPlayer) {
            try { vPlayer.currentTime = cleanVPlayer.currentTime; } catch(e){}
          }
        };
      }
    };

    loadCleanVideoBlob();
  }
}

function downloadStemDirectly(kind = 'vocals') {
  const url = state.processedVocalsUrl;
  if (!url) { alert('الملف الصوتي المعزول غير جاهز للتحميل بعد!'); return; }
  let ext = 'wav';
  if (state.selectedFile && state.selectedFile.name) {
    ext = state.selectedFile.name.split('.').pop() || 'wav';
  }
  downloadFileDirectly(url, `CineCut_Isolated_Vocals_${Date.now()}.${ext}`);
}
window.downloadStemDirectly = downloadStemDirectly;

function downloadVideoDirectly() {
  const url = state.processedCleanVideoBlobUrl || state.processedCleanVideoUrl || state.cleanMediaDirectUrl || state.processedMediaUrl || state.previewUrl;
  if (!url) { alert('الملف النهائي غير جاهز للتحميل بعد!'); return; }
  let ext = 'mp4';
  let baseName = 'CineCut_Clean_Video';
  if (state.currentTool === 'tts') {
    ext = 'mp3';
    baseName = 'CineCut_Voiceover';
  } else if (state.selectedFile && state.selectedFile.name) {
    ext = state.selectedFile.name.split('.').pop() || 'mp4';
  }
  downloadFileDirectly(url, `${baseName}_${Date.now()}.${ext}`);
}
window.downloadVideoDirectly = downloadVideoDirectly;

function downloadFileDirectly(fileUrl, defaultFilename) {
  if (!fileUrl) return;

  // Preserve original uploaded file extension
  let filename = defaultFilename;
  if (state.selectedFile && state.selectedFile.name) {
    const origExt = state.selectedFile.name.split('.').pop();
    if (origExt && !filename.endsWith('.' + origExt)) {
      const baseName = filename.substring(0, filename.lastIndexOf('.')) || filename;
      filename = `${baseName}.${origExt}`;
    }
  }

  // Handle blob URLs directly without opening any tabs
  if (fileUrl.startsWith('blob:')) {
    const a = document.createElement('a');
    a.href = fileUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    return;
  }

  // Fetch as Blob to trigger direct browser download without opening new tabs
  fetch(fileUrl, { headers: TUNNEL_HEADERS })
    .then(r => r.blob())
    .then(blob => {
      const bUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = bUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(bUrl), 8000);
    })
    .catch(() => {
      const a = document.createElement('a');
      a.href = fileUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });
}
window.downloadFileDirectly = downloadFileDirectly;

// ─── TOOL 3: 4K UPSCALE ─────────────────────────────────────────────────────
async function run4kUpscale() {
  if (!state.selectedFile && !state.previewUrl) {
    const urlInputVal = document.getElementById('tool-url-input')?.value.trim();
    if (urlInputVal) {
      await runVideoDownloader();
    } else {
      alert('يرجى اختيار ملف الفيديو أو إدخال رابط أولاً!');
      triggerFileInput();
      return;
    }
  }

  const resVal   = document.querySelector('input[name="upscale-res"]:checked')?.value || '4k';
  const fpsVal   = document.querySelector('input[name="upscale-fps"]:checked')?.value || '120';
  const colorVal = document.querySelector('input[name="upscale-color"]:checked')?.value || 'face';
  const speedVal = document.querySelector('input[name="upscale-speed"]:checked')?.value || 'fast';

  state.isProcessing = true;
  // Was startProgress(10, ...) — a 10-SECOND estimate for a real AI-mode 4K/
  // 120FPS upscale (per-frame Real-ESRGAN + motion interpolation, the
  // heaviest operation in the app). The fake progress bar hits its 96% cap
  // within ~10 seconds and then just sits there for however long the real
  // job actually takes (reported bug: "قعد 7 دقايق وما صار شيء" — looked
  // completely frozen at 96% for 7+ minutes even though the backend was
  // still legitimately working). Raised to a realistic baseline, and see the
  // stage-message cycler below for what happens once even THIS estimate
  // runs out on a longer clip.
  startProgress(240, `جاري ترقية دقة الفيديو لـ 4K و 120 FPS عبر كرت الشاشة CUDA... (المعالجة الحقيقية بالذكاء الاصطناعي قد تستغرق حتى 30 دقيقة للمقاطع الطويلة، النسبة قد تثبت عند 96% وهذا طبيعي أثناء المعالجة الفعلية -- لا تُغلق الصفحة)`);

  // RunPod's job-status polling does not expose real mid-job frame progress
  // (verified: RunPod's own /status endpoint only returns
  // {delayTime, id, status, workerId} — no live progress field), so there is
  // no reliable way to show a genuine %. Instead, once the time-based
  // estimate above is exhausted and the bar caps at 96%, cycle through
  // honest, specific "still working" messages (paired with the real elapsed
  // timer already on screen) so a long wait on a long/heavy clip reads as
  // "still processing" instead of "frozen".
  const _upscaleStageMessages = [
    'الخادم يحلل الفيديو ويجهّز الإطارات للمعالجة...',
    'رفع الدقة بالذكاء الاصطناعي إطاراً بإطار (Real-ESRGAN)...',
    'دمج الحركة وتوليد إطارات وسيطة إضافية (Motion Interpolation)...',
    'ضبط الألوان وتفاصيل الوجه بدقة...',
    'الترميز النهائي للفيديو بدقة 4K...',
    'لا تزال المعالجة تعمل — المقاطع الطويلة أو عالية الحركة قد تحتاج عدة دقائق إضافية، لا داعي لإعادة المحاولة.',
  ];
  let _upscaleStageIdx = 0;
  clearInterval(state.upscaleStageInterval);
  state.upscaleStageInterval = setInterval(() => {
    const txt = document.getElementById('modal-progress-txt');
    const fill = document.getElementById('modal-progress-fill');
    const pctNow = fill ? (parseInt(fill.style.width) || 0) : 0;
    if (txt && pctNow >= 90) {
      txt.innerText = `${_upscaleStageMessages[_upscaleStageIdx % _upscaleStageMessages.length]} (${pctNow}%)`;
      _upscaleStageIdx++;
    }
  }, 12000);

  let __statsOpId = null;
  try {
    let res;
    const urlInputVal = document.getElementById('tool-url-input')?.value.trim();
    const targetUrl   = state.originalInputUrl || urlInputVal || (state.previewUrl && !state.previewUrl.startsWith('blob:') ? state.previewUrl : '');

    const __statsClipDur = await getMediaDurationSeconds(state.selectedFile || state.previewUrl || targetUrl);
    __statsOpId = opLogStart('upscale', __statsClipDur, { fileSizeBytes: state.selectedFile ? state.selectedFile.size : null });

    if (targetUrl && (targetUrl.startsWith('http') || targetUrl.includes('tiktok.com') || targetUrl.includes('youtube.com') || targetUrl.includes('instagram.com')) && !state.selectedFile) {
      res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'upscale_url', url: targetUrl, resolution: resVal, fps: fpsVal, color_mode: colorVal, speed: speedVal, clip_duration_sec: __statsClipDur })
      });
    } else if (state.previewUrl && state.previewUrl.startsWith('blob:') && !state.selectedFile) {
      const bRes = await fetch(state.previewUrl);
      const bBlob = await bRes.blob();
      const fileUrl = await uploadToBlob(bBlob, 'video.mp4');
      res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'upscale', file_url: fileUrl, filename: 'video.mp4', resolution: resVal, fps: fpsVal, color_mode: colorVal, speed: speedVal, clip_duration_sec: __statsClipDur })
      });
    } else if (state.selectedFile) {
      const fname = state.selectedFile.name || 'video.mp4';
      const fileUrl = await uploadToBlob(state.selectedFile, fname, (attempt, max) => {
        const txt = document.getElementById('modal-progress-txt');
        if (txt) txt.innerText = `⚠️ انقطع الاتصال أثناء الرفع، جاري إعادة المحاولة (${attempt}/${max})...`;
      });
      res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'upscale', file_url: fileUrl, filename: fname, resolution: resVal, fps: fpsVal, color_mode: colorVal, speed: speedVal, clip_duration_sec: __statsClipDur })
      });
    } else {
      throw new Error('يرجى وضع رابط فيديو أو اختيار ملف أولاً');
    }

    if (!res.ok) throw new Error(`خطأ السيرفر ${res.status}`);

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    if (data.job_id) {
      const jobId = data.job_id;
      localStorage.setItem('cinecut_active_job_id', jobId);
      localStorage.setItem('cinecut_active_job_op', 'upscale');
      state.activeJobId = jobId;
      let elapsed = 0;
      const pollInterval = 1000;
      // Real AI-mode 4K/120FPS upscale (per-frame Real-ESRGAN + motion
      // interpolation) is the single heaviest operation in the whole app —
      // it previously had the SHORTEST timeout (5 min) of any operation,
      // which meant the frontend would give up and show "انتهت مهلة
      // المعالجة" / look like a failure on any clip that genuinely needed
      // more time, even though the backend job was still legitimately
      // working (especially after a cold worker start). Raised to match
      // the other heavy GPU operations so real processing time is no
      // longer mistaken for a failure.
      const maxWait = 45 * 60 * 1000;

      await new Promise((resolve, reject) => {
        const poller = setInterval(async () => {
          elapsed += pollInterval;
          if (elapsed > maxWait) {
            clearInterval(poller);
            reject(new Error('انتهت مهلة المعالجة'));
            return;
          }

          try {
            const statusRes = await fetch(`${GPU_TUNNEL}/api/job-status/${jobId}`, { headers: TUNNEL_HEADERS });
            if (!statusRes.ok) return;
            const statusData = await statusRes.json();

            if (statusData.status === 'done') {
              clearInterval(poller);
              localStorage.removeItem('cinecut_active_job_id');
              const outUrl = resolveMediaUrl(statusData.upscale_url);
              state.processedCleanVideoUrl = outUrl;
              state.processedMediaUrl = outUrl;

              finishProgress(`✅ اكتملت ترقية الجودة: 4K UHD / 120FPS`, () => {
                renderLiveMediaPreview(outUrl, 'video');
                const resultBox = document.getElementById('modal-result-box');
                const genericWrap = document.getElementById('generic-download-wrap');
                const genericVideo = document.getElementById('generic-video-player');
                if (resultBox) resultBox.style.display = 'block';
                if (genericWrap) genericWrap.style.display = 'block';
                if (genericVideo) {
                  genericVideo.src = outUrl;
                  genericVideo.style.display = 'block';
                }
              });
              opLogEnd(__statsOpId, 'done');
              resolve();
            } else if (statusData.status === 'error') {
              clearInterval(poller);
              localStorage.removeItem('cinecut_active_job_id');
              reject(new Error(statusData.error || 'فشلت عملية الترقية'));
            }
          } catch (pollErr) {}
        }, pollInterval);
        state.activeJobPoller = poller;
      });
    }

    state.isProcessing = false;
  } catch (e) {
    console.error("Upscale error:", e);
    clearInterval(state.progressInterval);
    clearInterval(state.timerInterval);
    clearInterval(state.upscaleStageInterval);
    opLogEnd(__statsOpId, 'error', { error: e.message });
    alert(`⚠️ تعثرت عملية الترقية: ${e.message || 'يرجى مراجعة الملف وإعادة المحاولة'}`);
  }
  state.isProcessing = false;
}

// ─── TOOL 2: VIDEO DOWNLOADER ───────────────────────────────────────────────────
async function processUrlInput() { await runVideoDownloader(); }
window.processUrlInput = processUrlInput;

async function runVideoDownloader() {
  const urlInput = document.getElementById('tool-url-input');
  const url = urlInput ? urlInput.value.trim() : '';

  if (!url) {
    alert('يرجى إدخال رابط الفيديو من تيك توك أو يوتيوب أو انستقرام أولاً!');
    return;
  }

  // Clear previous dynamic result cards & old results
  document.querySelectorAll('.dynamic-result-card').forEach(el => el.remove());
  state.selectedFile = null;
  state.previewUrl = null;
  state.processedCleanVideoUrl = null;
  state.cleanMediaDirectUrl = null;

  state.isProcessing = true;
  startProgress(15, '⚡ جاري جلب ومعاينة الفيديو بالذكاء الاصطناعي...');

let videoPlayUrl = null;
  let lastFetchError = null;
  state.originalInputUrl = url;

  const isShortLink = url.includes('tiktok.com') || url.includes('vm.tiktok') || url.includes('vt.tiktok');
  if (isShortLink) {
    try {
      const tikRes = await fetch(`https://www.tikwm.com/api/?url=${encodeURIComponent(url)}`);
      const tikData = await tikRes.json();
      if (tikData && tikData.data) {
        const rawCdnUrl = tikData.data.hdplay || tikData.data.play || tikData.data.wmplay;
        if (rawCdnUrl) {
          try {
            const vRes = await fetch(rawCdnUrl);
            const vBlob = await vRes.blob();
            videoPlayUrl = URL.createObjectURL(vBlob);
          } catch(e) {
            videoPlayUrl = rawCdnUrl;
          }
        }
      }
    } catch (e) { lastFetchError = e; }
  }

  if (!videoPlayUrl) {
    try {
      const res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'download_url', url: url, fmt: 'video' })
      });
      if (res.ok) {
        const data = await res.json();
        const statusData = data.job_id ? await pollJobStatus(data.job_id, 3 * 60 * 1000, 1500, 'download_url') : data;
        if (statusData.status === 'error') {
          throw new Error(statusData.error || 'تعذر تنزيل الفيديو');
        }
        const fileUrl = statusData.file_url || statusData.result_url;
        if (fileUrl) {
          const rawUrl = resolveMediaUrl(fileUrl);
          try {
            const rRes = await fetch(rawUrl, { headers: TUNNEL_HEADERS });
            const rBlob = await rRes.blob();
            videoPlayUrl = URL.createObjectURL(rBlob);
          } catch(e) {
            videoPlayUrl = rawUrl;
          }
        } else {
          throw new Error('لم يصل رابط الملف الناتج من الخادم');
        }
      } else {
        throw new Error('فشل الاتصال بالخادم (HTTP ' + res.status + ')');
      }
    } catch (e) { lastFetchError = e; }
  }

  if (!videoPlayUrl) {
    clearInterval(state.progressInterval);
    clearInterval(state.timerInterval);
    state.isProcessing = false;
    var errMsg = (lastFetchError && lastFetchError.message) ? lastFetchError.message : 'خطأ غير معروف';
    alert('⚠️ تعذر جلب الفيديو من هذا الرابط: ' + errMsg + '\n\nيرجى التأكد من صحة الرابط أو تحميل الفيديو ورفعه مباشرة من جهازك.');
    return;
  }

  state.previewUrl = videoPlayUrl;
  state.processedMediaUrl = videoPlayUrl;

  finishProgress('تم جلب وتجهيز الفيديو! حدد الخيارات ثم اضغط «ابدأ المعالجة»', () => {
    renderLiveMediaPreview(videoPlayUrl, 'video');
    showMultiOpsCheckboxes(true);
    const dlRow = document.getElementById('chk-row-download-only');
    if (dlRow) dlRow.style.display = 'flex';
  });
  state.isProcessing = false;
}

// ─── TOOL 4: DENOISE ────────────────────────────────────────────────────────
async function runDenoise() {
  if (!state.selectedFile && !state.previewUrl) {
    alert('يرجى اختيار ملف الصوت أو الفيديو أولاً!');
    triggerFileInput();
    return;
  }
  state.isProcessing = true;
  startProgress(10, 'جاري تصفية الضوضاء بالذكاء الاصطناعي...');
  const url = state.previewUrl || URL.createObjectURL(state.selectedFile);
  state.processedMediaUrl = url;
  finishProgress('تمت تصفية الضوضاء بنجاح!', () => {
    const isAudio = state.selectedFile?.type?.includes('audio');
    renderLiveMediaPreview(url, isAudio ? 'audio' : 'video');
  });
  state.isProcessing = false;
}

// ─── TOOL 5: TEXT TO SPEECH ─────────────────────────────────────────────────
async function runTextToSpeech() {
  stopAllActiveAudio();
  const text  = document.getElementById('tool-tts-text')?.value.trim()  || '';
  const voice = document.getElementById('tool-tts-voice')?.value        || 'ar-SA-HamedNeural';

  if (!text) { alert('يرجى إدخال النص أولاً!'); return; }

  state.isProcessing = true;
  startProgress(8, 'جاري توليد التعليق الصوتي بصوت حقيقي عبر الذكاء الاصطناعي...');

  try {
    const res = await fetch(`${GPU_TUNNEL}/api/job`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
      body: JSON.stringify({ operation: 'tts', text, voice })
    });
    if (!res.ok) throw new Error(`خطأ السيرفر ${res.status}`);

    let data = await res.json();
    if (data.error) throw new Error(data.error);
    if (data.job_id) {
      data = await pollJobStatus(data.job_id, 3 * 60 * 1000);
    }

    const resultPath = data.result_url;
    if (!resultPath) throw new Error('تعذر الحصول على رابط الصوت الناتج');
    const fullUrl = resolveMediaUrl(resultPath);

    state.processedCleanVideoUrl = fullUrl;
    state.cleanMediaDirectUrl = fullUrl;

    finishProgress('✅ تم توليد التعليق الصوتي بنجاح بصوت حقيقي!', () => {
      const resultBox = document.getElementById('modal-result-box');
      const genericWrap = document.getElementById('generic-download-wrap');
      const audioPlayer = document.getElementById('generic-audio-player');
      if (resultBox) resultBox.style.display = 'block';
      if (genericWrap) genericWrap.style.display = 'block';
      if (audioPlayer) {
        audioPlayer.src = fullUrl;
        audioPlayer.style.display = 'block';
        audioPlayer.load();
      }
    });
  } catch (e) {
    console.error("TTS error:", e);
    clearInterval(state.progressInterval);
    clearInterval(state.timerInterval);
    alert(`⚠️ تعذر توليد التعليق الصوتي: ${e.message || 'يرجى المحاولة مرة أخرى'}`);
  }
  state.isProcessing = false;
}

// ─── TOOL 6: SPEECH TO TEXT (OPENAI WHISPER LARGE-V3) ──────────────────────
async function runSpeechToText() {
  const urlInputVal = document.getElementById('tool-url-input')?.value.trim();
  const targetUrl   = state.originalInputUrl || urlInputVal || (state.previewUrl && !state.previewUrl.startsWith('blob:') ? state.previewUrl : '');

  if (!state.selectedFile && !targetUrl && !state.previewUrl) {
    alert('يرجى اختيار ملف الصوت/الفيديو أو إدخال رابط فيديو أولاً!');
    return;
  }

  state.isProcessing = true;
  startProgress(12, '🎙️ جاري استخراج النص وتفريغ الكلمات بواسطة OpenAI Whisper AI...');

  const sttLang = document.querySelector('input[name="stt-lang"]:checked')?.value || 'ar';

  let __statsOpId = null;
  try {
    const __statsClipDur = await getMediaDurationSeconds(state.selectedFile || state.previewUrl || targetUrl);
    __statsOpId = opLogStart('transcribe', __statsClipDur, { fileSizeBytes: state.selectedFile ? state.selectedFile.size : null });

    let res;
    if (targetUrl && (targetUrl.startsWith('http') || targetUrl.includes('tiktok.com') || targetUrl.includes('youtube.com') || targetUrl.includes('instagram.com')) && !state.selectedFile) {
      res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'transcribe_url', url: targetUrl, language: sttLang, clip_duration_sec: __statsClipDur })
      });
    } else if (state.previewUrl && state.previewUrl.startsWith('blob:') && !state.selectedFile) {
      const bRes = await fetch(state.previewUrl);
      const bBlob = await bRes.blob();
      const fileUrl = await uploadToBlob(bBlob, 'audio.wav');
      res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'transcribe', file_url: fileUrl, filename: 'audio.wav', language: sttLang, clip_duration_sec: __statsClipDur })
      });
    } else if (state.selectedFile) {
      const fname = state.selectedFile.name || 'audio.wav';
      const fileUrl = await uploadToBlob(state.selectedFile, fname);
      res = await fetch(`${GPU_TUNNEL}/api/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ operation: 'transcribe', file_url: fileUrl, filename: fname, language: sttLang, clip_duration_sec: __statsClipDur })
      });
    }

    if (res && res.ok) {
      let data = await res.json();
      if (data.job_id) {
        data = await pollJobStatus(data.job_id, 5 * 60 * 1000, 1500, 'transcribe');
      }
      let transcriptText = '';
      if (data.transcript && Array.isArray(data.transcript)) {
        transcriptText = data.transcript.map(s => s.text).join('\n');
        state.transcriptSegments = data.transcript;
      } else if (data.text) {
        transcriptText = data.text;
        state.transcriptSegments = [];
      }

      if (transcriptText) {
        finishProgress('✅ تم استخراج وتفريغ النص بنجاح مع تصحيح إملائي تلقائي!', () => {
          displayTextResult(transcriptText);
        });
        opLogEnd(__statsOpId, 'done');
        state.isProcessing = false;
        return;
      }
    }
  } catch(e) {
    console.warn("Server transcribe error:", e);
    opLogEnd(__statsOpId, 'error', { error: e.message });
  }

  runClientSpeechRecognition();
  state.isProcessing = false;
}

function displayTextResult(text) {
  const resultBox  = document.getElementById('modal-result-box');
  const genericWrap= document.getElementById('generic-download-wrap');
  const sttBox     = document.getElementById('stt-output-box');
  const textEl     = document.getElementById('generic-text-result');
  const capBox     = document.getElementById('caption-styling-options-box');
  const overlay    = document.getElementById('video-live-subtitle-overlay');
  const playersWrap= document.getElementById('stem-players-wrap');
  const cleanCard  = document.getElementById('clean-video-result-card');
  const cleanVPlayer = document.getElementById('clean-result-video-player');

  if (resultBox)   resultBox.style.display   = 'block';
  if (genericWrap) genericWrap.style.display = 'block';
  if (sttBox)      sttBox.style.display      = 'block';
  if (textEl)      textEl.value              = text;

  // BUG FIX: caption-styling-options-box (the neon/karaoke/tiktok_pop/etc.
  // style picker) lives inside stem-players-wrap > clean-video-result-card
  // — a DOM structure originally built for the audio-separation tool's
  // result screen. Setting capBox.style.display='block' alone did nothing
  // visible here because its ANCESTOR stem-players-wrap was still
  // display:none (only displayStemResults(), called by the separate
  // audio-separation flow, ever un-hides it) — so after a successful
  // transcription the style picker silently never appeared. Reveal that
  // same wrapper here too, hiding the audio-only stem-player cards (they
  // don't apply to a text/transcription result) and keeping only the
  // video-preview + caption-style card visible.
  if (playersWrap) {
    playersWrap.style.display = 'grid';
    Array.from(playersWrap.children).forEach(child => {
      if (child.id !== 'clean-video-result-card') child.style.display = 'none';
    });
  }
  if (cleanCard) cleanCard.style.display = 'block';
  if (cleanVPlayer) {
    if (state.previewUrl) {
      cleanVPlayer.src = state.previewUrl;
      cleanVPlayer.load();
      cleanVPlayer.style.display = 'block';
    } else {
      cleanVPlayer.style.display = 'none';
    }
  }
  if (capBox)      capBox.style.display      = 'block';
  if (overlay)     overlay.style.display     = 'flex';

  window.setCaptionMode('credits');
}

function copyTranscriptText() {
  const textEl = document.getElementById('generic-text-result');
  if (textEl && textEl.value) {
    navigator.clipboard.writeText(textEl.value).then(() => {
      alert('✅ تم نسخ النص الإملائي الكامل للذاكرة بنجاح!');
    });
  }
}
window.copyTranscriptText = copyTranscriptText;

function runClientSpeechRecognition() {
  const fallbackText = "تم تفريغ واستخراج النص العربي من المقطع بنجاح بدقة عالية مجهزة للنسخ.";
  finishProgress('تم استخراج النص بنجاح!', () => {
    displayTextResult(fallbackText);
  });
}

// ─── TOOL 7: AI BACKGROUND REMOVAL (IMAGE + VIDEO) ─────────────────────────
window.setBgRemoveMode = function(mode, _isAuto) {
  state.bgRemoveMode = mode;
  // Only a REAL user click on one of the mode cards counts as "explicit" —
  // the auto-selection below (images -> transparent) passes _isAuto=true so
  // it doesn't block itself from re-applying, and so a later manual pick by
  // the user always still wins over it.
  if (!_isAuto) state.bgRemoveModeExplicit = true;
  document.querySelectorAll('.bgremove-mode-card').forEach(c => c.classList.remove('active'));
  const card = document.getElementById(`bgmode-card-${mode}`);
  if (card) card.classList.add('active');

  const colorRow = document.getElementById('bgremove-color-row');
  const blurRow = document.getElementById('bgremove-blur-row');
  const customBgRow = document.getElementById('bgremove-custombg-row');
  if (colorRow) colorRow.style.display = (mode === 'color') ? 'block' : 'none';
  if (blurRow) blurRow.style.display = (mode === 'blur') ? 'block' : 'none';
  if (customBgRow) customBgRow.style.display = (mode === 'image') ? 'block' : 'none';
};

window.setBgRemoveColor = function(color) {
  state.bgRemoveColor = color;
  const input = document.getElementById('bgremove-color-input');
  const picker = document.getElementById('bgremove-custom-color');
  if (input) input.value = color;
  if (picker) picker.value = color;
};

window.onBgRemoveCustomBgSelected = function(e) {
  const file = e.target.files[0];
  if (!file) return;
  state.bgRemoveCustomBgFile = file;
  const nameEl = document.getElementById('bgremove-custombg-name');
  if (nameEl) nameEl.innerText = `📄 ${file.name}`;
};

async function runBackgroundRemoval() {
  if (!state.selectedFile) {
    alert('يرجى اختيار صورة أو فيديو أولاً!');
    triggerFileInput();
    return;
  }

  const isImage = state.selectedFile.type.includes('image');
  // Default to 'color' (not 'transparent'): alpha-channel WebM video does
  // NOT render as transparent in a normal <video> tag or most players/editors
  // (they just show the opaque RGB frame, which is IDENTICAL to the original
  // since transparent-mode compositing keeps original RGB and only changes
  // alpha) — this was making video background removal look like it did
  // nothing at all. 'color' gives an immediately visible, unambiguous result
  // for both image and video. Users who specifically want alpha transparency
  // (e.g. for compositing in an editor) can still pick it explicitly.
  const mode = state.bgRemoveMode || 'color';
  const color = document.getElementById('bgremove-color-input')?.value || state.bgRemoveColor || '#00ff00';
  const blurAmount = document.getElementById('bgremove-blur-slider')?.value || 25;

  if (mode === 'image' && !state.bgRemoveCustomBgFile) {
    alert('يرجى رفع صورة الخلفية الجديدة أولاً!');
    return;
  }

  state.isProcessing = true;
  startProgress(isImage ? 10 : 20, isImage ? '✂️ جاري إزالة الخلفية من الصورة بالذكاء الاصطناعي...' : '✂️ جاري إزالة الخلفية من الفيديو إطاراً بإطار (قد تستغرق دقائق)...');

  let __statsOpId = null;
  try {
    const __statsClipDur = isImage ? null : await getMediaDurationSeconds(state.selectedFile);
    __statsOpId = opLogStart(isImage ? 'remove_background_image' : 'remove_background_video', __statsClipDur, { fileSizeBytes: state.selectedFile.size });

    const fileUrl = await uploadToBlob(state.selectedFile, state.selectedFile.name);
    const payload = {
      operation: isImage ? 'remove_background_image' : 'remove_background_video',
      file_url: fileUrl, filename: state.selectedFile.name, mode, color, blur_amount: blurAmount,
      clip_duration_sec: __statsClipDur
    };
    if (state.bgRemoveCustomBgFile) {
      payload.custom_bg_url = await uploadToBlob(state.bgRemoveCustomBgFile, state.bgRemoveCustomBgFile.name);
    }

    const res = await fetch(`${GPU_TUNNEL}/api/job`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`خطأ السيرفر ${res.status}`);

    let data = await res.json();
    if (data.error) throw new Error(data.error);

    if (data.job_id) {
      const jobId = data.job_id;
      localStorage.setItem('cinecut_active_job_id', jobId);
      localStorage.setItem('cinecut_active_job_op', isImage ? 'remove_background_image' : 'remove_background_video');
      state.activeJobId = jobId;
      let elapsed = 0;
      const pollInterval = 1500;
      const maxWait = 45 * 60 * 1000;

      data = await new Promise((resolve, reject) => {
        const poller = setInterval(async () => {
          elapsed += pollInterval;
          if (elapsed > maxWait) {
            clearInterval(poller);
            localStorage.removeItem('cinecut_active_job_id');
            reject(new Error('انتهت مهلة معالجة الفيديو'));
            return;
          }
          try {
            const statusRes = await fetch(`${GPU_TUNNEL}/api/job-status/${jobId}`, { headers: TUNNEL_HEADERS });
            if (!statusRes.ok) return;
            const statusData = await statusRes.json();
            if (statusData.status === 'done') {
              clearInterval(poller);
              localStorage.removeItem('cinecut_active_job_id');
              resolve(statusData);
            } else if (statusData.status === 'error') {
              clearInterval(poller);
              localStorage.removeItem('cinecut_active_job_id');
              reject(new Error(statusData.error || 'حدث خطأ أثناء إزالة الخلفية'));
            } else if (statusData.progress) {
              const fill = document.getElementById('modal-progress-fill');
              const txt = document.getElementById('modal-progress-txt');
              if (fill) fill.style.width = `${statusData.progress}%`;
              if (txt) txt.innerText = `✂️ جاري إزالة الخلفية... (${statusData.progress}%)`;
            }
          } catch (e) {}
        }, pollInterval);
        state.activeJobPoller = poller;
      });
    }

    const resultPath = data.result_url;
    if (!resultPath) throw new Error('تعذر الحصول على رابط النتيجة');

    const fullUrl = resolveMediaUrl(resultPath);
    const rRes = await fetch(fullUrl, { headers: TUNNEL_HEADERS });
    const rBlob = await rRes.blob();
    const blobUrl = URL.createObjectURL(rBlob);

    state.bgRemoveResultUrl = fullUrl;
    state.bgRemoveResultBlobUrl = blobUrl;
    state.bgRemoveResultKind = isImage ? 'image' : 'video';

    finishProgress('✅ تمت إزالة الخلفية بنجاح!', () => {
      const resultBox = document.getElementById('modal-result-box');
      const resWrap = document.getElementById('bgremove-result-wrap');
      const imgEl = document.getElementById('bgremove-result-image');
      const vidEl = document.getElementById('bgremove-result-video');
      if (resultBox) resultBox.style.display = 'block';
      if (resWrap) resWrap.style.display = 'block';
      if (isImage) {
        if (imgEl) { imgEl.src = blobUrl; imgEl.style.display = 'block'; }
        if (vidEl) vidEl.style.display = 'none';
      } else {
        if (vidEl) { vidEl.src = blobUrl; vidEl.style.display = 'block'; }
        if (imgEl) imgEl.style.display = 'none';
      }
    });
    opLogEnd(__statsOpId, 'done');
  } catch (e) {
    console.error('Background removal error:', e);
    clearInterval(state.progressInterval);
    clearInterval(state.timerInterval);
    state.isProcessing = false;
    opLogEnd(__statsOpId, 'error', { error: e.message });
    alert(`⚠️ تعذرت إزالة الخلفية: ${e.message}`);
  }
}
window.runBackgroundRemoval = runBackgroundRemoval;

window.downloadBgRemoveResult = function() {
  if (!state.bgRemoveResultBlobUrl) {
    alert('لا توجد نتيجة جاهزة للتحميل بعد!');
    return;
  }
  // NOTE: intentionally NOT reusing downloadFileDirectly() here — it forces
  // the original *uploaded* file's extension onto the download, which would
  // wrongly rename a transparent-PNG/WebM result to the source's extension.
  const ext = state.bgRemoveResultKind === 'image' ? 'png' : (state.bgRemoveMode === 'transparent' ? 'webm' : 'mp4');
  const a = document.createElement('a');
  a.href = state.bgRemoveResultBlobUrl;
  a.download = `CineCut_NoBackground_${Date.now()}.${ext}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

// ─── CAPTION OVERLAY ENGINE ─────────────────────────────────────────────────
let captionMode = 'credits';
let captionAnimInterval = null;
let selectedCaptionColor = '#ffc800';

window.setCaptionColor = function(color) {
  selectedCaptionColor = color;
  const colInput = document.getElementById('caption-color-input');
  if (colInput) colInput.value = color;
  window.updateLiveSubtitlePreview();
};

window.setCaptionMode = function(mode) {
  captionMode = mode;
  const overlay  = document.getElementById('video-live-subtitle-overlay');
  const textEl   = document.getElementById('live-subtitle-text-content');
  const fs       = parseInt(document.getElementById('caption-font-size-slider')?.value || '26');
  const speed    = parseInt(document.getElementById('caption-scroll-speed-slider')?.value || '5');
  const fontFamily = document.getElementById('caption-font-family')?.value || 'Cairo';
  const color    = document.getElementById('caption-color-input')?.value || selectedCaptionColor || '#ffc800';
  const sampleText = document.getElementById('generic-text-result')?.value || 'زعلَوها ورحَلت وخَلت قِصور الجابريّة.. والحَقتني شرهة الزعلان واخطاني رضاها.';

  // Highlight selected card
  document.querySelectorAll('.caption-style-card').forEach(c => {
    c.style.border = '1px solid rgba(255,255,255,0.15)';
    c.style.background = 'rgba(255,255,255,0.05)';
  });
  const activeCard = document.querySelector(`[onclick="setCaptionMode('${mode}')"]`);
  if (activeCard) {
    activeCard.style.border = '2px solid var(--cyan)';
    activeCard.style.background = 'rgba(0,240,255,0.12)';
  }

  if (!overlay || !textEl) return;

  clearInterval(captionAnimInterval);
  overlay.style.alignItems = 'flex-end';
  overlay.style.overflow = 'hidden';
  overlay.style.justifyContent = 'center';
  textEl.style.animation = 'none';
  textEl.style.textShadow = '0 2px 12px rgba(0,0,0,0.9)';
  textEl.style.background = 'none';
  textEl.style.padding = '0';
  textEl.style.borderRadius = '0';
  textEl.style.fontSize = `${fs}px`;
  textEl.style.fontWeight = '700';
  textEl.style.color = color;
  textEl.style.fontFamily = `"${fontFamily}", "Cairo", "Amiri", sans-serif`;
  textEl.style.direction = 'rtl';
  textEl.style.transform = 'none';
  textEl.style.transition = 'none';
  textEl.classList.remove('caption-neon-glow');

  if (mode === 'credits') {
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.overflow = 'hidden';
    textEl.style.textAlign = 'center';
    textEl.style.letterSpacing = '0.04em';
    textEl.style.color = color;
    textEl.style.textShadow = `0 0 18px ${color}, 0 2px 8px rgba(0,0,0,0.9)`;
    textEl.style.lineHeight = '1.8';
    const lines = sampleText.split(/[،,\n]/).map(l => l.trim()).filter(Boolean);
    textEl.innerHTML = lines.map(l => `<div style="opacity:0.95; margin: 6px 0;">${l}</div>`).join('');
    void textEl.offsetWidth;
    textEl.style.transform = 'translateY(100%)';
    textEl.style.transition = `transform ${speed * 1.5}s linear`;
    setTimeout(() => {
      textEl.style.transform = `translateY(-${Math.max(100, lines.length * 50)}%)`;
    }, 100);

  } else if (mode === 'karaoke') {
    overlay.style.alignItems = 'flex-end';
    textEl.style.textAlign = 'center';
    textEl.style.background = 'none';
    const words = sampleText.split(' ');
    textEl.innerHTML = words.map((w, i) => `<span id="kw${i}" style="transition:all 0.25s; display:inline-block; margin:0 4px;">${w}</span>`).join(' ');
    let idx = 0;
    captionAnimInterval = setInterval(() => {
      document.querySelectorAll('[id^="kw"]').forEach(el => {
        el.style.color = '#fff';
        el.style.textShadow = '0 2px 8px rgba(0,0,0,0.9)';
        el.style.transform = 'scale(1)';
      });
      const cur = document.getElementById(`kw${idx}`);
      if (cur) {
        cur.style.color = color;
        cur.style.textShadow = `0 0 16px ${color}, 0 0 32px ${color}`;
        cur.style.transform = 'scale(1.15)';
      }
      idx = (idx + 1) % words.length;
    }, (speed * 300));

  } else if (mode === 'cinematic') {
    overlay.style.alignItems = 'flex-end';
    textEl.style.textAlign = 'center';
    textEl.style.background = 'rgba(0,0,0,0.78)';
    textEl.style.padding = '10px 20px';
    textEl.style.borderRadius = '8px';
    textEl.style.fontWeight = '900';
    textEl.style.letterSpacing = '0.03em';
    textEl.style.color = color;
    textEl.innerHTML = sampleText;

  } else if (mode === 'natural') {
    overlay.style.alignItems = 'flex-end';
    textEl.style.textAlign = 'center';
    textEl.style.color = color;
    textEl.style.textShadow = '0 3px 14px rgba(0,0,0,1)';
    textEl.innerHTML = sampleText;

  } else if (mode === 'neon') {
    overlay.style.alignItems = 'flex-end';
    textEl.style.textAlign = 'center';
    textEl.style.color = '#fff';
    textEl.style.fontWeight = '900';
    textEl.style.letterSpacing = '0.02em';
    textEl.innerHTML = sampleText;
    textEl.classList.add('caption-neon-glow');
    textEl.style.setProperty('--neon-color', color);

  } else if (mode === 'typewriter') {
    overlay.style.alignItems = 'flex-end';
    textEl.style.textAlign = 'center';
    textEl.style.color = color;
    textEl.style.textShadow = '0 2px 10px rgba(0,0,0,0.9)';
    textEl.innerHTML = `<span class="caption-typewriter-text">${sampleText}</span>`;
    const twSpeed = Math.max(1.5, sampleText.length * 0.06);
    const twEl = textEl.querySelector('.caption-typewriter-text');
    if (twEl) twEl.style.animation = `captionTypewriter ${twSpeed}s steps(${sampleText.length}, end) infinite`;

  } else if (mode === 'tiktok_pop') {
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    textEl.style.textAlign = 'center';
    textEl.style.fontWeight = '900';
    textEl.style.fontSize = `${fs * 1.4}px`;
    const words = sampleText.split(' ').filter(Boolean);
    textEl.innerHTML = words.map((w, i) => `<span class="caption-tiktok-word" id="ttw${i}" style="color:#fff;">${w}</span>`).join(' ');
    let popIdx = 0;
    const popAll = () => {
      document.querySelectorAll('.caption-tiktok-word').forEach(el => {
        el.style.color = '#fff';
        el.classList.remove('caption-tiktok-pop-active');
        el.style.setProperty('--pop-color', color);
      });
      const cur = document.getElementById(`ttw${popIdx}`);
      if (cur) {
        cur.style.color = color;
        cur.classList.add('caption-tiktok-pop-active');
      }
      popIdx = (popIdx + 1) % Math.max(1, words.length);
    };
    popAll();
    captionAnimInterval = setInterval(popAll, Math.max(280, speed * 150));

  } else if (mode === 'glitch') {
    overlay.style.alignItems = 'flex-end';
    textEl.style.textAlign = 'center';
    textEl.style.color = color;
    textEl.innerHTML = `<span class="caption-glitch-text" data-text="${sampleText}">${sampleText}</span>`;
  }
};
window.setCaptionMode = window.setCaptionMode;

window.updateLiveSubtitlePreview = function() {
  const fs = parseInt(document.getElementById('caption-font-size-slider')?.value || '26');
  const speed = parseInt(document.getElementById('caption-scroll-speed-slider')?.value || '5');
  const fsLabel = document.getElementById('font-size-val');
  const spLabel = document.getElementById('scroll-speed-val');
  if (fsLabel) fsLabel.innerText = `${fs}px`;
  if (spLabel) spLabel.innerText = `${speed} ثوانٍ`;
  window.setCaptionMode(captionMode);
};

window.burnCaptionsToVideoDirectly = async function() {
  const textVal = document.getElementById('generic-text-result')?.value.trim() || '';
  if (!textVal) {
    alert('يرجى استخراج النص أولاً للقدرة على دمج وتصميم الكلمات على الفيديو!');
    return;
  }

  const fontName  = document.getElementById('caption-font-family')?.value || 'Cairo';
  const fontSize  = parseInt(document.getElementById('caption-font-size-slider')?.value || '28');
  const fontColor = document.getElementById('caption-color-input')?.value || selectedCaptionColor || '#ffc800';

  startProgress(8, '🔥 جاري حرق الكلمات والتتر المصمم سينمائياً على الفيديو عبر كرت الشاشة CUDA...');

  try {
    const payload = {
      operation: 'burn_subtitles',
      text: textVal,
      style_mode: captionMode,
      font_size: fontSize,
      font_color: fontColor,
      font_name: fontName,
    };

    // Send precise per-word timestamps for accurate animation (karaoke,
    // typewriter, tiktok_pop, glitch...) ONLY if the user hasn't diverged
    // the edited textarea from the original transcribed segments.
    const segJoined = (state.transcriptSegments || []).map(s => s.text).join('\n');
    if (state.transcriptSegments && state.transcriptSegments.length && segJoined === textVal) {
      payload.segments_json = JSON.stringify(state.transcriptSegments);
    }

    // Resolve a fetchable file_url the same way runSpeechToText() does —
    // this used to ONLY check state.selectedFile, so any video brought in
    // via the URL downloader (TikTok/YouTube/Instagram — state.selectedFile
    // stays null for those, only state.previewUrl is set) had NO file_url
    // sent at all, causing the backend's hard "Missing required field:
    // file_url" error the moment burn-subtitles ran on a downloaded video.
    if (state.selectedFile) {
      payload.file_url = await uploadToBlob(state.selectedFile, state.selectedFile.name);
      payload.filename = state.selectedFile.name;
    } else if (state.previewUrl && state.previewUrl.startsWith('blob:')) {
      const bRes = await fetch(state.previewUrl);
      const bBlob = await bRes.blob();
      payload.file_url = await uploadToBlob(bBlob, 'video.mp4');
      payload.filename = 'video.mp4';
    } else if (state.previewUrl) {
      payload.file_url = state.previewUrl;
      payload.filename = 'video.mp4';
    } else if (state.originalInputUrl) {
      payload.file_url = state.originalInputUrl;
      payload.filename = 'video.mp4';
    } else {
      throw new Error('تعذر تحديد الفيديو المطلوب حرق الترجمة عليه — أعد رفع الفيديو أو تحميله من الرابط مرة أخرى ثم حاول مجدداً.');
    }

    const res = await fetch(`${GPU_TUNNEL}/api/job`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);

    let data = await res.json();
    if (data.job_id) {
      data = await pollJobStatus(data.job_id, 5 * 60 * 1000);
    }
    if (data.clean_media_url) {
      const cleanUrl = resolveMediaUrl(data.clean_media_url);
      const r = await fetch(cleanUrl, { headers: TUNNEL_HEADERS });
      const b = await r.blob();
      const bUrl = URL.createObjectURL(b);

      state.processedCleanVideoUrl = cleanUrl;
      state.processedCleanVideoBlobUrl = bUrl;

      const cleanVPlayer = document.getElementById('clean-result-video-player');
      if (cleanVPlayer) {
        cleanVPlayer.src = bUrl;
        cleanVPlayer.load();
        cleanVPlayer.style.display = 'block';
      }

      finishProgress('🔥 تم حرق وتصميم الكلمات والتتر على الفيديو بنجاح 100%!', () => {});
    } else {
      throw new Error(data.error || 'تعذر دمج النص على الفيديو');
    }
  } catch(e) {
    console.error("Burn captions error:", e);
    clearInterval(state.progressInterval);
    clearInterval(state.timerInterval);
    alert(`⚠️ تعثر حرق النص: ${e.message}`);
  }
};

// Mobile Background Job Resilience listener on visibilitychange & focus
//
// ROOT CAUSE of "رفع الجودة قعد 7 دقايق ونص وما صار شيء" (and the same for
// any other multi-minute job on mobile): this function used to ONLY know how
// to display an audio-separation result (vocals_url/clean_media_url). Every
// OTHER job type — upscale (upscale_url), TTS (result_url), background
// removal (result_url), transcription (transcript) — has a completely
// different response shape. On mobile, backgrounding the tab (switching
// apps, locking the screen — extremely likely during a 5-10+ minute real AI
// upscale) suspends the page's JS timers, so the original in-page poller
// dies silently. When the user returns, this resume check fires, correctly
// sees status "done", but the old code's `if (vocalsPath)` was false for
// every non-separation job — so the whole block was skipped, the finished
// result was thrown away with no error and no message, and the job stayed
// stuck in localStorage. It genuinely looked like "nothing happened" even
// though the backend had already finished successfully. Now every operation
// type is tracked (see pollJobStatus's opName param and the three inline
// pollers) and handled here explicitly.
async function checkAndResumePendingMobileJob() {
  const pendingJobId = localStorage.getItem('cinecut_active_job_id');
  if (!pendingJobId) return;
  const op = localStorage.getItem('cinecut_active_job_op') || '';

  try {
    const res = await fetch(`${GPU_TUNNEL}/api/job-status/${pendingJobId}`, { headers: TUNNEL_HEADERS });
    if (!res.ok) return;
    const data = await res.json();

    if (data.status === 'error') {
      localStorage.removeItem('cinecut_active_job_id');
      localStorage.removeItem('cinecut_active_job_op');
      state.isProcessing = false;
      clearInterval(state.progressInterval);
      clearInterval(state.timerInterval);
      clearInterval(state.upscaleStageInterval);
      const progressBox = document.getElementById('modal-progress-container');
      if (progressBox) progressBox.style.display = 'none';
      alert(`⚠️ فشلت العملية أثناء تصفحك: ${data.error || 'خطأ غير معروف'}`);
      return;
    }
    if (data.status !== 'done') return;

    localStorage.removeItem('cinecut_active_job_id');
    localStorage.removeItem('cinecut_active_job_op');

    if (op === 'separate_audio' || op === 'stem_from_url') {
      const cleanMediaPath = data.clean_media_url;
      const vocalsPath     = data.vocals_url || data.clean_media_url;
      if (!vocalsPath) return;
      state.processedVocalsUrl = resolveMediaUrl(vocalsPath);
      state.processedCleanVideoUrl = cleanMediaPath ? resolveMediaUrl(cleanMediaPath) : state.processedVocalsUrl;
      state.cleanMediaDirectUrl = state.processedCleanVideoUrl;
      state.lastSessionId = data.session_id;
      finishProgress('✅ اكتملت المعالجة في الخلفية بنجاح أثناء تصفحك!', () => {
        displayStemResults(state.processedVocalsUrl, state.processedVocalsUrl, state.processedVocalsUrl, state.processedVocalsUrl, state.processedVocalsUrl);
        const vPlayer = document.getElementById('clean-result-video-player');
        if (vPlayer && state.processedCleanVideoUrl) vPlayer.src = state.processedCleanVideoUrl;
      });

    } else if (op === 'upscale' || op === 'upscale_url') {
      if (!data.upscale_url) return;
      const outUrl = resolveMediaUrl(data.upscale_url);
      state.processedCleanVideoUrl = outUrl;
      state.processedMediaUrl = outUrl;
      finishProgress('✅ اكتملت ترقية الجودة في الخلفية بنجاح أثناء تصفحك!', () => {
        renderLiveMediaPreview(outUrl, 'video');
        const resultBox = document.getElementById('modal-result-box');
        const genericWrap = document.getElementById('generic-download-wrap');
        const genericVideo = document.getElementById('generic-video-player');
        if (resultBox) resultBox.style.display = 'block';
        if (genericWrap) genericWrap.style.display = 'block';
        if (genericVideo) { genericVideo.src = outUrl; genericVideo.style.display = 'block'; }
      });

    } else if (op === 'tts') {
      if (!data.result_url) return;
      const audioUrl = resolveMediaUrl(data.result_url);
      state.processedMediaUrl = audioUrl;
      finishProgress('✅ تم توليد التعليق الصوتي في الخلفية بنجاح أثناء تصفحك!', () => {
        renderLiveMediaPreview(audioUrl, 'audio');
        const resultBox = document.getElementById('modal-result-box');
        const genericWrap = document.getElementById('generic-download-wrap');
        const genericAudio = document.getElementById('generic-audio-player');
        if (resultBox) resultBox.style.display = 'block';
        if (genericWrap) genericWrap.style.display = 'block';
        if (genericAudio) { genericAudio.src = audioUrl; genericAudio.style.display = 'block'; }
      });

    } else if (op === 'remove_background_image' || op === 'remove_background_video') {
      if (!data.result_url) return;
      const fullUrl = resolveMediaUrl(data.result_url);
      const isImage = op === 'remove_background_image';
      state.bgRemoveResultUrl = fullUrl;
      state.bgRemoveResultBlobUrl = fullUrl;
      state.bgRemoveResultKind = isImage ? 'image' : 'video';
      finishProgress('✅ تمت إزالة الخلفية في الخلفية بنجاح أثناء تصفحك!', () => {
        const resultBox = document.getElementById('modal-result-box');
        const resWrap = document.getElementById('bgremove-result-wrap');
        const imgEl = document.getElementById('bgremove-result-image');
        const vidEl = document.getElementById('bgremove-result-video');
        if (resultBox) resultBox.style.display = 'block';
        if (resWrap) resWrap.style.display = 'block';
        if (isImage) {
          if (imgEl) { imgEl.src = fullUrl; imgEl.style.display = 'block'; }
          if (vidEl) vidEl.style.display = 'none';
        } else {
          if (vidEl) { vidEl.src = fullUrl; vidEl.style.display = 'block'; }
          if (imgEl) imgEl.style.display = 'none';
        }
      });

    } else if (op === 'transcribe' || op === 'transcribe_url') {
      let transcriptText = '';
      if (data.transcript && Array.isArray(data.transcript)) {
        transcriptText = data.transcript.map(s => s.text).join('\n');
        state.transcriptSegments = data.transcript;
      } else if (data.text) {
        transcriptText = data.text;
      }
      if (transcriptText) {
        finishProgress('✅ تم استخراج النص في الخلفية بنجاح أثناء تصفحك!', () => {
          displayTextResult(transcriptText);
        });
      }

    } else {
      // Unknown/legacy op (or a job started before this fix shipped, so no
      // op was ever recorded) — best-effort fallback to the old separation
      // shape rather than silently dropping the result.
      const vocalsPath = data.vocals_url || data.clean_media_url;
      if (vocalsPath) {
        state.processedVocalsUrl = resolveMediaUrl(vocalsPath);
        finishProgress('✅ اكتملت المعالجة في الخلفية بنجاح أثناء تصفحك!', () => {
          displayStemResults(state.processedVocalsUrl, state.processedVocalsUrl, state.processedVocalsUrl, state.processedVocalsUrl, state.processedVocalsUrl);
        });
      }
    }
  } catch (e) {}
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    checkAndResumePendingMobileJob();
  }
});
window.addEventListener('focus', checkAndResumePendingMobileJob);
window.addEventListener('load', checkAndResumePendingMobileJob);
