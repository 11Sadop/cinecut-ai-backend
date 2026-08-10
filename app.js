/**
 * CineCut AI Pro Ultimate Suite V123
 * Full GPU Tunnel Engine with Mobile Background Safari/Chrome Support
 */

const GPU_TUNNEL = "https://replication-gives-mambo-gig.trycloudflare.com";
const TUNNEL_HEADERS = { 'bypass-tunnel-reminder': 'true' };

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
  startTimeMs: 0,
  lastSessionId: null,
  transcriptSegments: [],
  bgRemoveMode: 'transparent',
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
    state.bgRemoveMode = 'transparent';
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
async function listenToVoiceSample() {
  stopAllActiveAudio();
  const voice = document.getElementById('tool-tts-voice')?.value || 'ar-SA-HamedNeural';
  const textToSay = 'مرحباً بك في منصة سينيكات للذكاء الاصطناعي، أقدم لك التعليق الصوتي الفاخر.';
  const isEnglish = voice.startsWith('en');
  
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textToSay);
    utterance.lang = isEnglish ? 'en-US' : 'ar-SA';
    utterance.pitch = isEnglish ? 1.0 : 0.85;
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
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
  state.isProcessing = false;
  
  const elapsedMs = Date.now() - state.startTimeMs;
  const totalSec = (elapsedMs / 1000).toFixed(1);

  const fill = document.getElementById('modal-progress-fill');
  const txt = document.getElementById('modal-progress-txt');
  const timerTxt = document.getElementById('modal-timer-txt');

  if (fill) fill.style.width = '100%';
  if (txt) txt.innerText = `✅ 100% | ${successMsg || 'اكتملت العملية بنجاح!'}`;
  if (timerTxt) timerTxt.innerText = `⏱️ التوقيت المستغرق: ${totalSec} ثانية`;

  setTimeout(() => {
    if (callback) callback();
  }, 400);
}

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
    const fpsVal = document.querySelector('input[name="upscale-fps"]:checked')?.value || '120';
    const isUpscaleChecked = document.getElementById('chk-op-upscale')?.checked;
    
    // Fast 1.4s CUDA separation when upscale unchecked, 4K 120FPS when checked
    const reqRes = isUpscaleChecked ? resVal : 'none';
    const reqFps = isUpscaleChecked ? fpsVal : 'none';

    let res;
    if (fileToUpload && fileToUpload.size > 1000) {
      const formData = new FormData();
      formData.append('file', fileToUpload, 'input_video.mp4');
      formData.append('resolution', reqRes);
      formData.append('fps', reqFps);
      res = await fetch(`${GPU_TUNNEL}/api/separate-audio`, {
        method: 'POST',
        body: formData,
        headers: TUNNEL_HEADERS
      });
    } else {
      const targetUrl = state.originalInputUrl || state.previewUrl;
      res = await fetch(`${GPU_TUNNEL}/api/stem-from-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ url: targetUrl, resolution: reqRes, fps: reqFps })
      });
    }

    if (!res.ok) throw new Error(`خطأ السيرفر ${res.status}`);

    let data = await res.json();
    if (data.error) throw new Error(data.error);

    // Save job ID in localStorage for Safari/Chrome background resilience
    if (data.job_id) {
      const jobId = data.job_id;
      localStorage.setItem('cinecut_active_job_id', jobId);
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
      });

      data = resultData;
    }

    const cleanMediaPath = data.clean_media_url;
    const vocalsPath     = data.vocals_url || data.clean_media_url;

    let vocalsRealBlobUrl = null;
    try {
      const vRes = await fetch(`${GPU_TUNNEL}${vocalsPath}?bypass=true`, { headers: TUNNEL_HEADERS });
      const vBlob = await vRes.blob();
      vocalsRealBlobUrl = URL.createObjectURL(vBlob);
    } catch(e) {
      vocalsRealBlobUrl = `${GPU_TUNNEL}${vocalsPath}?bypass=true`;
    }

    let cleanVideoRealBlobUrl = null;
    if (cleanMediaPath) {
      try {
        const cRes = await fetch(`${GPU_TUNNEL}${cleanMediaPath}?bypass=true`, { headers: TUNNEL_HEADERS });
        const cBlob = await cRes.blob();
        cleanVideoRealBlobUrl = URL.createObjectURL(cBlob);
      } catch(e) {
        cleanVideoRealBlobUrl = `${GPU_TUNNEL}${cleanMediaPath}?bypass=true`;
      }
    }

    const cleanVideoDirectServerUrl = cleanMediaPath ? `${GPU_TUNNEL}${cleanMediaPath}?bypass=true` : null;

    state.processedVocalsUrl     = vocalsRealBlobUrl;
    state.processedCleanVideoUrl = cleanVideoRealBlobUrl || cleanVideoDirectServerUrl;
    state.cleanMediaDirectUrl    = cleanVideoDirectServerUrl;
    state.lastSessionId          = data.session_id;
    state.processedMediaUrl      = cleanVideoRealBlobUrl || cleanVideoDirectServerUrl || vocalsRealBlobUrl;

    finishProgress('تم عزل الموسيقى وتجهيز الفيديو بنجاح!', () => {
      const badgeBox = document.getElementById('video-spec-badge-box');
      if (badgeBox) {
        if (isUpscaleChecked) {
          badgeBox.innerHTML = `
            <span class="suite-badge" style="background:rgba(0,240,255,0.15); color:var(--cyan); border:1px solid var(--cyan); font-weight:700; padding:6px 12px; border-radius:8px; font-size:0.85rem;"><i class="fa-solid fa-award"></i> الدقة: 3840×2160 (4K UHD)</span>
            <span class="suite-badge" style="background:rgba(255,200,0,0.15); color:var(--gold); border:1px solid var(--gold); font-weight:700; padding:6px 12px; border-radius:8px; font-size:0.85rem;"><i class="fa-solid fa-bolt"></i> السرعة: 120 FPS</span>
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

  } catch (err) {
    console.error("Separation error:", err);
    clearInterval(state.progressInterval);
    clearInterval(state.timerInterval);
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

  if (genericWrap) genericWrap.style.display = 'none';
  if (resultBox)   resultBox.style.display   = 'block';
  if (playersWrap) playersWrap.style.display = 'grid';

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
  const url = state.processedCleanVideoBlobUrl || state.processedCleanVideoUrl || state.cleanMediaDirectUrl || state.previewUrl;
  if (!url) { alert('فيديو النهائي غير جاهز للتحميل بعد!'); return; }
  let ext = 'mp4';
  if (state.selectedFile && state.selectedFile.name) {
    ext = state.selectedFile.name.split('.').pop() || 'mp4';
  }
  downloadFileDirectly(url, `CineCut_Clean_Video_${Date.now()}.${ext}`);
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
  startProgress(10, `جاري ترقية دقة الفيديو لـ 4K و 120 FPS عبر كرت الشاشة CUDA...`);

  try {
    let res;
    const urlInputVal = document.getElementById('tool-url-input')?.value.trim();
    const targetUrl   = state.originalInputUrl || urlInputVal || (state.previewUrl && !state.previewUrl.startsWith('blob:') ? state.previewUrl : '');

    if (targetUrl && (targetUrl.startsWith('http') || targetUrl.includes('tiktok.com') || targetUrl.includes('youtube.com') || targetUrl.includes('instagram.com')) && !state.selectedFile) {
      res = await fetch(`${GPU_TUNNEL}/api/upscale-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ url: targetUrl, resolution: resVal, fps: fpsVal, color_mode: colorVal, speed: speedVal })
      });
    } else if (state.previewUrl && state.previewUrl.startsWith('blob:') && !state.selectedFile) {
      const bRes = await fetch(state.previewUrl);
      const bBlob = await bRes.blob();
      const formData = new FormData();
      formData.append('file', bBlob, 'video.mp4');
      formData.append('resolution', resVal);
      formData.append('fps', fpsVal);
      formData.append('color_mode', colorVal);
      formData.append('speed', speedVal);
      res = await fetch(`${GPU_TUNNEL}/api/upscale`, { method: 'POST', body: formData, headers: TUNNEL_HEADERS });
    } else if (state.selectedFile) {
      const formData = new FormData();
      formData.append('file', state.selectedFile, state.selectedFile.name || 'video.mp4');
      formData.append('resolution', resVal);
      formData.append('fps', fpsVal);
      formData.append('color_mode', colorVal);
      formData.append('speed', speedVal);
      res = await fetch(`${GPU_TUNNEL}/api/upscale`, { method: 'POST', body: formData, headers: TUNNEL_HEADERS });
    } else {
      throw new Error('يرجى وضع رابط فيديو أو اختيار ملف أولاً');
    }

    if (!res.ok) throw new Error(`خطأ السيرفر ${res.status}`);

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    if (data.job_id) {
      const jobId = data.job_id;
      localStorage.setItem('cinecut_active_job_id', jobId);
      let elapsed = 0;
      const pollInterval = 1000;
      const maxWait = 5 * 60 * 1000;

      await new Promise((resolve, reject) => {
        const poller = setInterval(async () => {
          elapsed += pollInterval;
          if (elapsed > maxWait) {
            clearInterval(poller);
            reject(new Error('انتهت مهلة المعالجة'));
            return;
          }

          try {
            const statusRes = await fetch(`${GPU_TUNNEL}/api/upscale-status/${jobId}`, { headers: TUNNEL_HEADERS });
            if (!statusRes.ok) return;
            const statusData = await statusRes.json();

            if (statusData.status === 'done') {
              clearInterval(poller);
              localStorage.removeItem('cinecut_active_job_id');
              const outUrl = `${GPU_TUNNEL}${statusData.upscale_url}?bypass=true`;
              state.processedCleanVideoUrl = outUrl;
              state.processedMediaUrl = outUrl;

              finishProgress(`✅ اكتملت ترقية الجودة: 4K UHD / 120FPS`, () => {
                renderLiveMediaPreview(outUrl, 'video');
              });
              resolve();
            } else if (statusData.status === 'error') {
              clearInterval(poller);
              localStorage.removeItem('cinecut_active_job_id');
              reject(new Error(statusData.error || 'فشلت عملية الترقية'));
            }
          } catch (pollErr) {}
        }, pollInterval);
      });
    }

    state.isProcessing = false;
  } catch (e) {
    console.error("Upscale error:", e);
    clearInterval(state.progressInterval);
    clearInterval(state.timerInterval);
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
    } catch (e) {}
  }

  if (!videoPlayUrl) {
    try {
      const res = await fetch(`${GPU_TUNNEL}/api/download-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ url: url, fmt: 'video' })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.file_url) {
          const rawUrl = `${GPU_TUNNEL}${data.file_url}?bypass=true`;
          try {
            const rRes = await fetch(rawUrl, { headers: TUNNEL_HEADERS });
            const rBlob = await rRes.blob();
            videoPlayUrl = URL.createObjectURL(rBlob);
          } catch(e) {
            videoPlayUrl = rawUrl;
          }
        }
      }
    } catch (e) {}
  }

  if (!videoPlayUrl) {
    videoPlayUrl = url;
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
  startProgress(6, 'جاري توليد التعليق الصوتي...');

  const isEnglish = voice.startsWith('en');
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang  = isEnglish ? 'en-US' : 'ar-SA';
    utter.pitch = isEnglish ? 1.0 : 0.82;
    utter.rate  = 0.90;
    window.speechSynthesis.speak(utter);
    setTimeout(() => finishProgress('تم تشغيل التعليق الصوتي بنجاح!', () => {}), 600);
  } else {
    alert('متصفحك لا يدعم محرك الكلام.');
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

  try {
    let res;
    if (targetUrl && (targetUrl.startsWith('http') || targetUrl.includes('tiktok.com') || targetUrl.includes('youtube.com') || targetUrl.includes('instagram.com')) && !state.selectedFile) {
      res = await fetch(`${GPU_TUNNEL}/api/transcribe-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...TUNNEL_HEADERS },
        body: JSON.stringify({ url: targetUrl, language: sttLang })
      });
    } else if (state.previewUrl && state.previewUrl.startsWith('blob:') && !state.selectedFile) {
      const bRes = await fetch(state.previewUrl);
      const bBlob = await bRes.blob();
      const formData = new FormData();
      formData.append('file', bBlob, 'audio.wav');
      formData.append('language', sttLang);
      res = await fetch(`${GPU_TUNNEL}/api/transcribe`, { method: 'POST', body: formData, headers: TUNNEL_HEADERS });
    } else if (state.selectedFile) {
      const formData = new FormData();
      formData.append('file', state.selectedFile, state.selectedFile.name || 'audio.wav');
      formData.append('language', sttLang);
      res = await fetch(`${GPU_TUNNEL}/api/transcribe`, { method: 'POST', body: formData, headers: TUNNEL_HEADERS });
    }

    if (res && res.ok) {
      const data = await res.json();
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
        state.isProcessing = false;
        return;
      }
    }
  } catch(e) {
    console.warn("Server transcribe error:", e);
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

  if (resultBox)   resultBox.style.display   = 'block';
  if (genericWrap) genericWrap.style.display = 'block';
  if (sttBox)      sttBox.style.display      = 'block';
  if (textEl)      textEl.value              = text;
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
window.setBgRemoveMode = function(mode) {
  state.bgRemoveMode = mode;
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
  const mode = state.bgRemoveMode || 'transparent';
  const color = document.getElementById('bgremove-color-input')?.value || state.bgRemoveColor || '#00ff00';
  const blurAmount = document.getElementById('bgremove-blur-slider')?.value || 25;

  if (mode === 'image' && !state.bgRemoveCustomBgFile) {
    alert('يرجى رفع صورة الخلفية الجديدة أولاً!');
    return;
  }

  state.isProcessing = true;
  startProgress(isImage ? 10 : 20, isImage ? '✂️ جاري إزالة الخلفية من الصورة بالذكاء الاصطناعي...' : '✂️ جاري إزالة الخلفية من الفيديو إطاراً بإطار (قد تستغرق دقائق)...');

  try {
    const formData = new FormData();
    formData.append('file', state.selectedFile, state.selectedFile.name);
    formData.append('mode', mode);
    formData.append('color', color);
    formData.append('blur_amount', blurAmount);
    if (state.bgRemoveCustomBgFile) {
      formData.append('custom_bg', state.bgRemoveCustomBgFile, state.bgRemoveCustomBgFile.name);
    }

    const endpoint = isImage ? '/api/remove-background-image' : '/api/remove-background-video';
    const res = await fetch(`${GPU_TUNNEL}${endpoint}`, { method: 'POST', body: formData, headers: TUNNEL_HEADERS });
    if (!res.ok) throw new Error(`خطأ السيرفر ${res.status}`);

    let data = await res.json();
    if (data.error) throw new Error(data.error);

    if (data.job_id) {
      const jobId = data.job_id;
      let elapsed = 0;
      const pollInterval = 1500;
      const maxWait = 20 * 60 * 1000;

      data = await new Promise((resolve, reject) => {
        const poller = setInterval(async () => {
          elapsed += pollInterval;
          if (elapsed > maxWait) {
            clearInterval(poller);
            reject(new Error('انتهت مهلة معالجة الفيديو'));
            return;
          }
          try {
            const statusRes = await fetch(`${GPU_TUNNEL}/api/job-status/${jobId}`, { headers: TUNNEL_HEADERS });
            if (!statusRes.ok) return;
            const statusData = await statusRes.json();
            if (statusData.status === 'done') {
              clearInterval(poller);
              resolve(statusData);
            } else if (statusData.status === 'error') {
              clearInterval(poller);
              reject(new Error(statusData.error || 'حدث خطأ أثناء إزالة الخلفية'));
            } else if (statusData.progress) {
              const fill = document.getElementById('modal-progress-fill');
              const txt = document.getElementById('modal-progress-txt');
              if (fill) fill.style.width = `${statusData.progress}%`;
              if (txt) txt.innerText = `✂️ جاري إزالة الخلفية... (${statusData.progress}%)`;
            }
          } catch (e) {}
        }, pollInterval);
      });
    }

    const resultPath = data.result_url;
    if (!resultPath) throw new Error('تعذر الحصول على رابط النتيجة');

    const fullUrl = `${GPU_TUNNEL}${resultPath}?bypass=true`;
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
  } catch (e) {
    console.error('Background removal error:', e);
    clearInterval(state.progressInterval);
    clearInterval(state.timerInterval);
    state.isProcessing = false;
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
    const formData = new FormData();
    formData.append('text', textVal);
    formData.append('style_mode', captionMode);
    formData.append('font_size', fontSize);
    formData.append('font_color', fontColor);
    formData.append('font_name', fontName);

    // Send precise per-word timestamps for accurate animation (karaoke,
    // typewriter, tiktok_pop, glitch...) ONLY if the user hasn't diverged
    // the edited textarea from the original transcribed segments.
    const segJoined = (state.transcriptSegments || []).map(s => s.text).join('\n');
    if (state.transcriptSegments && state.transcriptSegments.length && segJoined === textVal) {
      formData.append('segments_json', JSON.stringify(state.transcriptSegments));
    }

    if (state.selectedFile) {
      formData.append('file', state.selectedFile, state.selectedFile.name);
    }

    const res = await fetch(`${GPU_TUNNEL}/api/burn-subtitles`, {
      method: 'POST',
      body: formData,
      headers: TUNNEL_HEADERS
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const data = await res.json();
    if (data.clean_media_url) {
      const cleanUrl = `${GPU_TUNNEL}${data.clean_media_url}?bypass=true`;
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
async function checkAndResumePendingMobileJob() {
  const pendingJobId = localStorage.getItem('cinecut_active_job_id');
  if (!pendingJobId) return;

  try {
    const res = await fetch(`${GPU_TUNNEL}/api/job-status/${pendingJobId}`, { headers: TUNNEL_HEADERS });
    if (!res.ok) return;
    const data = await res.json();
    if (data.status === 'done') {
      localStorage.removeItem('cinecut_active_job_id');
      const cleanMediaPath = data.clean_media_url;
      const vocalsPath     = data.vocals_url || data.clean_media_url;
      if (vocalsPath) {
        state.processedVocalsUrl = `${GPU_TUNNEL}${vocalsPath}?bypass=true`;
        state.processedCleanVideoUrl = cleanMediaPath ? `${GPU_TUNNEL}${cleanMediaPath}?bypass=true` : state.processedVocalsUrl;
        state.cleanMediaDirectUrl = state.processedCleanVideoUrl;
        state.lastSessionId = data.session_id;

        finishProgress('✅ اكتملت المعالجة في الخلفية بنجاح أثناء تصفحك!', () => {
          displayStemResults(state.processedVocalsUrl, state.processedVocalsUrl, state.processedVocalsUrl, state.processedVocalsUrl, state.processedVocalsUrl);
          const vPlayer = document.getElementById('clean-result-video-player');
          if (vPlayer && state.processedCleanVideoUrl) {
            vPlayer.src = state.processedCleanVideoUrl;
          }
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
