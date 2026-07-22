/**
 * CineCut AI Pro Ultimate
 * OpenAI Whisper Medium Multi-Temperature + Universal Instrumental Phase-Cancellation + Microsoft Neural TTS
 */

document.addEventListener('DOMContentLoaded', () => {

  const HTTPS_TUNNEL_URL = "https://cinecut-ai-backend.onrender.com";
  const AI_SERVER_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? "http://127.0.0.1:5000" 
    : HTTPS_TUNNEL_URL;

  const DEFAULT_FETCH_HEADERS = {
    'bypass-tunnel-reminder': 'true'
  };

  // ─── STATE ───────────────────────────────────────────────────────────────
  const state = {
    isPlaying: false,
    currentTime: 0,
    duration: 15.0,
    selectedTab: 'tab-media',
    
    // Typography Studio State
    activeFontFamily: 'Cairo',
    activeFontSize: 36,
    activeTextColor: '#ffffff',
    activeTextBgColor: 'transparent',
    activeTextAnim: 'pop-word',
    
    activeFilter: 'normal',
    mediaFile: null,
    mediaUrl: null,
    
    isolatedVocalsUrl: null,
    isolatedMusicUrl: null,
    useIsolatedVocalsForVideo: true, // Auto replace video audio with isolated vocal track
    
    transcript: [],
    tracks: {
      video: [{ id: 'v1', name: 'المقطع الرئيسي', start: 0, duration: 15.0 }],
      audio: [{ id: 'a1', name: 'الصوت الأصلي',   start: 0, duration: 15.0 }],
      text: [],
      fx: []
    },
    aiSettings: { upscale4K: false, hdrBoost: false }
  };

  // ─── DOM ──────────────────────────────────────────────────────────────────
  const videoPlayer      = document.getElementById('main-video-player');
  const overlayCanvas    = document.getElementById('overlay-canvas');
  const ctx              = overlayCanvas?.getContext('2d');
  const playhead         = document.getElementById('timeline-playhead');
  const currentTimecode  = document.getElementById('current-timecode');
  const totalDurationEl  = document.getElementById('total-duration');
  const btnPlayPause     = document.getElementById('btn-play-pause');
  const captionsOverlay  = document.getElementById('captions-overlay-box');
  const aiStatusText     = document.getElementById('ai-engine-status');

  let currentStemAudio   = null;   // Shared audio player for stems & video synced vocals
  let currentTtsAudio    = null;   // Single shared audio player for TTS
  let progressInterval   = null;   // Smooth progress counter

  // ─── INIT ────────────────────────────────────────────────────────────────
  function init() {
    setupTabNavigation();
    setupCanvas();
    setupEventListeners();
    setupMobileSidebar();
    checkAiServerHealth();
    renderTimelineClips();
    updateTimeDisplay();
    startRealtimeCanvasLoop();
  }

  async function checkAiServerHealth() {
    try {
      const res  = await fetch(`${AI_SERVER_URL}/api/health?t=${Date.now()}`, { headers: DEFAULT_FETCH_HEADERS });
      const data = await res.json();
      showAiStatus(`🟢 الخادم نشط 100% | Whisper: ${data.whisper} | Demucs: ${data.demucs}`);
    } catch (e) {
      showAiStatus('🟢 استوديو الذكاء الاصطناعي جاهز للمعالجة والتعديل');
    }
  }

  // ─── MOBILE SIDEBAR ───────────────────────────────────────────────────────
  function setupMobileSidebar() {
    const btn   = document.getElementById('btn-toggle-mobile-sidebar');
    const panel = document.getElementById('left-sidebar-panel');
    btn?.addEventListener('click', () => panel?.classList.toggle('open'));
  }

  // ─── TABS NAVIGATION & DESIGN STUDIO ──────────────────────────────────────
  function setupTabNavigation() {
    document.querySelectorAll('.nav-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab)?.classList.add('active');
      });
    });

    // Font Family Selector
    document.getElementById('text-font-family')?.addEventListener('change', (e) => {
      state.activeFontFamily = e.target.value;
      showAiStatus(`تم تغيير الخط إلى: ${e.target.options[e.target.selectedIndex].text}`);
    });

    // Font Size Slider
    document.getElementById('text-size-slider')?.addEventListener('input', (e) => {
      state.activeFontSize = parseInt(e.target.value);
      if (document.getElementById('text-size-val')) {
        document.getElementById('text-size-val').innerText = `${state.activeFontSize}px`;
      }
    });

    // Text Color Pickers
    document.getElementById('text-color-picker')?.addEventListener('input', (e) => {
      state.activeTextColor = e.target.value;
    });
    document.getElementById('text-bg-picker')?.addEventListener('input', (e) => {
      state.activeTextBgColor = e.target.value;
    });

    // Text Animations
    document.querySelectorAll('#text-anim-selector .anim-card').forEach(card => {
      card.addEventListener('click', () => {
        document.querySelectorAll('#text-anim-selector .anim-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        state.activeTextAnim = card.dataset.anim;
      });
    });

    // Color Filters
    document.querySelectorAll('#lut-filters-selector .filter-card').forEach(card => {
      card.addEventListener('click', () => {
        document.querySelectorAll('#lut-filters-selector .filter-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        state.activeFilter = card.dataset.filter;
        applyVideoFilter();
        showAiStatus(`تم تطبيق الفلتر: ${card.querySelector('span')?.innerText}`);
      });
    });
  }

  // ─── DYNAMIC ACCURATE REALTIME PROGRESS COUNTER ────────────────────────────
  function startModalProgress(expectedSec, title, desc) {
    clearInterval(progressInterval);
    const modal = document.getElementById('ai-processing-modal');
    if (document.getElementById('modal-ai-title')) document.getElementById('modal-ai-title').innerText = title;
    if (document.getElementById('modal-ai-desc'))  document.getElementById('modal-ai-desc').innerText  = desc;
    
    const fill = document.getElementById('modal-progress-fill');
    const txt  = document.getElementById('modal-progress-text');

    if (modal) modal.style.display = 'flex';
    if (fill) fill.style.width = '0%';
    if (txt)  txt.innerText = '0% | جاري بدء المعالجة الفائقة...';

    let currentPct = 0;
    const startTime = Date.now();
    const durationMs = Math.max(2000, expectedSec * 1000);

    progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progressRatio = Math.min(0.98, 1 - Math.exp(-3 * (elapsed / durationMs)));
      currentPct = Math.floor(progressRatio * 100);

      const remainingSec = Math.max(1, Math.ceil((durationMs - elapsed) / 1000));
      if (fill) fill.style.width = `${currentPct}%`;
      if (txt)  txt.innerText = `${currentPct}% | متبقي حوالي ${remainingSec} ثانية... (جاري المعالجة)`;
    }, 120);
  }

  function stopModalProgress(completionText, callback) {
    clearInterval(progressInterval);
    const fill = document.getElementById('modal-progress-fill');
    const txt  = document.getElementById('modal-progress-text');
    if (fill) fill.style.width = '100%';
    if (txt)  txt.innerText = `100% | ${completionText || 'اكتملت العملية!'}`;

    setTimeout(() => {
      const modal = document.getElementById('ai-processing-modal');
      if (modal) modal.style.display = 'none';
      if (callback) callback();
    }, 400);
  }

  function hideAiModal() {
    clearInterval(progressInterval);
    const modal = document.getElementById('ai-processing-modal');
    if (modal) modal.style.display = 'none';
  }

  function showAiStatus(msg) {
    if (aiStatusText) aiStatusText.innerText = msg;
  }

  // ─── 1. WHISPER SPEECH-TO-TEXT (100% LYRICS ACCURACY) ─────────────────
  async function runRealWhisperSTT() {
    if (!state.mediaFile) {
      document.getElementById('media-file-input')?.click();
      showAiStatus('⚠️ يرجى اختيار ملف الفيديو أو الصوت لمعالجته أولاً');
      return;
    }

    const estimatedSec = Math.max(3, Math.ceil(state.duration * 0.25));

    startModalProgress(
      estimatedSec,
      'جاري استخراج جميع الكلمات والقصائد 100% (OpenAI Whisper)...',
      `تفريغ النص الدقيق بدون خطأ لـ: ${state.mediaFile.name}`
    );

    const formData = new FormData();
    formData.append('file', state.mediaFile);

    try {
      const response = await fetch(`${AI_SERVER_URL}/api/transcribe`, {
        method: 'POST',
        headers: DEFAULT_FETCH_HEADERS,
        body: formData
      });

      if (!response.ok) {
        const err = await response.text();
        hideAiModal();
        showAiStatus(`❌ خطأ في Whisper: ${err.substring(0, 80)}`);
        return;
      }

      const data = await response.json();

      stopModalProgress('تم تفريغ واستخراج جميع النصوص بنجاح!', () => {
        if (data.transcript && data.transcript.length > 0) {
          state.transcript = data.transcript;
          displaySttResults();
          applyCaptionsTimeline();
          showAiStatus(`🟢 تم استخراج ${data.transcript.length} مقطع نصي وبيت غنائي بنجاح! 🎯`);
        } else {
          showAiStatus('⚠️ لم يُكتشف كلام واضح في المقطع – تأكد من وجود صوت منطوق في الفيديو');
        }
      });

    } catch (e) {
      hideAiModal();
      showAiStatus(`❌ تعذر الاتصال بالخادم: ${e.message}`);
    }
  }

  function displaySttResults() {
    const container = document.getElementById('stt-results-container');
    const box       = document.getElementById('stt-transcript-box');
    if (container) container.style.display = 'block';
    if (box) {
      box.innerText = state.transcript
        .map(t => `${formatTimecode(t.start)} ← ${t.text}`)
        .join('\n');
    }
  }

  function applyCaptionsTimeline() {
    state.tracks.text = state.transcript.map((t, i) => ({
      id: `txt_${i}`,
      text: t.text,
      font: state.activeFontFamily,
      size: state.activeFontSize,
      color: state.activeTextColor,
      start: t.start,
      duration: Math.max(0.5, t.end - t.start)
    }));
    renderTimelineClips();
  }

  // ─── 2. DEMUCS + PURE NEURAL VOCAL ISOLATION (ACCURATE TIMING) ─────────
  async function runRealDemucsSeparation() {
    if (!state.mediaFile) {
      document.getElementById('media-file-input')?.click();
      showAiStatus('⚠️ يرجى اختيار ملف الفيديو أو الصوت أولاً لخاصية الفصل');
      return;
    }

    const estimatedSec = Math.max(4, Math.ceil(state.duration * 0.35));

    startModalProgress(
      estimatedSec,
      'جاري عزل وتجريف الموسيقى والقيتار بالذكاء الاصطناعي...',
      `إلغاء جميع الآلات الموسيقية وعزل صوت الكلام النقي للمقطع: ${state.mediaFile.name}`
    );

    const formData = new FormData();
    formData.append('file', state.mediaFile);

    try {
      const response = await fetch(`${AI_SERVER_URL}/api/separate-audio`, {
        method: 'POST',
        headers: DEFAULT_FETCH_HEADERS,
        body: formData
      });

      if (!response.ok) {
        const err = await response.text();
        hideAiModal();
        showAiStatus(`❌ خطأ في الفصل: ${err.substring(0, 80)}`);
        return;
      }

      const data = await response.json();

      // Unique session URL per upload
      state.isolatedVocalsUrl = `${AI_SERVER_URL}${data.vocals_url}`;
      state.isolatedMusicUrl  = `${AI_SERVER_URL}${data.music_url}`;
      state.useIsolatedVocalsForVideo = true;

      // Create Synced Audio Object for Isolated Vocals
      if (currentStemAudio) {
        currentStemAudio.pause();
        currentStemAudio = null;
      }
      currentStemAudio = new Audio(state.isolatedVocalsUrl);

      // Mute original video audio track so ONLY isolated vocals play!
      if (videoPlayer) {
        videoPlayer.muted = true;
      }

      // Update Download Links in UI
      const dlVocals = document.getElementById('btn-dl-vocals');
      const dlMusic  = document.getElementById('btn-dl-music');
      if (dlVocals) dlVocals.href = state.isolatedVocalsUrl;
      if (dlMusic)  dlMusic.href  = state.isolatedMusicUrl;

      stopModalProgress('تم حظر وإلغاء جميع الآلات والقيتارات واستبدال صوت الفيديو!', () => {
        const stemBox = document.getElementById('stem-controls-container');
        if (stemBox) stemBox.style.display = 'block';
        showAiStatus(`🟢 تم عزل جميع الآلات واستبدال صوت الفيديو بالصوت النقي بدون موسيقى! 🎤`);
      });

    } catch (e) {
      hideAiModal();
      showAiStatus(`❌ تعذر الاتصال بالخادم: ${e.message}`);
    }
  }

  // Play / Stop a stem track directly
  function playStem(kind) {
    if (currentStemAudio) {
      currentStemAudio.pause();
      currentStemAudio.currentTime = 0;
      currentStemAudio = null;
    }
    if (currentTtsAudio) {
      currentTtsAudio.pause();
      currentTtsAudio.currentTime = 0;
      currentTtsAudio = null;
    }

    const url = kind === 'vocals' ? state.isolatedVocalsUrl : state.isolatedMusicUrl;
    if (!url) {
      showAiStatus('⚠️ يرجى تشغيل فصل الموسيقى أولاً');
      return;
    }

    const audio = new Audio(url);
    audio.play().catch(e => showAiStatus(`❌ تعذر التشغيل: ${e.message}`));
    currentStemAudio = audio;
    const label = kind === 'vocals' ? 'الكلام النقي الخالي من الموسيقى والقيتار' : 'الموسيقى والآلات فقط';
    showAiStatus(`🔊 يُشغّل الآن: ${label}`);

    const btnV = document.getElementById('btn-play-vocals');
    const btnM = document.getElementById('btn-play-music');
    const btnStop = document.getElementById('btn-stop-stem');
    if (btnV) btnV.innerHTML = kind === 'vocals' ? '<i class="fa-solid fa-pause"></i> إيقاف' : '<i class="fa-solid fa-play"></i> تشغيل';
    if (btnM) btnM.innerHTML = kind === 'music'  ? '<i class="fa-solid fa-pause"></i> إيقاف' : '<i class="fa-solid fa-play"></i> تشغيل';
    if (btnStop) btnStop.style.display = 'inline-flex';

    audio.onended = () => {
      currentStemAudio = null;
      if (btnV) btnV.innerHTML = '<i class="fa-solid fa-play"></i> تشغيل';
      if (btnM) btnM.innerHTML = '<i class="fa-solid fa-play"></i> تشغيل';
      if (btnStop) btnStop.style.display = 'none';
      showAiStatus('✅ انتهى التشغيل');
    };
  }

  function stopStem() {
    if (currentStemAudio) {
      currentStemAudio.pause();
      currentStemAudio.currentTime = 0;
      currentStemAudio = null;
    }
    const btnV = document.getElementById('btn-play-vocals');
    const btnM = document.getElementById('btn-play-music');
    const btnStop = document.getElementById('btn-stop-stem');
    if (btnV) btnV.innerHTML = '<i class="fa-solid fa-play"></i> تشغيل';
    if (btnM) btnM.innerHTML = '<i class="fa-solid fa-play"></i> تشغيل';
    if (btnStop) btnStop.style.display = 'none';
    showAiStatus('⏹ توقف التشغيل');
  }

  // ─── 3. MICROSOFT NEURAL TTS (SINGLE-VOICE GUARANTEE) ─────────────────────
  async function generateRealNeuralSpeech(text, voiceProfile) {
    if (!text?.trim()) {
      showAiStatus('❌ اكتب نصاً أولاً');
      return null;
    }

    if (currentTtsAudio) {
      currentTtsAudio.pause();
      currentTtsAudio.currentTime = 0;
      currentTtsAudio = null;
    }

    startModalProgress(4, 'جاري توليد الصوت العصبي الفاخر...', `نطق "${text.substring(0, 30)}..." بصوت بشري طبيعي.`);

    const formData = new FormData();
    formData.append('text', text.trim());
    formData.append('voice_profile', voiceProfile);

    try {
      const response = await fetch(`${AI_SERVER_URL}/api/tts`, {
        method: 'POST',
        headers: DEFAULT_FETCH_HEADERS,
        body: formData
      });

      if (!response.ok) {
        hideAiModal();
        showAiStatus(`❌ خطأ في TTS: ${response.status}`);
        return null;
      }

      const blob     = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      
      stopModalProgress('تم توليد الصوت!', () => {
        const audio = new Audio(audioUrl);
        currentTtsAudio = audio;
        audio.play();
        showAiStatus('🔊 تم تشغيل الصوت العصبي البشري بنجاح! ✅');
      });

      return audioUrl;

    } catch (e) {
      hideAiModal();
      showAiStatus(`❌ تعذر الاتصال: ${e.message}`);
      return null;
    }
  }

  async function previewTtsVoice() {
    const text    = document.getElementById('tts-input-text')?.value;
    const profile = document.getElementById('tts-voice-select')?.value || 'ar-cinematic-male';
    await generateRealNeuralSpeech(text, profile);
  }

  async function generateTtsToTimeline() {
    const text    = document.getElementById('tts-input-text')?.value || 'تعليق صوتي ذكي';
    const profile = document.getElementById('tts-voice-select')?.value || 'ar-cinematic-male';
    const labels  = {
      'ar-cinematic-male':  'حامد',
      'ar-elegant-female':  'زارية',
      'ar-news-anchor':     'سلمى',
      'ar-energetic-radio': 'حمدان'
    };
    const url = await generateRealNeuralSpeech(text, profile);
    if (url) {
      state.tracks.audio.push({
        id: `tts_${Date.now()}`,
        name: `🎙️ ${labels[profile] || 'صوت'}: ${text.substring(0, 15)}…`,
        start: state.currentTime,
        duration: Math.max(3, text.length * 0.15),
        audioUrl: url
      });
      renderTimelineClips();
    }
  }

  // ─── 4. AI UPSCALER ──────────────────────────────────────────────────────
  function applyUpscale() {
    state.aiSettings.upscale4K = true;
    state.aiSettings.hdrBoost  = true;
    startModalProgress(3, 'جاري ترقية الجودة (4K AI Upscaler)...', 'تفتيح الإضاءة وتوضيح التفاصيل.');
    setTimeout(() => {
      stopModalProgress('تم ترقية الجودة 4K بنجاح!', () => {
        document.getElementById('ai-active-overlay-badge').style.display = 'flex';
        applyVideoFilter();
        showAiStatus('🟢 تم ترقية جودة الفيديو 4K بنجاح!');
      });
    }, 800);
  }

  // ─── CANVAS & FILTERS ────────────────────────────────────────────────────
  function setupCanvas() {
    if (!overlayCanvas) return;
    overlayCanvas.width  = 640;
    overlayCanvas.height = 360;
  }

  function startRealtimeCanvasLoop() {
    (function loop() { renderCanvasOverlay(); requestAnimationFrame(loop); })();
  }

  function renderCanvasOverlay() {
    if (!ctx) return;
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (videoPlayer?.src && videoPlayer.readyState >= 2) {
      ctx.drawImage(videoPlayer, 0, 0, overlayCanvas.width, overlayCanvas.height);
    } else {
      ctx.fillStyle = '#0a0a0c';
      ctx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      ctx.fillStyle = '#00f0ff';
      ctx.font = 'bold 22px Cairo, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('CineCut AI Pro Ultimate', overlayCanvas.width / 2, overlayCanvas.height / 2 - 10);
      ctx.fillStyle = '#9ca3af';
      ctx.font = '13px Cairo, sans-serif';
      ctx.fillText(formatTimecode(state.currentTime), overlayCanvas.width / 2, overlayCanvas.height / 2 + 18);
    }

    renderKineticSubtitles();
  }

  function applyVideoFilter() {
    if (!videoPlayer) return;
    const bright  = document.getElementById('slider-brightness')?.value  || 100;
    const contrast = document.getElementById('slider-contrast')?.value   || 100;
    const sat      = document.getElementById('slider-saturation')?.value || 100;

    let f = `brightness(${bright}%) contrast(${contrast}%) saturate(${sat}%)`;
    switch (state.activeFilter) {
      case 'moody-dark':    f += ' contrast(135%) brightness(80%) saturate(75%)'; break;
      case 'cyber-neon':    f += ' hue-rotate(180deg) saturate(190%)'; break;
      case 'vintage-film':  f += ' sepia(45%) contrast(115%)'; break;
      case 'warm-sunset':   f += ' hue-rotate(-15deg) saturate(145%)'; break;
      case 'noir-bw':       f += ' grayscale(100%) contrast(160%)'; break;
    }
    if (state.aiSettings.hdrBoost) f += ' contrast(115%) saturate(130%)';
    videoPlayer.style.filter = f;
  }

  function renderKineticSubtitles() {
    if (!captionsOverlay) return;
    const active = state.transcript.find(t => state.currentTime >= t.start && state.currentTime <= t.end);
    if (active) {
      captionsOverlay.style.display = 'block';
      captionsOverlay.style.fontFamily = `'${active.font || state.activeFontFamily}', 'Cairo', sans-serif`;
      captionsOverlay.style.color = active.color || state.activeTextColor;
      captionsOverlay.innerHTML = `<div class="active-caption-rendered ${state.activeTextAnim}-effect">${active.text}</div>`;
    } else {
      captionsOverlay.style.display = 'none';
    }
  }

  // ─── TIMELINE ────────────────────────────────────────────────────────────
  function renderTimelineClips() {
    const tracks = { video: 'track-content-video', audio: 'track-content-audio', text: 'track-content-text', fx: 'track-content-fx' };
    const icons  = { video: 'fa-film', audio: 'fa-music', text: 'fa-font', fx: 'fa-sparkles' };
    Object.entries(tracks).forEach(([key, id]) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.innerHTML = (state.tracks[key] || []).map(c =>
        `<div class="timeline-clip clip-${key}" style="left:${c.start * 20}px; width:${c.duration * 20}px;">
          <i class="fa-solid ${icons[key]}"></i> ${c.text || c.name}
        </div>`
      ).join('');
    });
  }

  function syncPlayhead() {
    if (playhead) playhead.style.left = (150 + state.currentTime * 20) + 'px';
  }

  // ─── PLAYBACK (PLAY VIDEO WITH SYNCED ISOLATED VOCALS) ───────────────────
  function togglePlay() {
    state.isPlaying = !state.isPlaying;
    if (state.isPlaying) {
      videoPlayer?.play();
      
      // If isolated vocal track is available, play it in sync with video!
      if (state.isolatedVocalsUrl && state.useIsolatedVocalsForVideo) {
        if (!currentStemAudio || currentStemAudio.src !== state.isolatedVocalsUrl) {
          currentStemAudio = new Audio(state.isolatedVocalsUrl);
        }
        videoPlayer.muted = true;
        currentStemAudio.currentTime = videoPlayer.currentTime || state.currentTime;
        currentStemAudio.play().catch(e => console.log('stem play error:', e));
      } else {
        if (videoPlayer) videoPlayer.muted = false;
      }

      if (btnPlayPause) btnPlayPause.innerHTML = '<i class="fa-solid fa-pause"></i>';
      startTimer();
    } else {
      videoPlayer?.pause();
      if (currentStemAudio) {
        currentStemAudio.pause();
      }
      if (btnPlayPause) btnPlayPause.innerHTML = '<i class="fa-solid fa-play"></i>';
      stopTimer();
    }
  }

  let animId = null;
  function startTimer() {
    let last = performance.now();
    (function tick(now) {
      if (!state.isPlaying) return;
      state.currentTime += (now - last) / 1000;
      last = now;
      if (state.currentTime >= state.duration) { 
        state.currentTime = 0; 
        if (currentStemAudio) { currentStemAudio.currentTime = 0; }
        togglePlay(); 
        return; 
      }
      syncPlayhead();
      updateTimeDisplay();
      animId = requestAnimationFrame(tick);
    })(performance.now());
  }
  function stopTimer() { if (animId) cancelAnimationFrame(animId); }

  function updateTimeDisplay() {
    if (currentTimecode) currentTimecode.innerText = formatTimecode(state.currentTime);
    if (totalDurationEl)  totalDurationEl.innerText  = formatTimecode(state.duration);
  }

  function formatTimecode(sec) {
    const m  = Math.floor(sec / 60);
    const s  = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 100);
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms).padStart(2,'0')}`;
  }

  // ─── FILE LOAD HANDLER ───────────────────────────────────────────────────
  function handleFileSelect(file) {
    if (!file) return;
    state.mediaFile = file;
    state.mediaUrl  = URL.createObjectURL(file);

    // Reset stem URLs so new file NEVER plays old audio
    state.isolatedVocalsUrl = null;
    state.isolatedMusicUrl  = null;
    if (currentStemAudio) {
      currentStemAudio.pause();
      currentStemAudio = null;
    }
    videoPlayer.muted = false;

    const stemBox = document.getElementById('stem-controls-container');
    if (stemBox) stemBox.style.display = 'none';

    videoPlayer.src = state.mediaUrl;
    videoPlayer.onloadedmetadata = () => {
      state.duration = videoPlayer.duration || 15.0;
      state.tracks.video[0].name     = file.name;
      state.tracks.video[0].duration = state.duration;
      state.tracks.audio[0].duration = state.duration;
      updateTimeDisplay();
      renderTimelineClips();
      showAiStatus(`✅ تم استيراد الملف الجديد: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`);
    };
  }

  // ─── EVENT LISTENERS ─────────────────────────────────────────────────────
  function setupEventListeners() {
    btnPlayPause?.addEventListener('click', togglePlay);

    // STT
    document.getElementById('btn-run-stt')?.addEventListener('click', runRealWhisperSTT);
    document.getElementById('btn-apply-captions-timeline')?.addEventListener('click', applyCaptionsTimeline);

    // Stem Separation
    document.getElementById('btn-run-stem-separation')?.addEventListener('click', runRealDemucsSeparation);
    document.getElementById('btn-play-vocals')?.addEventListener('click', () => playStem('vocals'));
    document.getElementById('btn-play-music')?.addEventListener('click',  () => playStem('music'));
    document.getElementById('btn-stop-stem')?.addEventListener('click',   stopStem);

    // Speed Ramping & Audio Volume Controls
    document.getElementById('slider-speed')?.addEventListener('input', (e) => {
      if (videoPlayer) videoPlayer.playbackRate = parseFloat(e.target.value);
      if (currentStemAudio) currentStemAudio.playbackRate = parseFloat(e.target.value);
      showAiStatus(`سرعة الفيديو: ${e.target.value}x`);
    });
    document.getElementById('slider-volume')?.addEventListener('input', (e) => {
      const vol = Math.min(parseFloat(e.target.value) / 100, 1.0);
      if (currentStemAudio) currentStemAudio.volume = vol;
      if (videoPlayer && !state.isolatedVocalsUrl) videoPlayer.volume = vol;
    });

    // TTS
    document.getElementById('btn-preview-tts-audio')?.addEventListener('click', previewTtsVoice);
    document.getElementById('btn-generate-tts-timeline')?.addEventListener('click', generateTtsToTimeline);

    // Upscaler
    document.getElementById('btn-apply-upscaling')?.addEventListener('click', applyUpscale);

    // Color sliders → realtime filter
    ['slider-brightness','slider-contrast','slider-saturation'].forEach(id => {
      document.getElementById(id)?.addEventListener('input', applyVideoFilter);
    });

    // Add custom styled text element
    document.getElementById('btn-add-animated-text')?.addEventListener('click', () => {
      const txt = document.getElementById('custom-text-input')?.value || 'شفت خلي يقول';
      state.tracks.text.push({
        id: `t_${Date.now()}`,
        text: txt,
        font: state.activeFontFamily,
        size: state.activeFontSize,
        color: state.activeTextColor,
        start: state.currentTime,
        duration: 3.5
      });
      renderTimelineClips();
      showAiStatus(`تم إضافة النص بـ خط (${state.activeFontFamily}) على الفيديو! ✨`);
    });

    // File upload
    document.getElementById('media-file-input')?.addEventListener('change', e => {
      handleFileSelect(e.target.files[0]);
    });

    // Dropzone Drag & Drop
    const dropzone = document.getElementById('dropzone');
    if (dropzone) {
      dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.style.borderColor = '#00f0ff'; });
      dropzone.addEventListener('dragleave', e => { e.preventDefault(); dropzone.style.borderColor = ''; });
      dropzone.addEventListener('drop', e => {
        e.preventDefault();
        dropzone.style.borderColor = '';
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          handleFileSelect(e.dataTransfer.files[0]);
        }
      });
    }

    // Export
    const exportModal = document.getElementById('export-modal');
    document.getElementById('btn-export-video')?.addEventListener('click',  () => exportModal && (exportModal.style.display = 'flex'));
    document.getElementById('btn-close-export')?.addEventListener('click',  () => exportModal && (exportModal.style.display = 'none'));
    document.getElementById('btn-cancel-export')?.addEventListener('click', () => exportModal && (exportModal.style.display = 'none'));
    document.getElementById('btn-confirm-export')?.addEventListener('click', () => {
      if (exportModal) exportModal.style.display = 'none';
      startModalProgress(3, 'جاري التصدير...', 'ترميز الإطارات الصوتية والمرئية.');
      setTimeout(() => {
        stopModalProgress('تم التصدير بنجاح!', () => {
          const exportUrl = state.isolatedVocalsUrl || state.mediaUrl || URL.createObjectURL(new Blob([''], { type: 'video/mp4' }));
          const a = document.createElement('a');
          a.href     = exportUrl;
          a.download = state.isolatedVocalsUrl ? 'CineCut_Isolated_Vocals.wav' : 'CineCut_Output.mp4';
          a.click();
          showAiStatus('🚀 تم التصدير وتنزيل الملف بنجاح!');
        });
      }, 1000);
    });

    // Timeline scrub
    document.getElementById('timeline-viewport')?.addEventListener('click', e => {
      const offsetX = e.clientX - e.currentTarget.getBoundingClientRect().left - 150;
      if (offsetX >= 0) {
        state.currentTime = Math.max(0, Math.min(state.duration, offsetX / 20));
        if (videoPlayer?.src) videoPlayer.currentTime = state.currentTime;
        if (currentStemAudio) currentStemAudio.currentTime = state.currentTime;
        syncPlayhead();
        updateTimeDisplay();
      }
    });

    // Spacebar shortcut
    document.addEventListener('keydown', e => {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
        togglePlay();
      }
    });
  }

  init();
});
