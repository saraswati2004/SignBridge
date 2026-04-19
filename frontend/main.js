// ══════════════════════════════════════════════
//  PAGE ROUTING
// ══════════════════════════════════════════════
function showPage(page) {
  if (page === 'home') {
    document.getElementById('homePage').style.display = '';
    document.getElementById('translatorPage').style.display = 'none';
    stopCamera();
  } else if (page === 'translator') {
    document.getElementById('homePage').style.display = 'none';
    document.getElementById('translatorPage').style.display = '';
  }
}

function activateTranslator() {
  showPage('translator');
  setTimeout(startCamera, 300);
}

// ══════════════════════════════════════════════
//  SPACEBAR / ESC
// ══════════════════════════════════════════════
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') {
    e.preventDefault();
    const onHome = document.getElementById('homePage').style.display !== 'none';
    if (onHome) {
      document.getElementById('spaceKey')?.classList.add('pressed');
      activateTranslator();
      setTimeout(() => document.getElementById('spaceKey')?.classList.remove('pressed'), 150);
    } else {
      toggleCamera();
    }
  }
  if (e.code === 'Escape') showPage('home');
});

// ══════════════════════════════════════════════
//  CAMERA STATE
// ══════════════════════════════════════════════
let stream        = null;
let predicting    = false;
let sessionId     = 0;
let frameCount    = 0;
let sentence      = '';
let lastLetter    = '';
let debounceTimer = null;
let lastSend      = 0;
let isFetching    = false;   // FIX: gate so only ONE request is in-flight at a time

// FIX: increased interval — no point sending faster than server can respond (~80-120ms)
const FASTAPI_URL       = 'http://127.0.0.1:8000/api/predict/';
const SEND_INTERVAL     = 120;   // ms between sends  (was 80 — caused queue buildup)
const MIN_CONFIDENCE    = 0.60;  // minimum confidence to show/commit a prediction
const COMMIT_HOLD_MS    = 600;   // ms a letter must be held before appending to sentence
const NO_PREDICTION     = '-';

// ── Reusable offscreen canvas (avoid creating one per frame) ─────────────────
const tmpCanvas = document.createElement('canvas');
tmpCanvas.width  = 320;
tmpCanvas.height = 240;
const tmpCtx = tmpCanvas.getContext('2d');

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width:  { ideal: 640 },   // FIX: 640 is enough — 1280 wastes bandwidth
        height: { ideal: 480 },
        facingMode: 'user',
        frameRate: { ideal: 30 },
      }
    });
    const video = document.getElementById('videoFeed');
    video.srcObject = stream;
    video.style.display = 'block';
    document.getElementById('cameraOff').style.display  = 'none';
    document.getElementById('statusDot').classList.add('live');
    document.getElementById('statusText').textContent   = 'LIVE';
    document.getElementById('toggleCamBtn').textContent = '⏹ STOP CAMERA';

    predicting = true;
    isFetching = false;
    sessionId++;
    const mySession = sessionId;
    requestAnimationFrame((ts) => predictLoop(ts, mySession));
  } catch (err) {
    alert('Camera access denied. Please allow camera permissions and try again.');
    console.error(err);
  }
}

function stopCamera() {
  predicting = false;
  sessionId++;
  isFetching = false;
  clearTimeout(debounceTimer);
  lastLetter = '';

  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  const video = document.getElementById('videoFeed');
  video.srcObject = null;
  video.style.display = 'none';
  document.getElementById('cameraOff').style.display  = 'flex';

  const dot = document.getElementById('statusDot');
  dot.classList.remove('live');
  dot.style.background = '';
  document.getElementById('statusText').textContent   = 'OFFLINE';
  document.getElementById('toggleCamBtn').textContent = '📷 START CAMERA';
  clearCanvas();
}

function toggleCamera() {
  if (stream) stopCamera(); else startCamera();
}

// ══════════════════════════════════════════════
//  CANVAS OVERLAY
// ══════════════════════════════════════════════
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawGuideBox(handDetected) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const bw = canvas.width  * 0.55;
  const bh = canvas.height * 0.80;
  const x  = (canvas.width  - bw) / 2;
  const y  = (canvas.height - bh) / 2;

  const color    = handDetected ? 'rgba(0,229,176,1)'   : 'rgba(255,107,107,0.8)';
  const dimColor = handDetected ? 'rgba(0,229,176,0.4)' : 'rgba(255,107,107,0.3)';

  ctx.strokeStyle = dimColor;
  ctx.lineWidth   = 2;
  ctx.setLineDash([8, 4]);
  ctx.strokeRect(x, y, bw, bh);
  ctx.setLineDash([]);

  const cs = 22;
  ctx.strokeStyle = color;
  ctx.lineWidth   = 3;
  [[x,y],[x+bw,y],[x,y+bh],[x+bw,y+bh]].forEach(([cx,cy], i) => {
    const dx = i % 2 === 0 ? 1 : -1;
    const dy = i < 2 ? 1 : -1;
    ctx.beginPath();
    ctx.moveTo(cx + dx*cs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy*cs);
    ctx.stroke();
  });

  ctx.fillStyle = color;
  ctx.font      = 'bold 11px "Space Mono", monospace';
  const label   = handDetected ? '✓ HAND DETECTED' : 'SHOW HAND HERE';
  ctx.fillText(label, x + 8, y - 8);
}

// ══════════════════════════════════════════════
//  PREDICTION LOOP
// ══════════════════════════════════════════════
async function predictLoop(ts, mySession) {
  if (!predicting || mySession !== sessionId) return;
  requestAnimationFrame((ts2) => predictLoop(ts2, mySession));

  const video = document.getElementById('videoFeed');
  if (video.readyState < 2) return;

  // Sync canvas size to display size
  const dw = video.clientWidth  || video.videoWidth  || 640;
  const dh = video.clientHeight || video.videoHeight || 480;
  if (canvas.width !== dw || canvas.height !== dh) {
    canvas.width = dw; canvas.height = dh;
  }

  frameCount++;
  document.getElementById('frameCount').textContent = frameCount;

  // FIX: respect SEND_INTERVAL AND don't send if a request is still in-flight
  if (ts - lastSend < SEND_INTERVAL) return;
  if (isFetching) return;   // FIX: skip frame — previous request hasn't finished
  lastSend = ts;

  // ── Crop the guide-box region and downscale to 320×240 ──────────────────
  const vw = video.videoWidth  || 640;
  const vh = video.videoHeight || 480;

  // FIX: Send the FULL frame, not a cropped version!
  // Cropping ruins the MediaPipe coordinate scale (x, y, z are relative to image size)
  // and squashes the aspect ratio, which destroys the neural network's predictions.
  tmpCtx.drawImage(video, 0, 0, vw, vh, 0, 0, 320, 240);

  // FIX: this line was entirely missing — b64 was never defined
  const b64 = tmpCanvas.toDataURL('image/jpeg', 0.75).split(',')[1];

  isFetching = true;
  try {
    const res = await fetch(FASTAPI_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ frame: b64 }),
    });

    if (mySession !== sessionId || !predicting) { isFetching = false; return; }

    if (!res.ok) {
      console.warn('API error', res.status, await res.text());
      drawGuideBox(false);
      isFetching = false;
      return;
    }

    const data = await res.json();
    if (mySession !== sessionId || !predicting) { isFetching = false; return; }

    drawGuideBox(data.landmarks_detected === true);
    updateUI(data);

  } catch (err) {
    if (mySession !== sessionId || !predicting) { isFetching = false; return; }
    drawGuideBox(false);
    showOfflineState();
  } finally {
    isFetching = false;   // FIX: always release the gate
  }
}

// ══════════════════════════════════════════════
//  UI UPDATES
// ══════════════════════════════════════════════
function updateHandStatus(detected) {
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusText');
  if (detected) {
    dot.style.background = '#00e5b0';
    txt.textContent      = 'HAND DETECTED';
  } else {
    dot.style.background = '#ff6b6b';
    txt.textContent      = 'NO HAND';
  }
}

function updateUI({ prediction, confidence, top5, landmarks_detected }) {
  updateHandStatus(landmarks_detected === true);

  // FIX: only accept prediction when hand detected AND confidence is high enough
  const conf = Number(confidence || 0);
  const validPrediction =
    landmarks_detected === true &&
    typeof prediction === 'string' &&
    /^[A-Z]$/.test(prediction) &&
    conf >= MIN_CONFIDENCE
      ? prediction
      : NO_PREDICTION;

  document.getElementById('predLetter').textContent = validPrediction;

  const pct = validPrediction === NO_PREDICTION ? 0 : Math.round(conf * 100);
  document.getElementById('predConf').textContent = pct + '%';
  document.getElementById('confFill').style.width = pct + '%';

  // ── Sentence commit logic ────────────────────────────────────────────────
  if (validPrediction !== NO_PREDICTION) {
    if (validPrediction !== lastLetter) {
      // New letter — start hold timer
      lastLetter = validPrediction;
      clearTimeout(debounceTimer);
      const letterToAppend = validPrediction;
      debounceTimer = setTimeout(() => {
        if (lastLetter === letterToAppend) {
          sentence += letterToAppend;
          renderSentence();
        }
      }, COMMIT_HOLD_MS);
    }
    // else: same letter still held — timer keeps running, do nothing
  } else {
    // FIX: only clear lastLetter if it was a meaningful letter before
    // Don't thrash the debounce on every low-confidence frame —
    // give a 200ms grace period so brief occlusions don't break detection
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      lastLetter = '';
    }, 200);
  }

  // ── Top-5 predictions ────────────────────────────────────────────────────
  if (top5 && top5.length) {
    document.getElementById('topPredsList').innerHTML =
      top5.slice(0, 5).map(t => `
        <div class="top-pred-item">
          <span class="tp-letter">${t.letter}</span>
          <div class="tp-bar-wrap">
            <div class="tp-bar" style="width:${Math.round(t.conf * 100)}%"></div>
          </div>
          <span class="tp-pct">${Math.round(t.conf * 100)}%</span>
        </div>`).join('');
  } else {
    renderEmptyTop5();
  }
}

function showOfflineState() {
  updateHandStatus(false);
  document.getElementById('predLetter').textContent = '-';
  document.getElementById('predConf').textContent   = '0%';
  document.getElementById('confFill').style.width   = '0%';
  renderEmptyTop5();
  lastLetter = '';
}

function renderEmptyTop5() {
  document.getElementById('topPredsList').innerHTML = `
    <div class="top-pred-item"><span class="tp-letter">-</span><div class="tp-bar-wrap"><div class="tp-bar" style="width:0%"></div></div><span class="tp-pct">0%</span></div>
    <div class="top-pred-item"><span class="tp-letter">-</span><div class="tp-bar-wrap"><div class="tp-bar" style="width:0%"></div></div><span class="tp-pct">0%</span></div>
    <div class="top-pred-item"><span class="tp-letter">-</span><div class="tp-bar-wrap"><div class="tp-bar" style="width:0%"></div></div><span class="tp-pct">0%</span></div>`;
}

// ══════════════════════════════════════════════
//  SENTENCE CONTROLS
// ══════════════════════════════════════════════
function renderSentence() {
  document.getElementById('sentenceText').innerHTML =
    sentence + '<span class="cursor"></span>';
}
function addSpace()      { sentence += ' ';                 renderSentence(); }
function backspace()     { sentence = sentence.slice(0,-1); renderSentence(); }
function clearSentence() { sentence = ''; lastLetter = '';  renderSentence(); }
function copyText() {
  if (!sentence.trim()) return;
  navigator.clipboard.writeText(sentence).then(() => {
    const btn = event.target;
    btn.textContent = 'COPIED!';
    setTimeout(() => btn.textContent = 'COPY', 1500);
  });
}