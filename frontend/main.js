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
let frameCount    = 0;
let sentence      = '';
let lastLetter    = '';
let debounceTimer = null;
let lastSend      = 0;

const FASTAPI_URL = 'http://127.0.0.1:8000/api/predict/';
const SEND_INTERVAL = 250;
const MIN_UI_CONFIDENCE = 0.80;
const NO_PREDICTION = '-';

// Send the FULL frame at the native camera resolution — no downscaling.
// The server runs MediaPipe on the whole frame (no crop needed).
const USE_FULL_RES = true;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width:  { ideal: 1280 },
        height: { ideal: 720  },
        facingMode: 'user'
      }
    });
    const video = document.getElementById('videoFeed');
    video.srcObject = stream;
    video.style.display = 'block';
    document.getElementById('cameraOff').style.display   = 'none';
    document.getElementById('statusDot').classList.add('live');
    document.getElementById('statusText').textContent    = 'LIVE';
    document.getElementById('toggleCamBtn').textContent  = '⏹ STOP CAMERA';
    predicting = true;
    requestAnimationFrame(predictLoop);
  } catch (err) {
    alert('Camera access denied. Please allow camera permissions and try again.');
    console.error(err);
  }
}

function stopCamera() {
  predicting = false;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  const video = document.getElementById('videoFeed');
  video.srcObject = null;
  video.style.display = 'none';
  document.getElementById('cameraOff').style.display  = 'flex';
  document.getElementById('statusDot').classList.remove('live');
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

// Draws a centered guide box — just a UX hint, not a crop boundary.
// Server runs MediaPipe on the FULL frame so hand can be anywhere.
function drawGuideBox(handDetected) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Centre 55% of frame width, 80% of height
  const bw = canvas.width  * 0.55;
  const bh = canvas.height * 0.80;
  const x  = (canvas.width  - bw) / 2;
  const y  = (canvas.height - bh) / 2;

  const color = handDetected ? 'rgba(0,229,176,1)' : 'rgba(255,107,107,0.8)';
  const dimColor = handDetected ? 'rgba(0,229,176,0.4)' : 'rgba(255,107,107,0.3)';

  // Dashed border
  ctx.strokeStyle = dimColor;
  ctx.lineWidth   = 2;
  ctx.setLineDash([8, 4]);
  ctx.strokeRect(x, y, bw, bh);
  ctx.setLineDash([]);

  // Corner accents
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

  // Status label
  ctx.fillStyle = color;
  ctx.font      = 'bold 11px "Space Mono", monospace';
  const label   = handDetected ? '✓ HAND DETECTED' : 'SHOW HAND HERE';
  ctx.fillText(label, x + 8, y - 8);
}

// ══════════════════════════════════════════════
//  PREDICTION LOOP
// ══════════════════════════════════════════════
async function predictLoop(ts) {
  if (!predicting) return;
  requestAnimationFrame(predictLoop);

  const video = document.getElementById('videoFeed');
  if (video.readyState < 2) return;

  // Keep overlay canvas in sync with displayed video size
  const dw = video.clientWidth  || video.videoWidth  || 640;
  const dh = video.clientHeight || video.videoHeight || 480;
  if (canvas.width !== dw || canvas.height !== dh) {
    canvas.width = dw; canvas.height = dh;
  }

  frameCount++;
  document.getElementById('frameCount').textContent = frameCount;

  if (ts - lastSend < SEND_INTERVAL) return;
  lastSend = ts;

  // Capture FULL frame at native camera resolution
  const vw = video.videoWidth  || 640;
  const vh = video.videoHeight || 480;

  const tmp = document.createElement('canvas');
  tmp.width  = vw;
  tmp.height = vh;
  tmp.getContext('2d').drawImage(video, 0, 0, vw, vh);
  const b64 = tmp.toDataURL('image/jpeg', 0.85).split(',')[1];

  try {
    const res = await fetch(FASTAPI_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ frame: b64 }),
    });

    if (!res.ok) {
      console.warn('API error', res.status, await res.text());
      drawGuideBox(false);
      return;
    }

    const data = await res.json();
    // { prediction, confidence, top5:[{letter,conf}], landmarks_detected }
    drawGuideBox(data.landmarks_detected === true);
    updateUI(data);

  } catch (err) {
    // Server offline — show mock Bengali data so UI still works
    drawGuideBox(false);
    showOfflineState();
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
  const validPrediction =
    landmarks_detected === true &&
    typeof prediction === 'string' &&
    /^[A-Z]$/.test(prediction) &&
    Number(confidence || 0) >= MIN_UI_CONFIDENCE
      ? prediction
      : NO_PREDICTION;
  prediction = validPrediction === NO_PREDICTION ? '' : validPrediction;
  confidence = validPrediction === NO_PREDICTION ? 0 : confidence;

  document.getElementById('predLetter').textContent = prediction || '–';
  document.getElementById('predLetter').textContent = validPrediction;
  const pct = Math.round((validPrediction === NO_PREDICTION ? 0 : Number(confidence || 0)) * 100);
  document.getElementById('predConf').textContent   = pct + '%';
  document.getElementById('confFill').style.width   = pct + '%';

  // Auto-append to sentence — threshold 0.70 matches server THRESHOLD
  if (prediction && prediction !== '–' && confidence >= 0.70) {
    if (validPrediction !== lastLetter) {
      lastLetter = validPrediction;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        sentence += validPrediction;
        renderSentence();
      }, 500);
    }
  } else {
    lastLetter = '';
  }

  // Top-5
  if (top5 && top5.length) {
    document.getElementById('topPredsList').innerHTML =
      top5.slice(0,5).map(t => `
        <div class="top-pred-item">
          <span class="tp-letter">${t.letter}</span>
          <div class="tp-bar-wrap">
            <div class="tp-bar" style="width:${Math.round(t.conf*100)}%"></div>
          </div>
          <span class="tp-pct">${Math.round(t.conf*100)}%</span>
        </div>`).join('');
  } else {
    document.getElementById('topPredsList').innerHTML = `
      <div class="top-pred-item"><span class="tp-letter">-</span><div class="tp-bar-wrap"><div class="tp-bar" style="width:0%"></div></div><span class="tp-pct">0%</span></div>
      <div class="top-pred-item"><span class="tp-letter">-</span><div class="tp-bar-wrap"><div class="tp-bar" style="width:0%"></div></div><span class="tp-pct">0%</span></div>
      <div class="top-pred-item"><span class="tp-letter">-</span><div class="tp-bar-wrap"><div class="tp-bar" style="width:0%"></div></div><span class="tp-pct">0%</span></div>`;
  }
}

function showOfflineState() {
  updateHandStatus(false);
  document.getElementById('predLetter').textContent = '-';
  document.getElementById('predConf').textContent = '0%';
  document.getElementById('confFill').style.width = '0%';
  document.getElementById('topPredsList').innerHTML = `
    <div class="top-pred-item"><span class="tp-letter">-</span><div class="tp-bar-wrap"><div class="tp-bar" style="width:0%"></div></div><span class="tp-pct">0%</span></div>
    <div class="top-pred-item"><span class="tp-letter">-</span><div class="tp-bar-wrap"><div class="tp-bar" style="width:0%"></div></div><span class="tp-pct">0%</span></div>
    <div class="top-pred-item"><span class="tp-letter">-</span><div class="tp-bar-wrap"><div class="tp-bar" style="width:0%"></div></div><span class="tp-pct">0%</span></div>`;
  lastLetter = '';
}

// ══════════════════════════════════════════════
//  MOCK DATA (server offline)
// ══════════════════════════════════════════════
const mockLetters = ['A','B','C','D','E','F','G','H','I'];
let mockIdx = 0;
function mockPrediction() {
  const letter = mockLetters[mockIdx++ % mockLetters.length];
  const conf   = 0.75 + Math.random() * 0.22;
  const top5   = mockLetters.slice(0,5).map((l,i) => ({
    letter: l,
    conf: i === 0 ? conf : Math.max(0.01, conf - i*0.13 - Math.random()*0.05),
  }));
  top5[0].letter = letter;
  updateUI({ prediction: letter, confidence: conf, top5, landmarks_detected: true });
}

// ══════════════════════════════════════════════
//  SENTENCE CONTROLS
// ══════════════════════════════════════════════
function renderSentence() {
  document.getElementById('sentenceText').innerHTML =
    sentence + '<span class="cursor"></span>';
}
function addSpace()      { sentence += ' ';                  renderSentence(); }
function backspace()     { sentence = sentence.slice(0,-1);  renderSentence(); }
function clearSentence() { sentence = ''; lastLetter = '';   renderSentence(); }
function copyText() {
  if (!sentence.trim()) return;
  navigator.clipboard.writeText(sentence).then(() => {
    const btn = event.target;
    btn.textContent = 'COPIED!';
    setTimeout(() => btn.textContent = 'COPY', 1500);
  });
}
