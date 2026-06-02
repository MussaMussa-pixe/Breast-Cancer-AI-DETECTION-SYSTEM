const API = 'http://localhost:8000';
let currentResult = null;

// ── Health Check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  const pill = document.getElementById('statusPill');
  const txt  = pill.querySelector('.status-text');
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
    const d = await r.json();
    if (d.status === 'ok' && d.model_loaded) {
      pill.classList.add('online');
      pill.classList.remove('offline');
      txt.textContent = 'Model Ready';
    } else {
      throw new Error('model not loaded');
    }
  } catch {
    pill.classList.add('offline');
    pill.classList.remove('online');
    txt.textContent = 'Offline';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ── Drag & Drop ───────────────────────────────────────────────────────────────
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

['dragenter','dragover'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave','drop'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', ev => { const f = ev.dataTransfer.files[0]; if (f) handleFile(f); });
dropZone.addEventListener('click', e => { if (!e.target.closest('button')) fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file) {
  if (!file.type.startsWith('image/')) { alert('Please upload an image file.'); return; }
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('previewImg').src = e.target.result;
    document.getElementById('previewFilename').textContent = file.name;
    document.getElementById('previewSize').textContent = formatBytes(file.size);
    document.getElementById('dropContent').style.display = 'none';
    document.getElementById('previewContent').style.display = 'flex';
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('analyzeBtn')._file = file;
  };
  reader.readAsDataURL(file);
}

function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function resetUpload() {
  document.getElementById('dropContent').style.display = 'flex';
  document.getElementById('previewContent').style.display = 'none';
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('analyzeBtn')._file = null;
  fileInput.value = '';
}

// ── Analyze ───────────────────────────────────────────────────────────────────
async function analyzeImage() {
  const btn  = document.getElementById('analyzeBtn');
  const file = btn._file;
  if (!file) return;

  // Loading state
  btn.querySelector('.btn-text').style.display = 'none';
  btn.querySelector('.btn-loader').style.display = 'flex';
  btn.disabled = true;

  try {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(`${API}/predict`, { method: 'POST', body: fd });
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || 'Prediction failed');
    }
    const data = await r.json();
    currentResult = data;
    showResults(data);
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    addChatContextMessage(data);
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    btn.querySelector('.btn-text').style.display = 'flex';
    btn.querySelector('.btn-loader').style.display = 'none';
    btn.disabled = false;
  }
}

// ── Show Results ──────────────────────────────────────────────────────────────
function showResults(d) {
  const isPositive = d.prediction.toLowerCase().includes('positive');
  const sec = document.getElementById('resultsSection');
  sec.style.display = 'block';

  // Header card
  const iconWrap = document.getElementById('resultIconWrap');
  iconWrap.className = 'result-icon-wrap ' + (isPositive ? 'positive' : 'negative');
  iconWrap.innerHTML = isPositive
    ? `<svg viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="2"><path d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;

  const lbl = document.getElementById('resultLabel');
  lbl.textContent = d.prediction;
  lbl.className = 'result-label ' + (isPositive ? 'positive' : 'negative');
  document.getElementById('resultSublabel').textContent = isPositive
    ? 'IDC cancer patterns detected in tissue image'
    : 'No IDC cancer patterns detected in tissue image';

  document.getElementById('rcdValue').textContent = d.confidence.toFixed(1) + '%';
  setTimeout(() => {
    document.getElementById('rcdBar').style.width = d.confidence + '%';
  }, 100);

  // Report meta
  document.getElementById('reportMeta').innerHTML =
    `Report ID: <strong>BCR-${Date.now().toString(36).toUpperCase()}</strong> &nbsp;|&nbsp; ` +
    `Generated: <strong>${d.timestamp}</strong> &nbsp;|&nbsp; File: <strong>${d.filename}</strong>`;

  // Report body
  const riskClass = d.risk_level === 'High' ? 'risk-high' : d.risk_level === 'Moderate' ? 'risk-moderate' : 'risk-low';
  const recs = isPositive
    ? ['Schedule an immediate appointment with a breast oncologist or surgeon',
       'Request clinical breast examination and imaging (mammogram / MRI)',
       'A tissue biopsy may be required for definitive diagnosis',
       'Bring this AI report to your medical appointment',
       'Seek a second medical opinion if desired']
    : ['Continue regular breast cancer screening as recommended by your doctor',
       'Perform monthly breast self-examinations',
       'Schedule annual mammograms (recommended for women 40+)',
       'Maintain a healthy lifestyle to reduce cancer risk',
       'Consult your doctor if you notice any unusual changes'];

  const clinicalNotes = isPositive
    ? ['AI detected tissue morphology consistent with IDC patterns',
       `Prediction confidence: ${d.confidence.toFixed(1)}% (Risk Level: ${d.risk_level})`,
       'ResNet50 model trained on 277,524 IDC histopathology patches',
       'Binary classification: Class 1 (IDC Positive) detected']
    : ['AI did not detect IDC-consistent tissue morphology patterns',
       `Prediction confidence: ${d.confidence.toFixed(1)}% (Risk Level: ${d.risk_level})`,
       'ResNet50 model trained on 277,524 IDC histopathology patches',
       'Binary classification: Class 0 (IDC Negative) detected'];

  document.getElementById('reportBody').innerHTML = `
    <div class="report-grid">
      <div class="report-item">
        <div class="report-item-label">Diagnosis Result</div>
        <div class="report-item-value" style="color:${isPositive?'#f87171':'#4ade80'}">${d.prediction}</div>
      </div>
      <div class="report-item">
        <div class="report-item-label">Confidence Score</div>
        <div class="report-item-value">${d.confidence.toFixed(2)}%</div>
      </div>
      <div class="report-item">
        <div class="report-item-label">Risk Level</div>
        <div class="report-item-value"><span class="risk-badge ${riskClass}">${d.risk_level}</span></div>
      </div>
      <div class="report-item">
        <div class="report-item-label">AI Model</div>
        <div class="report-item-value" style="font-size:.85rem">ResNet50</div>
      </div>
    </div>

    <div class="report-section">
      <h4>Clinical AI Notes</h4>
      <ul class="report-list">${clinicalNotes.map(n=>`<li>${n}</li>`).join('')}</ul>
    </div>

    <div class="report-section">
      <h4>Recommended Actions</h4>
      <ul class="report-list">${recs.map(r=>`<li>${r}</li>`).join('')}</ul>
    </div>

    <div class="report-section">
      <h4>About This Analysis</h4>
      <ul class="report-list">
        <li>This analysis uses a ResNet50 model fine-tuned on the IDC Histopathology Dataset (Kaggle)</li>
        <li>The model classifies 224×224 tissue patches as IDC-Positive or IDC-Negative</li>
        <li>IDC (Invasive Ductal Carcinoma) is the most common form of breast cancer (~80% of cases)</li>
        <li>AI screening tools are meant to assist, not replace, clinical diagnostic procedures</li>
      </ul>
    </div>`;
}

function newAnalysis() {
  resetUpload();
  currentResult = null;
  document.getElementById('resultsSection').style.display = 'none';
  document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
}

// ── Chatbot ───────────────────────────────────────────────────────────────────
function addChatContextMessage(d) {
  const isPos = d.prediction.toLowerCase().includes('positive');
  const text = `📊 Analysis complete! Result: **${d.prediction}** (${d.confidence.toFixed(1)}% confidence, Risk: ${d.risk_level}). You can now ask me about what this means, treatment options, or next steps.`;
  appendMsg('ai', text);
  document.getElementById('chatSection').scrollIntoView({ behavior: 'smooth' });
}

function appendMsg(role, text) {
  const box = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = `chat-msg ${role === 'ai' ? 'ai-msg' : 'user-msg'}`;

  const avatarHtml = role === 'ai'
    ? `<div class="msg-avatar"><svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="12" fill="url(#msgG)"/><path d="M8 12C8 9 10 7 12 7C14 7 16 9 16 12C16 15 14 17 12 17" stroke="white" stroke-width="1.5" stroke-linecap="round"/><circle cx="12" cy="12" r="2" fill="white"/><defs><linearGradient id="msgG" x1="0" y1="0" x2="24" y2="24"><stop offset="0%" stop-color="#e879a0"/><stop offset="100%" stop-color="#9b4fca"/></linearGradient></defs></svg></div>`
    : `<div class="msg-avatar"><svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="12" fill="#333"/><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="#888"/></svg></div>`;

  // Convert markdown-like **bold** to <strong>
  const html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');

  div.innerHTML = role === 'ai'
    ? `${avatarHtml}<div class="msg-bubble ai-bubble">${html}</div>`
    : `${avatarHtml}<div class="msg-bubble user-bubble">${html}</div>`;

  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function showTyping() {
  const box = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'chat-msg ai-msg';
  div.id = 'typingIndicator';
  div.innerHTML = `<div class="msg-avatar"><svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="12" fill="url(#tG)"/><defs><linearGradient id="tG" x1="0" y1="0" x2="24" y2="24"><stop offset="0%" stop-color="#e879a0"/><stop offset="100%" stop-color="#9b4fca"/></linearGradient></defs></svg></div><div class="msg-bubble ai-bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

  appendMsg('user', msg);
  showTyping();

  try {
    const body = { message: msg };
    if (currentResult) {
      body.prediction = currentResult.prediction;
      body.confidence = currentResult.confidence;
    }
    const r = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const d = await r.json();
    hideTyping();
    appendMsg('ai', d.reply);
  } catch {
    hideTyping();
    appendMsg('ai', "I'm having trouble connecting to the server. Please make sure the backend is running and try again.");
  }
}

function sendQuick(msg) {
  document.getElementById('chatInput').value = msg;
  sendMessage();
}

// Header scroll effect
window.addEventListener('scroll', () => {
  const h = document.getElementById('header');
  h.style.background = window.scrollY > 50 ? 'rgba(10,10,15,0.95)' : 'rgba(10,10,15,0.8)';
});
