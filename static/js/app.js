// ===== STATE =====
let currentImageData = null;
let cameraStream = null;
let loadingInterval = null;

// ===== INIT =====
window.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
});

async function checkModelStatus() {
    try {
        const res = await fetch('/model-status');
        const data = await res.json();
        const badge = document.getElementById('model-badge');
        const warning = document.getElementById('model-warning');
        if (data.loaded) {
            badge.textContent = `✅ ${data.type} · ${data.num_classes} classes`;
            badge.style.background = 'rgba(63,185,80,0.1)';
            badge.style.borderColor = 'rgba(63,185,80,0.3)';
            badge.style.color = 'var(--green)';
        } else {
            badge.textContent = '⚠️ Model not loaded';
            badge.style.background = 'rgba(210,153,34,0.1)';
            badge.style.borderColor = 'rgba(210,153,34,0.3)';
            badge.style.color = 'var(--yellow)';
            warning.classList.remove('hidden');
        }
    } catch (e) {
        console.error('Model status check failed:', e);
    }
}

// ===== MODE =====
function switchMode(mode) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
    document.querySelectorAll('.input-section').forEach(s => {
        s.classList.remove('active');
        s.classList.add('hidden');
    });
    const section = document.getElementById(`${mode}-mode`);
    section.classList.remove('hidden');
    section.classList.add('active');
    if (mode !== 'camera') stopCamera();
    resetImageState();
}

// ===== UPLOAD =====
function handleDragOver(e) { e.preventDefault(); document.getElementById('dropzone').classList.add('drag-over'); }
function handleDragLeave() { document.getElementById('dropzone').classList.remove('drag-over'); }
function handleDrop(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) processFile(file);
}
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) processFile(file);
}
function processFile(file) {
    const reader = new FileReader();
    reader.onload = e => {
        currentImageData = e.target.result;
        const img = document.getElementById('preview-img');
        img.src = currentImageData;
        img.classList.remove('hidden');
        document.getElementById('dropzone-inner').classList.add('hidden');
        enableClassify();
    };
    reader.readAsDataURL(file);
}

// ===== CAMERA =====
async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 } }
        });
        const video = document.getElementById('camera-feed');
        video.srcObject = cameraStream;
        video.classList.remove('hidden');
        document.getElementById('camera-placeholder').style.display = 'none';
        document.getElementById('camera-snap').classList.add('hidden');
        document.getElementById('start-cam-btn').classList.add('hidden');
        document.getElementById('snap-btn').classList.remove('hidden');
        document.getElementById('stop-cam-btn').classList.remove('hidden');
        document.getElementById('retake-btn').classList.add('hidden');
        disableClassify();
    } catch {
        alert('Camera access denied or unavailable.');
    }
}
function snapPhoto() {
    const video = document.getElementById('camera-feed');
    const canvas = document.getElementById('camera-canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    currentImageData = canvas.toDataURL('image/jpeg', 0.92);
    const snap = document.getElementById('camera-snap');
    snap.src = currentImageData;
    snap.classList.remove('hidden');
    video.classList.add('hidden');
    document.getElementById('snap-btn').classList.add('hidden');
    document.getElementById('retake-btn').classList.remove('hidden');
    enableClassify();
}
function retakePhoto() {
    document.getElementById('camera-feed').classList.remove('hidden');
    document.getElementById('camera-snap').classList.add('hidden');
    document.getElementById('snap-btn').classList.remove('hidden');
    document.getElementById('retake-btn').classList.add('hidden');
    currentImageData = null;
    disableClassify();
}
function stopCamera() {
    if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
    ['camera-feed','camera-snap'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.srcObject = null; el.classList.add('hidden'); }
    });
    const ph = document.getElementById('camera-placeholder');
    if (ph) ph.style.display = '';
    ['start-cam-btn','stop-cam-btn','snap-btn','retake-btn'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add('hidden');
    });
    const startBtn = document.getElementById('start-cam-btn');
    if (startBtn) startBtn.classList.remove('hidden');
}

// ===== URL =====
function loadFromURL() {
    const url = document.getElementById('image-url-input').value.trim();
    if (!url) return;
    const img = document.getElementById('url-preview-img');
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext('2d').drawImage(img, 0, 0);
        try {
            currentImageData = canvas.toDataURL('image/jpeg');
        } catch {
            currentImageData = url; // fallback
        }
        enableClassify();
    };
    img.onerror = () => alert('Could not load image from URL.');
    img.src = url;
    img.classList.remove('hidden');
}

// ===== HELPERS =====
function enableClassify() { document.getElementById('classify-btn').disabled = false; }
function disableClassify() { document.getElementById('classify-btn').disabled = true; }
function resetImageState() {
    currentImageData = null;
    disableClassify();
    const pi = document.getElementById('preview-img');
    if (pi) { pi.src = ''; pi.classList.add('hidden'); }
    const di = document.getElementById('dropzone-inner');
    if (di) di.classList.remove('hidden');
    const ui = document.getElementById('url-preview-img');
    if (ui) { ui.src = ''; ui.classList.add('hidden'); }
    const urlIn = document.getElementById('image-url-input');
    if (urlIn) urlIn.value = '';
}

// ===== CLASSIFY =====
async function classifyWaste() {
    if (!currentImageData) return;
    showLoading();
    startLoadingSteps();
    try {
        const res = await fetch('/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: currentImageData })
        });
        const data = await res.json();
        stopLoadingSteps();
        if (data.success) {
            displayResult(data.result);
        } else if (data.model_missing) {
            showError('Model not trained yet. Run: python model_training/train.py --data_dir "path/to/dataset"');
        } else {
            showError(data.error || 'Classification failed.');
        }
    } catch (err) {
        stopLoadingSteps();
        showError('Network error: ' + err.message);
    }
}

function showLoading() {
    document.getElementById('default-state').classList.add('hidden');
    document.getElementById('loading-state').classList.remove('hidden');
    document.getElementById('result-state').classList.add('hidden');
    document.getElementById('classify-btn').disabled = true;
    document.getElementById('classify-btn-text').textContent = 'Analyzing...';
}
function startLoadingSteps() {
    const steps = ['step-1','step-2','step-3'];
    let i = 0;
    steps.forEach(s => document.getElementById(s).classList.remove('active'));
    document.getElementById(steps[0]).classList.add('active');
    loadingInterval = setInterval(() => {
        document.getElementById(steps[i]).classList.remove('active');
        i = (i + 1) % steps.length;
        document.getElementById(steps[i]).classList.add('active');
    }, 1000);
}
function stopLoadingSteps() {
    if (loadingInterval) { clearInterval(loadingInterval); loadingInterval = null; }
}
function showError(msg) {
    document.getElementById('loading-state').classList.add('hidden');
    document.getElementById('default-state').classList.remove('hidden');
    document.getElementById('default-icon').textContent = '⚠️';
    document.getElementById('default-title').textContent = 'Error';
    document.getElementById('default-sub').textContent = msg;
    document.getElementById('classify-btn').disabled = false;
    document.getElementById('classify-btn-text').textContent = 'Try Again';
}

// ===== DISPLAY RESULT =====
function displayResult(r) {
    document.getElementById('loading-state').classList.add('hidden');
    document.getElementById('result-state').classList.remove('hidden');

    const severity = (r.severity || 'Yellow').toLowerCase();
    const score = r.score || 5;
    const riskPct = ((score - 1) / 9) * 100;

    // Risk pointer
    document.getElementById('risk-pointer').style.left = `${riskPct}%`;

    // Severity class
    const rs = document.getElementById('result-state');
    rs.className = `severity-${severity}`;

    // Score
    animateScore(0, score);

    // Zone label
    const zones = { green: '🟢 Green Zone', yellow: '🟡 Yellow Zone', red: '🔴 Red Zone' };
    document.getElementById('zone-label').textContent = zones[severity] || '🟡 Yellow Zone';
    document.getElementById('zone-desc').textContent =
        `${r.biodegradable ? 'Biodegradable' : `Takes ${r.decompose_time} to decompose`}. ${r.health_risk} environmental impact.`;

    // Tags
    const tags = document.getElementById('result-tags');
    tags.innerHTML = `
        <span class="tag ${r.biodegradable ? 'tag-green' : 'tag-red'}">${r.biodegradable ? '🌱 Biodegradable' : '⚠️ Non-Biodegradable'}</span>
        <span class="tag tag-blue">🗑️ ${r.item_name}</span>
        <span class="tag tag-${severity === 'green' ? 'green' : severity === 'red' ? 'red' : 'yellow'}">${r.category}</span>
    `;

    // Confidence bar
    const confRaw = r.confidence_raw || parseFloat(r.confidence) || 0;
    document.getElementById('conf-value').textContent = r.confidence;
    document.getElementById('conf-fill').style.width = `${confRaw}%`;

    // Top 3
    const top3 = document.getElementById('top3-section');
    top3.innerHTML = (r.top3 || []).map(t => `
        <div class="top3-item">
            <span>${t.display || t.class}</span>
            <div class="top3-bar"><div class="top3-fill" style="width:${t.prob}%"></div></div>
            <span>${t.prob}%</span>
        </div>`).join('');

    // Metrics
    document.getElementById('metric-category').textContent = r.category || '—';
    document.getElementById('metric-risk').textContent = r.health_risk || '—';
    const sev = document.getElementById('metric-severity');
    sev.textContent = r.severity || '—';
    sev.style.color = severity === 'green' ? 'var(--green)' : severity === 'red' ? 'var(--red)' : 'var(--yellow)';
    document.getElementById('metric-decompose').textContent = r.decompose_time || '—';
    document.getElementById('metric-bio').textContent = r.biodegradable ? '✅ Yes' : '❌ No';
    document.getElementById('metric-confidence').textContent = r.confidence || '—';

    // Fun fact
    if (r.fun_fact) {
        document.getElementById('fun-fact-text').textContent = r.fun_fact;
        document.getElementById('fun-fact-box').classList.remove('hidden');
    } else {
        document.getElementById('fun-fact-box').classList.add('hidden');
    }

    // Disposal
    document.getElementById('disposal-text').textContent = r.disposal || '—';

    // Upcycling
    const grid = document.getElementById('upcycle-grid');
    grid.innerHTML = (r.upcycling || []).map(u => `
        <div class="upcycle-card">
            <div class="upcycle-card-title">${u.title}</div>
            <div class="upcycle-card-desc">${u.desc}</div>
        </div>`).join('');

    // Re-enable button
    document.getElementById('classify-btn').disabled = false;
    document.getElementById('classify-btn-text').textContent = 'Classify Again';
}

function animateScore(from, to) {
    let current = from;
    const el = document.getElementById('score-num');
    const step = () => {
        if (current < to) { current++; el.textContent = current; setTimeout(step, 80); }
    };
    step();
}
