/* ==========================================================================
   AgriVision AI - Client Controller
   ========================================================================== */

let currentDetectionData = null;
let currentTab = 'annotated'; // 'annotated' or 'spray'

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  setupEventListeners();
  await loadDatasetStats();
  await loadDatasetSamples();
  // Run default detection on first sample
  runDetectionForSample('agri_0_1009.jpeg');
}

function setupEventListeners() {
  // File Upload
  const fileInput = document.getElementById('fileInput');
  const dropzone = document.getElementById('dropzone');

  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFileSelect);

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  // Viewport Tabs
  const tabAnnotated = document.getElementById('tabAnnotated');
  const tabSpray = document.getElementById('tabSpray');

  tabAnnotated.addEventListener('click', () => switchTab('annotated'));
  tabSpray.addEventListener('click', () => switchTab('spray'));
}

async function loadDatasetStats() {
  try {
    const res = await fetch('/api/stats');
    const stats = await res.json();

    document.getElementById('statTotalImages').textContent = stats.total_images.toLocaleString();
    document.getElementById('statTotalAnnotations').textContent = stats.total_bounding_boxes.toLocaleString();
    document.getElementById('statCropCount').textContent = stats.class_distribution.crop.count.toLocaleString();
    document.getElementById('statWeedCount').textContent = stats.class_distribution.weed.count.toLocaleString();

    renderPipelineSteps(stats.pipeline_steps);
  } catch (err) {
    console.error('Error loading stats:', err);
  }
}

function renderPipelineSteps(steps) {
  const container = document.getElementById('pipelineContainer');
  if (!container) return;

  container.innerHTML = steps.map(s => `
    <div class="pipeline-card">
      <div class="step-num">0${s.step}</div>
      <h3 class="step-title">${s.title}</h3>
      <p class="step-desc">${s.description}</p>
      <span class="step-stat">${s.stat}</span>
    </div>
  `).join('');
}

async function loadDatasetSamples() {
  try {
    const res = await fetch('/api/samples?limit=24');
    const samples = await res.json();
    const container = document.getElementById('sampleChipsContainer');

    if (!container) return;

    container.innerHTML = samples.map((s, idx) => `
      <div class="sample-chip ${idx === 0 ? 'active' : ''}" onclick="selectSampleChip(this, '${s.filename}')">
        <img src="/api/image/${s.filename}" alt="${s.filename}">
        <span>${s.filename}</span>
      </div>
    `).join('');
  } catch (err) {
    console.error('Error loading samples:', err);
  }
}

function selectSampleChip(element, filename) {
  document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
  element.classList.add('active');
  runDetectionForSample(filename);
}

function handleFileSelect(e) {
  if (e.target.files.length > 0) {
    handleFileUpload(e.target.files[0]);
  }
}

async function handleFileUpload(file) {
  showLoader(true);
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/detect', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    renderDetectionResult(data);
  } catch (err) {
    console.error('Error uploading image:', err);
  } finally {
    showLoader(false);
  }
}

async function runDetectionForSample(filename) {
  showLoader(true);
  try {
    const res = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_filename: filename })
    });
    const data = await res.json();
    renderDetectionResult(data);
  } catch (err) {
    console.error('Error running detection:', err);
  } finally {
    showLoader(false);
  }
}

function renderDetectionResult(data) {
  currentDetectionData = data;

  const viewportImage = document.getElementById('viewportImage');
  if (currentTab === 'annotated') {
    viewportImage.src = data.images.annotated;
  } else {
    viewportImage.src = data.images.spray_simulation;
  }

  // Update Analysis Cards
  document.getElementById('analysisCrops').textContent = data.counts.crop_count;
  document.getElementById('analysisWeeds').textContent = data.counts.weed_count;
  document.getElementById('analysisSavings').textContent = `${data.pesticide_metrics.pesticide_saved_percent}%`;
  document.getElementById('analysisTargetLiters').textContent = `${data.pesticide_metrics.targeted_spray_liters} L/ha`;

  renderSpraySimulationCanvas(data);
}

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tabAnnotated').classList.toggle('active', tab === 'annotated');
  document.getElementById('tabSpray').classList.toggle('active', tab === 'spray');

  if (currentDetectionData && currentDetectionData.images) {
    const viewportImage = document.getElementById('viewportImage');
    viewportImage.src = tab === 'annotated' ? currentDetectionData.images.annotated : currentDetectionData.images.spray_simulation;
  }
}

function showLoader(show) {
  const loader = document.getElementById('loaderOverlay');
  if (loader) {
    loader.classList.toggle('active', show);
  }
}

function renderSpraySimulationCanvas(data) {
  const container = document.getElementById('sprayCanvasBox');
  if (!container) return;

  const weedCount = data.counts.weed_count;
  const cropCount = data.counts.crop_count;
  const savings = data.pesticide_metrics.pesticide_saved_percent;

  container.innerHTML = `
    <div style="padding: 1.5rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="color: var(--accent-crop); font-family: var(--font-mono); font-weight: 700;">
          [ROBOTIC SPRAY BOOM SIMULATOR] - 8 SMART NOZZLES ACTIVE
        </span>
        <span style="background: rgba(16,185,129,0.15); color: var(--accent-crop); padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
          SYSTEM READY
        </span>
      </div>

      <div style="display: flex; gap: 8px; justify-content: space-around; margin: 1.5rem 0;">
        ${Array.from({ length: 8 }).map((_, i) => {
          const isFiring = weedCount > 0 && (i % 2 === 1 || i === 3);
          return `
            <div style="flex: 1; text-align: center; background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); padding: 12px; border-radius: 8px;">
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 6px;">NOZZLE 0${i + 1}</div>
              <div style="width: 14px; height: 14px; border-radius: 50%; margin: 0 auto; background: ${isFiring ? 'var(--accent-weed)' : 'var(--accent-crop)'}; box-shadow: 0 0 10px ${isFiring ? 'var(--accent-weed)' : 'var(--accent-crop)'};"></div>
              <div style="font-size: 0.7rem; margin-top: 6px; color: ${isFiring ? 'var(--accent-weed)' : 'var(--accent-crop)'}; font-weight: 700;">
                ${isFiring ? 'SPRAYING WEED' : 'BYPASS CROP'}
              </div>
            </div>
          `;
        }).join('')}
      </div>

      <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted); border-top: 1px solid var(--border-color); padding-top: 12px;">
        <span>Targeted Spray Efficiency: <strong style="color: var(--accent-crop);">${savings}% Chemical Saved</strong></span>
        <span>Soil Health Impact: <strong style="color: var(--accent-cyan);">Zero Toxic Runoff</strong></span>
      </div>
    </div>
  `;
}
