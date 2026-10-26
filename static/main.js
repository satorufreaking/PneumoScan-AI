/**
 * ============================================================================
 * PneumoScan AI - Clinical Dashboard Frontend Controller
 * Features:
 * 1. Dark/Light theme toggle with localStorage persistence
 * 2. File validation (type & size <= 15MB)
 * 3. Interactive Image Viewer (mouse-wheel zoom, pan, brightness/contrast filters, reset)
 *    - Note: Display filters apply to viewport only; raw image sent to model.
 * 4. Recent Scans History (max 20 entries, quota-safe thumbnails, reload scan)
 * 5. Explainable AI: Grad-CAM curtain slider, opacity blend, side-by-side
 * 6. Dual visual confidence meter & sensitivity threshold calibration (default 0.50)
 * ============================================================================
 */

// Application State
let selectedBase64 = null;
let currentGradCamData = null;
let currentThreshold = 0.50; // Default clinical decision cutoff is 0.50

// Viewer Pan & Zoom State
let viewerZoom = 1.0;
let viewerPanX = 0;
let viewerPanY = 0;
let viewerBrightness = 100;
let viewerContrast = 100;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;

// Maximum allowed file size: 15MB
const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024;
const HISTORY_STORAGE_KEY = "pneumoscan_scans_history";
const THEME_STORAGE_KEY = "clinical_theme";

// DOM Elements - Theme
const themeToggleBtn = document.getElementById("theme-toggle");

// DOM Elements - Input & Upload
const fileUpload = document.getElementById("file-upload");
const fileDrag = document.getElementById("file-drag");
const uploadCaption = document.getElementById("upload-caption");
const uploadError = document.getElementById("upload-error");
const uploadErrorText = document.getElementById("upload-error-text");
const loader = document.getElementById("loader");

// DOM Elements - Interactive Viewer
const viewerContainer = document.getElementById("viewer-container");
const viewerViewport = document.getElementById("viewer-viewport");
const imagePreview = document.getElementById("image-preview");
const viewerControls = document.getElementById("viewer-controls");
const valBrightness = document.getElementById("val-brightness");
const valContrast = document.getElementById("val-contrast");
const valZoom = document.getElementById("val-zoom");
const sliderBrightness = document.getElementById("slider-brightness");
const sliderContrast = document.getElementById("slider-contrast");

// DOM Elements - Results Card
const resultsCard = document.getElementById("results-card");
const xaiCard = document.getElementById("xai-card");
const diagnosisBadge = document.getElementById("diagnosis-badge");
const bandBadge = document.getElementById("band-badge");
const bandText = document.getElementById("band-text");
const rawScoreVal = document.getElementById("raw-score-val");
const activeThresholdLabel = document.getElementById("active-threshold-label");
const scaleCutoffText = document.getElementById("scale-cutoff-text");
const probNormalVal = document.getElementById("prob-normal-val");
const probPneumoniaVal = document.getElementById("prob-pneumonia-val");
const meterNormalFill = document.getElementById("meter-normal-fill");
const meterPneumoniaFill = document.getElementById("meter-pneumonia-fill");
const thresholdMarker = document.getElementById("threshold-marker");
const thresholdSlider = document.getElementById("threshold-slider");
const sliderThresholdVal = document.getElementById("slider-threshold-val");
const btnThresh35 = document.getElementById("btn-thresh-35");
const btnThresh50 = document.getElementById("btn-thresh-50");
const btnThresh75 = document.getElementById("btn-thresh-75");
const drivingCaptionText = document.getElementById("driving-caption-text");
const attentionSummaryText = document.getElementById("attention-summary-text");

// DOM Elements - Grad-CAM Slider
const sliderBox = document.getElementById("slider-box");
const sliderClipper = document.getElementById("slider-clipper");
const sliderHandle = document.getElementById("slider-handle");
const splitOrigImg = document.getElementById("split-orig-img");
const splitOverlayImg = document.getElementById("split-overlay-img");
const splitRange = document.getElementById("split-range");

// DOM Elements - Opacity Blend Viewer
const opacityOrigImg = document.getElementById("opacity-orig-img");
const opacityHeatImg = document.getElementById("opacity-heat-img");
const opacitySlider = document.getElementById("opacity-slider");
const opacityPercent = document.getElementById("opacity-percent");

// DOM Elements - Side-by-Side
const sideOrigImg = document.getElementById("side-orig-img");
const sideOverlayImg = document.getElementById("side-overlay-img");

// DOM Elements - History Table
const historyTableBody = document.getElementById("history-table-body");
const historyCount = document.getElementById("history-count");
const historyEmptyRow = document.getElementById("history-empty-row");

// ----------------------------------------------------------------------------
// 1. Theme Toggle Initialization
// ----------------------------------------------------------------------------
function initTheme() {
  const saved = localStorage.getItem(THEME_STORAGE_KEY) || "light";
  document.documentElement.setAttribute("data-theme", saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  const target = current === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", target);
  localStorage.setItem(THEME_STORAGE_KEY, target);
}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", toggleTheme);
}

// ----------------------------------------------------------------------------
// 2. Drag & Drop and File Selection with Validation
// ----------------------------------------------------------------------------
if (fileUpload) {
  fileUpload.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndProcessFile(e.target.files[0]);
    }
  });
}

if (fileDrag) {
  fileDrag.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileDrag.classList.add("dragover");
  });

  fileDrag.addEventListener("dragleave", (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileDrag.classList.remove("dragover");
  });

  fileDrag.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileDrag.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  });
}

function showUploadError(message) {
  if (uploadError && uploadErrorText) {
    uploadErrorText.innerText = message;
    uploadError.classList.remove("hidden");
  }
}

function hideUploadError() {
  if (uploadError) {
    uploadError.classList.add("hidden");
  }
}

function validateAndProcessFile(file) {
  hideUploadError();

  // Validate File Type
  const validTypes = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
  if (!validTypes.includes(file.type.toLowerCase())) {
    showUploadError("Unsupported format. Please select a valid chest radiograph (JPEG, PNG, WebP).");
    return;
  }

  // Validate File Size (Max 15MB)
  if (file.size > MAX_FILE_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    showUploadError(`File is too large (${sizeMb} MB). Maximum allowed size is 15 MB.`);
    return;
  }

  const reader = new FileReader();
  reader.onload = function (e) {
    selectedBase64 = e.target.result;
    displayImageInViewer(selectedBase64);
  };
  reader.readAsDataURL(file);
}

function loadSample(url) {
  hideUploadError();
  fetch(url)
    .then((res) => {
      if (!res.ok) throw new Error("Failed to load sample image");
      return res.blob();
    })
    .then((blob) => {
      const reader = new FileReader();
      reader.onload = function (e) {
        selectedBase64 = e.target.result;
        displayImageInViewer(selectedBase64);
      };
      reader.readAsDataURL(blob);
    })
    .catch((err) => {
      console.error("Error loading sample:", err);
      showUploadError("Could not load preset sample.");
    });
}

function displayImageInViewer(src) {
  imagePreview.src = src;
  viewerContainer.classList.remove("hidden");
  viewerControls.classList.remove("hidden");
  uploadCaption.classList.add("hidden");
  resetViewer();
}

// ----------------------------------------------------------------------------
// 3. Interactive Image Viewer (Pan, Mouse-Wheel Zoom, CSS Filters)
//    Rule: CSS filters and transformations affect DISPLAY only.
// ----------------------------------------------------------------------------
function updateViewerTransform() {
  if (!viewerViewport) return;
  viewerViewport.style.transform = `translate(${viewerPanX}px, ${viewerPanY}px) scale(${viewerZoom})`;
  viewerViewport.style.filter = `brightness(${viewerBrightness}%) contrast(${viewerContrast}%)`;

  if (valZoom) valZoom.innerText = `${viewerZoom.toFixed(1)}x`;
  if (valBrightness) valBrightness.innerText = `${viewerBrightness}%`;
  if (valContrast) valContrast.innerText = `${viewerContrast}%`;
}

function zoomAdjust(delta) {
  viewerZoom = Math.min(Math.max(0.5, viewerZoom + delta), 4.0);
  updateViewerTransform();
}

function onBrightnessChange(val) {
  viewerBrightness = parseInt(val, 10);
  updateViewerTransform();
}

function onContrastChange(val) {
  viewerContrast = parseInt(val, 10);
  updateViewerTransform();
}

function resetViewer() {
  viewerZoom = 1.0;
  viewerPanX = 0;
  viewerPanY = 0;
  viewerBrightness = 100;
  viewerContrast = 100;

  if (sliderBrightness) sliderBrightness.value = 100;
  if (sliderContrast) sliderContrast.value = 100;
  updateViewerTransform();
}

if (viewerContainer) {
  // Mouse-Wheel Zoom
  viewerContainer.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomDelta = e.deltaY < 0 ? 0.15 : -0.15;
    zoomAdjust(zoomDelta);
  }, { passive: false });

  // Mouse Drag to Pan
  viewerContainer.addEventListener("mousedown", (e) => {
    isPanning = true;
    panStartX = e.clientX - viewerPanX;
    panStartY = e.clientY - viewerPanY;
    viewerContainer.classList.add("panning");
  });

  window.addEventListener("mousemove", (e) => {
    if (!isPanning) return;
    viewerPanX = e.clientX - panStartX;
    viewerPanY = e.clientY - panStartY;
    updateViewerTransform();
  });

  window.addEventListener("mouseup", () => {
    if (isPanning) {
      isPanning = false;
      viewerContainer.classList.remove("panning");
    }
  });
}

// ----------------------------------------------------------------------------
// 4. Submit Image & Run Prediction with Grad-CAM
// ----------------------------------------------------------------------------
function submitImage() {
  if (!selectedBase64) {
    showUploadError("Please select or drop a chest radiograph first.");
    return;
  }

  hideUploadError();
  loader.classList.remove("hidden");

  // VIVA EXPLANATION: Send pristine raw image base64. Display filters are NOT baked in.
  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: selectedBase64 }),
  })
    .then((response) => {
      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      return response.json();
    })
    .then((data) => {
      loader.classList.add("hidden");
      currentGradCamData = data;
      renderResults(data);
      saveScanToHistory(data, selectedBase64);
    })
    .catch((err) => {
      console.error("Inference Error:", err);
      loader.classList.add("hidden");
      showUploadError("Inference failed. Check console or server logs.");
    });
}

// ----------------------------------------------------------------------------
// 5. Render Results, Confidence Band & Dual Confidence Meter
// ----------------------------------------------------------------------------
function renderResults(data) {
  const pneuProb = typeof data.pneumonia_prob === "number" ? data.pneumonia_prob : (data.score * 100);
  const normProb = typeof data.normal_prob === "number" ? data.normal_prob : (100 - pneuProb);

  // Raw score display
  if (rawScoreVal) rawScoreVal.innerText = (data.score || 0).toFixed(4);

  // Dual Visual Confidence Meter
  if (probNormalVal) probNormalVal.innerText = `${normProb.toFixed(1)}%`;
  if (probPneumoniaVal) probPneumoniaVal.innerText = `${pneuProb.toFixed(1)}%`;
  if (meterNormalFill) meterNormalFill.style.width = `${normProb}%`;
  if (meterPneumoniaFill) meterPneumoniaFill.style.width = `${pneuProb}%`;

  // Dynamic threshold evaluation (Default 0.50)
  applyThreshold(currentThreshold);

  // Computed Zone Activation (no word 'opacity')
  if (attentionSummaryText && data.attention_summary) {
    attentionSummaryText.innerText = data.attention_summary;
  }

  // Driving Prediction Caption
  if (drivingCaptionText && data.driving_caption) {
    drivingCaptionText.innerText = data.driving_caption;
  }

  // Load Images across Grad-CAM views
  const origSrc = data.original_image || selectedBase64;
  const overlaySrc = data.overlay_image || selectedBase64;
  const heatSrc = data.heatmap_image || data.overlay_image;

  // View 1: Split Curtain Slider
  splitOrigImg.src = origSrc;
  splitOverlayImg.src = overlaySrc;

  // View 2: Opacity Blend
  opacityOrigImg.src = origSrc;
  opacityHeatImg.src = heatSrc;

  // View 3: Side-by-Side
  sideOrigImg.src = origSrc;
  sideOverlayImg.src = overlaySrc;

  // Unhide cards
  resultsCard.classList.remove("hidden");
  xaiCard.classList.remove("hidden");

  // Keep slider width aligned
  setTimeout(updateSliderDimensions, 60);
}

// ----------------------------------------------------------------------------
// 6. Sensitivity & Threshold Calibration (Default Cutoff 0.50)
//    0.35 applies ONLY when user explicitly chooses High Sensitivity preset.
// ----------------------------------------------------------------------------
function applyThreshold(threshold) {
  if (!currentGradCamData) return;

  const score = currentGradCamData.score || 0;
  const isPneumonia = score >= threshold;

  // Diagnosis Badge
  if (diagnosisBadge) {
    diagnosisBadge.className = "badge";
    if (isPneumonia) {
      diagnosisBadge.classList.add("badge-pneumonia");
      diagnosisBadge.innerText = "PNEUMONIA DETECTED";
    } else {
      diagnosisBadge.classList.add("badge-normal");
      diagnosisBadge.innerText = "NORMAL / NEGATIVE";
    }
  }

  // Model Confidence Band (Renamed from Clinical Risk Level)
  if (bandBadge && bandText) {
    bandBadge.className = "badge band-badge";
    if (score >= 0.85 || score <= 0.15) {
      bandBadge.classList.add("band-high");
      bandText.innerText = "High Confidence";
    } else if (score >= 0.70 || score <= 0.30) {
      bandBadge.classList.add("band-moderate");
      bandText.innerText = "Moderate Confidence";
    } else {
      bandBadge.classList.add("band-low");
      bandText.innerText = "Low Confidence (Borderline)";
    }
  }

  // Threshold marker on dual confidence bar
  if (thresholdMarker) {
    thresholdMarker.style.left = `${threshold * 100}%`;
  }

  // Text labels
  if (activeThresholdLabel) activeThresholdLabel.innerText = threshold.toFixed(2);
  if (sliderThresholdVal) sliderThresholdVal.innerText = threshold.toFixed(2);
  if (scaleCutoffText) scaleCutoffText.innerText = threshold.toFixed(2);
}

function onThresholdChange(val) {
  currentThreshold = parseFloat(val) / 100;
  updatePresetButtons(val);
  applyThreshold(currentThreshold);
}

function setThreshold(percent) {
  currentThreshold = percent / 100;
  if (thresholdSlider) thresholdSlider.value = percent;
  updatePresetButtons(percent);
  applyThreshold(currentThreshold);
}

function updatePresetButtons(percent) {
  const p = parseInt(percent, 10);
  if (btnThresh35) btnThresh35.classList.toggle("active", p === 35);
  if (btnThresh50) btnThresh50.classList.toggle("active", p === 50);
  if (btnThresh75) btnThresh75.classList.toggle("active", p === 75);
}

// ----------------------------------------------------------------------------
// 7. Grad-CAM Curtain Slider, Opacity & Mode Switching
// ----------------------------------------------------------------------------
function updateSliderDimensions() {
  if (sliderBox && splitOrigImg) {
    const boxWidth = sliderBox.offsetWidth;
    splitOrigImg.style.width = `${boxWidth}px`;
  }
}

window.addEventListener("resize", updateSliderDimensions);

function onSplitSlide(val) {
  if (!sliderClipper || !sliderHandle) return;
  sliderClipper.style.width = `${val}%`;
  sliderHandle.style.left = `${val}%`;
  updateSliderDimensions();
}

function onOpacityChange(val) {
  if (!opacityHeatImg || !opacityPercent) return;
  opacityHeatImg.style.opacity = val / 100;
  opacityPercent.innerText = `${val}%`;
}

function switchMode(mode) {
  const tabs = {
    split: document.getElementById("tab-split"),
    opacity: document.getElementById("tab-opacity"),
    side: document.getElementById("tab-side"),
  };
  const views = {
    split: document.getElementById("view-split"),
    opacity: document.getElementById("view-opacity"),
    side: document.getElementById("view-side"),
  };

  Object.keys(tabs).forEach((k) => {
    if (k === mode) {
      tabs[k]?.classList.add("active");
      views[k]?.classList.remove("hidden");
    } else {
      tabs[k]?.classList.remove("active");
      views[k]?.classList.add("hidden");
    }
  });

  if (mode === "split") {
    setTimeout(updateSliderDimensions, 30);
  }
}

// ----------------------------------------------------------------------------
// 8. Recent Scans History (localStorage, max 20, quota guard)
// ----------------------------------------------------------------------------
function createThumbnail(base64Src, callback) {
  const img = new Image();
  img.onload = function () {
    const canvas = document.createElement("canvas");
    canvas.width = 48;
    canvas.height = 48;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, 48, 48);
    // Compressed JPEG thumbnail ~1.5KB guards localStorage quota
    const thumbData = canvas.toDataURL("image/jpeg", 0.5);
    callback(thumbData);
  };
  img.src = base64Src;
}

function getStoredHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.warn("Error reading scan history:", e);
    return [];
  }
}

function saveScanToHistory(data, originalBase64) {
  createThumbnail(originalBase64, (thumb) => {
    let history = getStoredHistory();

    const newScan = {
      id: "scan_" + Date.now(),
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", month: "short", day: "numeric" }),
      prediction: data.result,
      score: data.score,
      pneumonia_prob: data.pneumonia_prob,
      normal_prob: data.normal_prob,
      confidence_band: data.confidence_band,
      driving_caption: data.driving_caption,
      attention_summary: data.attention_summary,
      thumbnail: thumb,
      original_image: originalBase64,
      overlay_image: data.overlay_image,
      heatmap_image: data.heatmap_image,
    };

    // Prepend (newest first) and limit to max 20 entries
    history.unshift(newScan);
    if (history.length > 20) {
      history = history.slice(0, 20);
    }

    // Quota guard: Try saving, evict oldest if QuotaExceededError occurs
    let saved = false;
    while (!saved && history.length > 0) {
      try {
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
        saved = true;
      } catch (err) {
        console.warn("Storage quota exceeded, evicting oldest entry...");
        history.pop();
      }
    }

    renderHistoryTable();
  });
}

function renderHistoryTable() {
  if (!historyTableBody) return;
  const history = getStoredHistory();

  if (historyCount) {
    historyCount.innerText = `${history.length} scan${history.length === 1 ? "" : "s"}`;
  }

  historyTableBody.innerHTML = "";

  if (history.length === 0) {
    if (historyEmptyRow) {
      historyTableBody.appendChild(historyEmptyRow);
    } else {
      historyTableBody.innerHTML = `<tr><td colspan="6" class="history-empty">No scans recorded yet. Upload a chest radiograph to start analysis.</td></tr>`;
    }
    return;
  }

  history.forEach((scan) => {
    const tr = document.createElement("tr");

    const isPneu = (scan.prediction || "").toUpperCase().includes("PNEUMONIA");
    const badgeClass = isPneu ? "badge-pneumonia" : "badge-normal";
    const pneuPct = typeof scan.pneumonia_prob === "number" ? `${scan.pneumonia_prob.toFixed(1)}%` : `${(scan.score * 100).toFixed(1)}%`;

    tr.innerHTML = `
      <td><img src="${scan.thumbnail}" class="history-thumb" alt="Thumbnail" /></td>
      <td>${scan.timestamp}</td>
      <td><span class="badge ${badgeClass}" style="font-size: 11px; padding: 3px 8px;">${scan.prediction}</span></td>
      <td><strong>${pneuPct}</strong></td>
      <td><span style="font-size: 12px; color: var(--text-muted);">${scan.confidence_band || "Standard"}</span></td>
      <td style="text-align: right;">
        <button type="button" class="view-scan-btn" onclick="reloadScan('${scan.id}');">
          View
        </button>
      </td>
    `;
    historyTableBody.appendChild(tr);
  });
}

function reloadScan(id) {
  const history = getStoredHistory();
  const scan = history.find((s) => s.id === id);
  if (!scan) return;

  selectedBase64 = scan.original_image;
  currentGradCamData = scan;
  displayImageInViewer(selectedBase64);
  renderResults(scan);

  // Smooth scroll to results
  if (resultsCard) {
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function clearHistory() {
  if (confirm("Clear all recent scan records from browser storage?")) {
    try {
      localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch (e) {
      console.warn(e);
    }
    renderHistoryTable();
  }
}

// ----------------------------------------------------------------------------
// 9. Reset / Clear Workspace
// ----------------------------------------------------------------------------
function clearImage() {
  selectedBase64 = null;
  currentGradCamData = null;

  if (fileUpload) fileUpload.value = "";
  uploadCaption.classList.remove("hidden");
  viewerContainer.classList.add("hidden");
  viewerControls.classList.add("hidden");
  resultsCard.classList.add("hidden");
  xaiCard.classList.add("hidden");
  loader.classList.add("hidden");
  hideUploadError();
  resetViewer();
}

// ----------------------------------------------------------------------------
// Window Initializer & Global Exports
// ----------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  renderHistoryTable();
});

window.submitImage = submitImage;
window.clearImage = clearImage;
window.onSplitSlide = onSplitSlide;
window.onOpacityChange = onOpacityChange;
window.switchMode = switchMode;
window.loadSample = loadSample;
window.onThresholdChange = onThresholdChange;
window.setThreshold = setThreshold;
window.onBrightnessChange = onBrightnessChange;
window.onContrastChange = onContrastChange;
window.zoomAdjust = zoomAdjust;
window.resetViewer = resetViewer;
window.clearHistory = clearHistory;
window.reloadScan = reloadScan;
window.toggleTheme = toggleTheme;
