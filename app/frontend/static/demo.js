// AdSwapAI demo page logic (2026-09-02 build)
//
// Changes vs. the 2026-03-08 build:
//   - sample videos are referenced by name (no 25-55 MB re-upload per run)
//   - a single polling loop asks the backend for our job only (?job_id=)
//   - queue position, processing speed and ETA are shown
//   - backend errors are surfaced instead of polling forever
//   - "hold frames" (temporal smoothing) control + download link

const billboardImages = [
  'bilboards6.jpg',
  'bilboardsArtboard6.jpg',
  'bilboardsArtboard7.jpg',
  'bilboardsArtboard8.jpg',
  'bilboardsArtboard9.jpg',
  'bilboardsArtboard10.jpg',
  'bilboards1.jpg',
  'bilboards2.jpg',
  'bilboards3.jpg',
  'bilboards4.jpg',
  'bilboards5.jpg',
  'bilboards7.jpg',
  'bilboards8.jpg',
  'bilboardsArtboard1.jpg',
  'bilboardsArtboard2.jpg',
  'bilboardsArtboard3.jpg',
  'bilboardsArtboard4.jpg',
  'bilboardsArtboard5.jpg'
];

const sampleVideos = ['1.mp4', '2.mp4', '3.mp4'];

const MAX_UPLOAD_MB = 50;
const POLL_INTERVAL_MS = 1000;
const MAX_MISSING_POLLS = 10;

let selectedImage = billboardImages[0];
let selectedVideo = sampleVideos[0];

// DOM elements
const form = document.getElementById('procForm');
const progress = document.getElementById('progress');
const resultVid = document.getElementById('resultVid');
const resultDownload = document.getElementById('resultDownload');
const maskAlphaSlider = document.getElementById('maskAlpha');
const alphaValueSpan = document.getElementById('alphaValue');
const confThreshSlider = document.getElementById('confThreshold');
const threshValueSpan = document.getElementById('threshValue');
const minMaskSizeSlider = document.getElementById('minMaskSize');
const sizeValueSpan = document.getElementById('sizeValue');
const humanConfThreshSlider = document.getElementById('humanConfThreshold');
const humanThreshValueSpan = document.getElementById('humanThreshValue');
const holdFramesSlider = document.getElementById('holdFrames');
const holdValueSpan = document.getElementById('holdValue');
const maskControls = document.getElementById('maskControls');
const imageSelectPanel = document.getElementById('imageSelectPanel');
const gallery = document.getElementById('thumbnailGallery');
const previewImage = document.getElementById('previewImage');
const videoGallery = document.getElementById('videoGallery');
const sampleVideoContainer = document.getElementById('sampleVideoContainer');
const uploadVideoContainer = document.getElementById('uploadVideoContainer');

const placeholderSvg = (text, w, h) =>
  `data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"%3E%3Crect width="${w}" height="${h}" fill="%23f0f0f0"/%3E%3Ctext x="50%" y="50%" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle" fill="%23999"%3E${text}%3C/text%3E%3C/svg%3E`;

function generateUniqueId() {
  const timestamp = Date.now();
  const randomStr = Math.random().toString(36).substring(2, 10);
  return `${timestamp}-${randomStr}`;
}

function setActiveJobId(jobId) {
  if (jobId) sessionStorage.setItem('activeJobId', jobId);
  else sessionStorage.removeItem('activeJobId');
  window.activeJobId = jobId;
}

function getActiveJobId() {
  return window.activeJobId || sessionStorage.getItem('activeJobId');
}

function formatSeconds(sec) {
  if (!isFinite(sec) || sec < 0) return '?';
  sec = Math.round(sec);
  if (sec < 60) return `${sec}s`;
  return `${Math.floor(sec / 60)}m ${sec % 60}s`;
}

// --------------------------------------------------------------------------
// Init
// --------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
  window.activeJobId = null;
  if (window.progressPollInterval) clearInterval(window.progressPollInterval);
  if (!window.name) window.name = 'browser-' + Date.now();

  populateGallery();
  populateVideoGallery();
  applyModeVisibility();
  applyVideoSourceVisibility();

  progress.textContent = 'Ready to process video with AI technology';
  resultVid.style.display = 'none';
  resultDownload.classList.add('hide');

  alphaValueSpan.textContent = maskAlphaSlider.value;
  threshValueSpan.textContent = confThreshSlider.value;
  sizeValueSpan.textContent = Math.round(minMaskSizeSlider.value * 100);
  humanThreshValueSpan.textContent = humanConfThreshSlider.value;
  holdValueSpan.textContent = holdFramesSlider.value;

  // Resume polling if the page was reloaded during a job
  const previousJobId = getActiveJobId();
  if (previousJobId) {
    updateProcessButtonState(true);
    startProgressPolling(previousJobId);
  } else {
    updateProcessButtonState(false);
  }
});

function applyModeVisibility() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  maskControls.classList.toggle('hide', mode !== 'mask');
  imageSelectPanel.classList.toggle('hide', mode !== 'image');
}

function applyVideoSourceVisibility() {
  const source = document.querySelector('input[name="videoSource"]:checked').value;
  sampleVideoContainer.classList.toggle('hide', source !== 'sample');
  uploadVideoContainer.classList.toggle('hide', source !== 'upload');
  document.getElementById('video').required = source === 'upload';
}

function populateGallery() {
  gallery.innerHTML = '';
  billboardImages.forEach((image, index) => {
    const thumbnail = document.createElement('div');
    thumbnail.className = 'thumbnail';
    if (index === 0) thumbnail.classList.add('selected');

    const img = document.createElement('img');
    img.src = `/images/${image}`;
    img.alt = image;
    img.loading = 'lazy';
    img.onerror = function () { this.src = placeholderSvg('Image not found', 100, 80); };

    thumbnail.appendChild(img);
    thumbnail.addEventListener('click', () => selectImage(image, thumbnail));
    gallery.appendChild(thumbnail);
  });

  previewImage.src = `/images/${selectedImage}`;
  previewImage.onerror = function () { this.src = placeholderSvg('Preview not available', 200, 150); };
}

function populateVideoGallery() {
  videoGallery.innerHTML = '';
  sampleVideos.forEach((video, index) => {
    const thumbName = video.replace(/\.\w+$/, '.png');
    const thumbnail = document.createElement('div');
    thumbnail.className = 'thumbnail video-thumbnail';
    if (index === 0) thumbnail.classList.add('selected');

    const img = document.createElement('img');
    img.src = `/sampleVideos/${thumbName}`;
    img.alt = video;
    img.onerror = function () { this.src = placeholderSvg('No thumbnail', 100, 80); };

    thumbnail.appendChild(img);
    thumbnail.addEventListener('click', () => selectVideo(video, thumbnail));
    videoGallery.appendChild(thumbnail);
  });
}

function selectImage(image, thumbnailElement) {
  selectedImage = image;
  gallery.querySelectorAll('.thumbnail').forEach(t => t.classList.remove('selected'));
  thumbnailElement.classList.add('selected');
  previewImage.src = `/images/${image}`;
}

function selectVideo(video, thumbnailElement) {
  selectedVideo = video;
  videoGallery.querySelectorAll('.thumbnail').forEach(t => t.classList.remove('selected'));
  thumbnailElement.classList.add('selected');
}

// Slider value displays
maskAlphaSlider.addEventListener('input', () => { alphaValueSpan.textContent = maskAlphaSlider.value; });
confThreshSlider.addEventListener('input', () => { threshValueSpan.textContent = confThreshSlider.value; });
minMaskSizeSlider.addEventListener('input', () => { sizeValueSpan.textContent = Math.round(minMaskSizeSlider.value * 100); });
humanConfThreshSlider.addEventListener('input', () => { humanThreshValueSpan.textContent = humanConfThreshSlider.value; });
holdFramesSlider.addEventListener('input', () => { holdValueSpan.textContent = holdFramesSlider.value; });

document.querySelectorAll('input[name="mode"]').forEach(radio => radio.addEventListener('change', applyModeVisibility));
document.querySelectorAll('input[name="videoSource"]').forEach(radio => radio.addEventListener('change', applyVideoSourceVisibility));

function updateProcessButtonState(isProcessing) {
  const processButton = document.querySelector('button[type="submit"]');
  processButton.disabled = isProcessing;
  processButton.style.opacity = isProcessing ? '0.5' : '1';
  processButton.style.cursor = isProcessing ? 'not-allowed' : 'pointer';
}

// --------------------------------------------------------------------------
// Submit
// --------------------------------------------------------------------------
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (window.progressPollInterval) clearInterval(window.progressPollInterval);

  resultVid.style.display = 'none';
  resultVid.removeAttribute('src');
  resultDownload.classList.add('hide');

  const clientJobId = generateUniqueId();
  const formData = new FormData();
  const mode = form.querySelector('input[name="mode"]:checked').value;
  const videoSource = form.querySelector('input[name="videoSource"]:checked').value;

  formData.append('mode', mode);
  formData.append('client_job_id', clientJobId);
  formData.append('browser_session', window.name || 'browser-' + Date.now());

  if (videoSource === 'sample') {
    // The backend has the sample videos mounted: send only the name.
    formData.append('sample_video', selectedVideo);
  } else {
    const videoFile = document.getElementById('video').files[0];
    if (!videoFile) {
      progress.textContent = 'Please select a video file';
      return;
    }
    const acceptedFormats = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
    if (!acceptedFormats.includes(videoFile.type)) {
      progress.textContent = 'Error: Only MP4, MOV and AVI video formats are supported';
      return;
    }
    if (videoFile.size > MAX_UPLOAD_MB * 1024 * 1024) {
      progress.textContent = `Error: Video file must be under ${MAX_UPLOAD_MB}MB`;
      return;
    }
    formData.append('video', videoFile);
  }

  formData.append('conf_threshold', confThreshSlider.value);
  formData.append('min_mask_size', minMaskSizeSlider.value);
  formData.append('human_conf_threshold', humanConfThreshSlider.value);
  formData.append('enable_human_filter', document.getElementById('enableHumanFilter').checked);
  formData.append('hold_frames', holdFramesSlider.value);

  if (mode === 'mask') {
    const colorHex = document.getElementById('maskColor').value;
    formData.append('mask_color_r', parseInt(colorHex.substr(1, 2), 16));
    formData.append('mask_color_g', parseInt(colorHex.substr(3, 2), 16));
    formData.append('mask_color_b', parseInt(colorHex.substr(5, 2), 16));
    formData.append('mask_alpha', maskAlphaSlider.value);
  } else {
    try {
      const response = await fetch(`/images/${selectedImage}`);
      if (!response.ok) throw new Error(response.statusText);
      const blob = await response.blob();
      formData.append('replacement', new File([blob], selectedImage, { type: 'image/jpeg' }));
    } catch (error) {
      progress.textContent = `Error loading image: ${error.message}`;
      return;
    }
  }

  updateProcessButtonState(true);
  progress.textContent = videoSource === 'sample' ? 'Submitting job...' : 'Uploading video...';

  try {
    const resp = await fetch('/process', { method: 'POST', body: formData });
    if (!resp.ok) {
      let detail = resp.statusText || 'Server error';
      try {
        const body = await resp.json();
        if (body && body.error) detail = body.error;
      } catch (_) {
        try { detail = (await resp.text()).replace(/<[^>]+>/g, ' ').trim().slice(0, 200) || detail; } catch (_) { /* ignore */ }
      }
      progress.textContent = `Error: ${detail}`;
      updateProcessButtonState(false);
      return;
    }

    const data = await resp.json();
    const jobId = data.job_id || clientJobId;
    setActiveJobId(jobId);
    progress.textContent = data.queue_position > 1
      ? `Queued (position ${data.queue_position})...`
      : 'Queued, starting shortly...';
    startProgressPolling(jobId);
  } catch (error) {
    progress.textContent = `Error: ${error.message || 'Connection failed'}`;
    updateProcessButtonState(false);
  }
});

// --------------------------------------------------------------------------
// Polling
// --------------------------------------------------------------------------
function startProgressPolling(jobId) {
  if (window.progressPollInterval) clearInterval(window.progressPollInterval);
  let missing = 0;

  const stop = () => {
    clearInterval(window.progressPollInterval);
    window.progressPollInterval = null;
    setActiveJobId(null);
    updateProcessButtonState(false);
  };

  const showResult = (job) => {
    const url = job.video_url || ('/videos/' + job.final_output_filename);
    resultVid.src = url;
    resultVid.style.display = 'block';
    resultVid.load();
    resultDownload.href = url;
    resultDownload.download = job.final_output_filename || 'processed.mp4';
    resultDownload.classList.remove('hide');
    const speed = job.processing_fps ? ` (${job.processing_fps} fps, ${formatSeconds(job.elapsed_seconds)})` : '';
    progress.textContent = `Video ready!${speed}`;
  };

  const tick = async () => {
    let data;
    try {
      const response = await fetch(`/process-status?job_id=${encodeURIComponent(jobId)}`, { cache: 'no-store' });
      if (response.status === 404) {
        missing += 1;
        if (missing >= MAX_MISSING_POLLS) {
          progress.textContent = 'Processing job not found (server restarted?). Please try again.';
          stop();
        }
        return;
      }
      if (!response.ok) return;
      data = await response.json();
    } catch (error) {
      progress.textContent = `Error fetching progress: ${error.message}`;
      return;
    }

    const job = data[jobId] || Object.values(data).find(j => j.job_id === jobId);
    if (!job) {
      missing += 1;
      if (missing >= MAX_MISSING_POLLS) {
        progress.textContent = 'Processing job not found or finished.';
        stop();
      }
      return;
    }
    missing = 0;

    switch (job.status) {
      case 'queued':
        progress.textContent = `Queued (position ${job.queue_position || 1})... waiting for the GPU worker`;
        break;
      case 'processing': {
        const current = job.current_frame || 0;
        const total = job.total_frames || 0;
        const percent = total ? Math.round((current / total) * 100) : 0;
        let extra = '';
        if (job.start_time && current > 5) {
          const elapsed = Date.now() / 1000 - job.start_time;
          const fps = current / elapsed;
          if (total > current && fps > 0) {
            extra = ` · ${fps.toFixed(1)} fps · ETA ${formatSeconds((total - current) / fps)}`;
          }
        }
        progress.textContent = `Processing: ${percent}% (${current}/${total} frames)${extra}`;
        break;
      }
      case 'done':
        if (job.video_ready && job.final_output_filename) {
          showResult(job);
          stop();
        } else {
          progress.textContent = 'Processing complete. Video is being prepared...';
        }
        break;
      case 'error':
        progress.textContent = `Processing failed: ${job.error || 'unknown error'}`;
        stop();
        break;
      default:
        // Older backend without a status field
        if (job.is_processing) {
          const percent = Math.round(((job.current_frame || 0) / (job.total_frames || 1)) * 100);
          progress.textContent = `Processing: ${percent}%`;
        } else if (job.video_ready && job.final_output_filename) {
          showResult(job);
          stop();
        } else if (job.error) {
          progress.textContent = `Processing failed: ${job.error}`;
          stop();
        } else {
          progress.textContent = 'Processing complete. Video is being prepared...';
        }
    }
  };

  tick();
  window.progressPollInterval = setInterval(tick, POLL_INTERVAL_MS);
}

console.log('AdSwapAI demo page loaded');
