// AdSwapAI R&D, 2025-06-05: demo page controller (upload, poll /process-status, play result)
// List of available billboard images
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

    // List of available sample videos
    const sampleVideos = [
      '1.mp4',
      '2.mp4',
      '3.mp4'
    ];

    // Selected image path
    let selectedImage = billboardImages[0];
    let selectedVideo = sampleVideos[0];

    // DOM elements
    const form = document.getElementById('procForm');
    const progress = document.getElementById('progress');
    const resultVid = document.getElementById('resultVid');
    const maskAlphaSlider = document.getElementById('maskAlpha');
    const alphaValueSpan = document.getElementById('alphaValue');
    const confThreshSlider = document.getElementById('confThreshold');
    const threshValueSpan = document.getElementById('threshValue');
    const minMaskSizeSlider = document.getElementById('minMaskSize');
    const sizeValueSpan = document.getElementById('sizeValue');
    const humanConfThreshSlider = document.getElementById('humanConfThreshold');
    const humanThreshValueSpan = document.getElementById('humanThreshValue');
    const maskControls = document.getElementById('maskControls');
    const imageSelectPanel = document.getElementById('imageSelectPanel');
    const gallery = document.getElementById('thumbnailGallery');
    const previewImage = document.getElementById('previewImage');
    const videoGallery = document.getElementById('videoGallery');
    const sampleVideoContainer = document.getElementById('sampleVideoContainer');
    const uploadVideoContainer = document.getElementById('uploadVideoContainer');

    // Generate unique ID
    function generateUniqueId() {
      const timestamp = Date.now();
      const randomStr = Math.random().toString(36).substring(2, 10);
      const browserFingerprint = navigator.userAgent.split('').reduce((acc, char) => {
        return acc + char.charCodeAt(0);
      }, 0).toString(16).substring(0, 8);
      return `${timestamp}-${randomStr}-${browserFingerprint}`;
    }

    function setActiveJobId(jobId) {
      sessionStorage.setItem('activeJobId', jobId);
      window.activeJobId = jobId;
    }

    function getActiveJobId() {
      return window.activeJobId || sessionStorage.getItem('activeJobId');
    }

    // Initialize UI
    window.addEventListener('DOMContentLoaded', () => {
      sessionStorage.removeItem('activeJobId');
      window.activeJobId = null;
      window.serverJobId = null;

      if (window.progressPollInterval) {
        clearInterval(window.progressPollInterval);
      }

      if (!window.name) {
        window.name = 'browser-' + Date.now();
      }

      populateGallery();
      populateVideoGallery();

      const initialMode = document.querySelector('input[name="mode"]:checked').value;
      if (initialMode === 'mask') {
        maskControls.classList.remove('hide');
        imageSelectPanel.classList.add('hide');
      } else {
        maskControls.classList.add('hide');
        imageSelectPanel.classList.remove('hide');
      }

      const initialVideoSource = document.querySelector('input[name="videoSource"]:checked').value;
      if (initialVideoSource === 'sample') {
        sampleVideoContainer.classList.remove('hide');
        uploadVideoContainer.classList.add('hide');
        document.getElementById('video').required = false;
      } else {
        sampleVideoContainer.classList.add('hide');
        uploadVideoContainer.classList.remove('hide');
        document.getElementById('video').required = true;
      }

      progress.textContent = 'Ready to process video with AI technology';
      resultVid.style.display = 'none';

      const previousJobId = getActiveJobId();
      if (previousJobId) startProgressPolling(previousJobId);

      updateProcessButtonState(false);

      alphaValueSpan.textContent = maskAlphaSlider.value;
      threshValueSpan.textContent = confThreshSlider.value;
      sizeValueSpan.textContent = Math.round(minMaskSizeSlider.value * 100);
      humanThreshValueSpan.textContent = humanConfThreshSlider.value;
    });

    function populateGallery() {
      gallery.innerHTML = '';

      billboardImages.forEach((image, index) => {
        const thumbnail = document.createElement('div');
        thumbnail.className = 'thumbnail';
        if (index === 0) thumbnail.classList.add('selected');

        const img = document.createElement('img');
        img.src = `/images/${image}`;
        img.alt = image;

        img.onerror = function() {
          console.error(`Failed to load image: ${image}`);
          this.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="80" viewBox="0 0 100 80"%3E%3Crect width="100" height="80" fill="%23f0f0f0"/%3E%3Ctext x="50%" y="50%" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle" fill="%23999"%3EImage not found%3C/text%3E%3C/svg%3E';
        };

        thumbnail.appendChild(img);

        thumbnail.addEventListener('click', (function(imageName, thumbElement) {
          return function() {
            selectImage(imageName, thumbElement);
          };
        })(image, thumbnail));

        gallery.appendChild(thumbnail);
      });

      previewImage.src = `/images/${selectedImage}`;

      previewImage.onerror = function() {
        console.error(`Failed to load preview: ${selectedImage}`);
        this.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="150" viewBox="0 0 200 150"%3E%3Crect width="200" height="150" fill="%23f0f0f0"/%3E%3Ctext x="50%" y="50%" font-family="Arial" font-size="14" text-anchor="middle" dominant-baseline="middle" fill="%23999"%3EPreview not available%3C/text%3E%3C/svg%3E';
      };
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
        img.onerror = function() {
          console.error(`Thumbnail not found: ${thumbName}`);
          this.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="80" viewBox="0 0 100 80"%3E%3Crect width="100" height="80" fill="%23f0f0f0"/%3E%3Ctext x="50%" y="50%" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle" fill="%23999"%3ENo thumbnail%3C/text%3E%3C/svg%3E';
        };

        thumbnail.appendChild(img);

        thumbnail.addEventListener('click', () => {
          selectVideo(video, thumbnail);
        });

        videoGallery.appendChild(thumbnail);
      });
    }

    function selectImage(image, thumbnailElement) {
      console.log(`Selecting image: ${image}`);
      selectedImage = image;

      gallery.querySelectorAll('.thumbnail').forEach(thumb => {
        thumb.classList.remove('selected');
      });
      thumbnailElement.classList.add('selected');

      previewImage.src = `/images/${image}`;
    }

    function selectVideo(video, thumbnailElement) {
      console.log(`Selecting video: ${video}`);
      selectedVideo = video;

      videoGallery.querySelectorAll('.thumbnail').forEach(thumb => {
        thumb.classList.remove('selected');
      });
      thumbnailElement.classList.add('selected');
    }

    // Update input display values
    maskAlphaSlider.addEventListener('input', () => {
      alphaValueSpan.textContent = maskAlphaSlider.value;
    });

    confThreshSlider.addEventListener('input', () => {
      threshValueSpan.textContent = confThreshSlider.value;
    });

    minMaskSizeSlider.addEventListener('input', () => {
      sizeValueSpan.textContent = Math.round(minMaskSizeSlider.value * 100);
    });

    humanConfThreshSlider.addEventListener('input', () => {
      humanThreshValueSpan.textContent = humanConfThreshSlider.value;
    });

    // Mode switching
    document.querySelectorAll('input[name="mode"]').forEach(radio => {
      radio.addEventListener('change', e => {
        if (e.target.value === 'mask') {
          maskControls.classList.remove('hide');
          imageSelectPanel.classList.add('hide');
        } else {
          maskControls.classList.add('hide');
          imageSelectPanel.classList.remove('hide');
        }
      });
    });

    // Video source switching
    document.querySelectorAll('input[name="videoSource"]').forEach(radio => {
      radio.addEventListener('change', e => {
        if (e.target.value === 'sample') {
          sampleVideoContainer.classList.remove('hide');
          uploadVideoContainer.classList.add('hide');
          document.getElementById('video').required = false;
        } else {
          sampleVideoContainer.classList.add('hide');
          uploadVideoContainer.classList.remove('hide');
          document.getElementById('video').required = true;
        }
      });
    });

    function updateProcessButtonState(isProcessing) {
      const processButton = document.querySelector('button[type="submit"]');
      if (isProcessing) {
        processButton.disabled = true;
        processButton.style.opacity = "0.5";
        processButton.style.cursor = "not-allowed";
      } else {
        processButton.disabled = false;
        processButton.style.opacity = "1";
        processButton.style.cursor = "pointer";
      }
    }

    // Form submission
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      progress.textContent = 'Uploading...';
      resultVid.style.display = 'none';

      if (window.progressPollInterval) {
        clearInterval(window.progressPollInterval);
      }

      const clientJobId = generateUniqueId();
      console.log(`Generated unique job ID: ${clientJobId}`);

      setActiveJobId(clientJobId);

      const formData = new FormData();
      const mode = form.querySelector('input[name="mode"]:checked').value;
      const videoSource = form.querySelector('input[name="videoSource"]:checked').value;

      formData.append('mode', mode);
      formData.append('client_job_id', clientJobId);
      formData.append('browser_session', window.name || 'browser-' + Date.now());

      // Add video
      if (videoSource === 'sample') {
        try {
          progress.textContent = `Loading sample video...`;
          const response = await fetch(`/sampleVideos/${selectedVideo}`);
          if (!response.ok) {
            throw new Error(`Failed to load video: ${response.statusText}`);
          }
          const blob = await response.blob();
          formData.append('video', new File([blob], selectedVideo, { type: 'video/mp4' }));
        } catch (error) {
          progress.textContent = `Error loading sample video: ${error.message}`;
          return;
        }
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

        const maxSize = 50 * 1024 * 1024;
        if (videoFile.size > maxSize) {
          progress.textContent = 'Error: Video file must be under 50MB';
          return;
        }

        formData.append('video', videoFile);
      }

      // Add parameters
      formData.append('conf_threshold', confThreshSlider.value);
      formData.append('min_mask_size', minMaskSizeSlider.value);
      formData.append('human_conf_threshold', humanConfThreshSlider.value);
      formData.append('enable_human_filter', document.getElementById('enableHumanFilter').checked);

      // Mode-specific parameters
      if (mode === 'mask') {
        const colorHex = document.getElementById('maskColor').value;
        const r = parseInt(colorHex.substr(1, 2), 16);
        const g = parseInt(colorHex.substr(3, 2), 16);
        const b = parseInt(colorHex.substr(5, 2), 16);

        formData.append('mask_color_r', r);
        formData.append('mask_color_g', g);
        formData.append('mask_color_b', b);
        formData.append('mask_alpha', maskAlphaSlider.value);
      } else {
        try {
          const response = await fetch(`/images/${selectedImage}`);
          const blob = await response.blob();
          formData.append('replacement', new File([blob], selectedImage, { type: 'image/jpeg' }));
        } catch (error) {
          progress.textContent = `Error loading image: ${error.message}`;
          return;
        }
      }

      updateProcessButtonState(true);

      try {
        progress.textContent = `Submitting video for processing...`;

        const resp = await fetch('/process', {
          method: 'POST',
          body: formData
        });

        if (!resp.ok) {
          progress.textContent = `Error: ${resp.statusText || 'Server error'}`;
          console.error('Server returned error:', resp.status, resp.statusText);
          updateProcessButtonState(false);
          return;
        }

        const data = await resp.json();
        console.log("Response data from server:", data);

        const jobId = data.job_id || clientJobId;
        setActiveJobId(jobId);
        startProgressPolling(jobId);

        // Progress polling
        let processingComplete = false;
        const progressInterval = setInterval(async () => {
          try {
            const response = await fetch('/process-status');
            if (!response.ok) {
              console.error("Error fetching status:", response.status);
              return;
            }

            const data = await response.json();
            const jobs = Object.values(data);
            let ourJob = jobs.find(job => job.job_id === jobId);

            if (ourJob) {
              const currentFrame = ourJob.current_frame || 0;
              const totalFrames = ourJob.total_frames || 1;
              const percent = Math.round((currentFrame / totalFrames) * 100);

              progress.textContent = `Processing: ${percent}%`;

              if (!ourJob.is_processing && !processingComplete) {
                processingComplete = true;
                clearInterval(progressInterval);
                progress.textContent = "Processing complete. Video is being prepared...";
              }
            }
          } catch (error) {
            console.error("Error checking progress:", error);
          }
        }, 1000);

        // Handle video display
        if (data.success && data.path) {
          let videoPath = data.path;
          if (videoPath.startsWith('/workspace/processed_videos/')) {
            videoPath = videoPath.replace('/workspace/processed_videos/', '/processed_videos/');
          }

          const pathsToTry = [
            videoPath,
            videoPath.replace("/videos/", "/processed_videos/"),
            "/processed_videos/" + videoPath.split('/').pop(),
            "/videos/" + videoPath.split('/').pop(),
            `/videos/web_${videoPath.split('/').pop()}`
          ];

          resultVid.src = videoPath;
          resultVid.onerror = function() {
            console.error("Direct video path failed, trying alternatives");
            tryPathsRecursively(0);
          };

          resultVid.onloadeddata = function() {
            console.log("Video loaded successfully!");
            progress.textContent = 'Video ready!';
            resultVid.style.display = 'block';
            updateProcessButtonState(false);
          };

          resultVid.load();

          function tryPathsRecursively(index) {
            if (index >= pathsToTry.length) {
              console.error("All video paths failed");
              progress.textContent = "Could not load the video. Please try again.";
              updateProcessButtonState(false);
              return;
            }

            const path = pathsToTry[index];
            console.log(`Trying alternative path ${index+1}/${pathsToTry.length}: ${path}`);

            resultVid.src = path;
            resultVid.onerror = function() {
              tryPathsRecursively(index + 1);
            };

            resultVid.onloadeddata = function() {
              console.log(`Video loaded from alternative path: ${path}`);
              progress.textContent = 'Video ready!';
              resultVid.style.display = 'block';
              updateProcessButtonState(false);
            };

            resultVid.load();
          }
        }
      } catch (error) {
        progress.textContent = `Error: ${error.message || 'Connection failed'}`;
        console.error('Processing error:', error);
        updateProcessButtonState(false);
      }
    });

    function startProgressPolling(jobId) {
      if (window.progressPollInterval) clearInterval(window.progressPollInterval);

      window.progressPollInterval = setInterval(() => {
        fetch('/process-status')
          .then(response => response.json())
          .then(data => {
            let job = data[jobId] || Object.values(data).find(j => (j.job_id == jobId));
            if (job) {
              const currentFrame = job.current_frame || 0;
              const totalFrames = job.total_frames || 1;
              const percent = Math.round((currentFrame / totalFrames) * 100);
              if (job.is_processing) {
                progress.textContent = `Processing: ${percent}%`;
              } else if (job.video_ready && job.final_output_filename) {
                clearInterval(window.progressPollInterval);

                const videoPath = '/videos/' + job.final_output_filename;
                resultVid.src = videoPath;
                resultVid.style.display = 'block';
                resultVid.load();
                progress.textContent = 'Video ready!';
                updateProcessButtonState(false);
              } else {
                progress.textContent = 'Processing complete. Video is being prepared...';
              }
            } else {
              progress.textContent = 'Processing job not found or finished.';
            }
          })
          .catch(error => {
            progress.textContent = `Error fetching progress: ${error.message}`;
          });
      }, 1000);
    }

    // Analytics
    console.log('Altervision AI Billboard Replacement Tool Loaded');

    window.addEventListener('load', () => {
      console.log('Demo page loaded - tracking user engagement');
    });