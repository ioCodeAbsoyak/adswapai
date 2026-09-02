const container = document.getElementById('slider-container');
const sliderBar = document.getElementById('slider-bar');
const clipDiv = document.getElementById('clip-div');

// Drag logic (mouse and touch)
let isDragging = false;

function moveSlider(x) {
  const rect = container.getBoundingClientRect();
  let offsetX = x - rect.left;
  
  // Clamp to the container
  if (offsetX < 0) offsetX = 0;
  if (offsetX > rect.width) offsetX = rect.width;
  
  // Position as a percentage
  const percentage = (offsetX / rect.width) * 100;
  const rightPercentage = 100 - percentage;
  
  // Clip the top video with clip-path (no squeezing)
  clipDiv.style.clipPath = `inset(0 ${rightPercentage}% 0 0)`;
  
  // Slider bar pozisyonu
  sliderBar.style.left = offsetX + "px";
}

// Mouse events
sliderBar.addEventListener('mousedown', e => { 
  isDragging = true; 
  e.preventDefault();
});

document.addEventListener('mouseup', e => { 
  isDragging = false; 
});

document.addEventListener('mousemove', e => {
  if (!isDragging) return;
  moveSlider(e.clientX);
});

// Clicking the container jumps straight to that point
container.addEventListener('mousedown', e => {
  if (e.target === sliderBar || e.target.closest('.player-icons')) return;
  moveSlider(e.clientX);
  isDragging = true;
  e.preventDefault();
});

// Touch events
sliderBar.addEventListener('touchstart', e => { 
  isDragging = true; 
  e.preventDefault();
});

document.addEventListener('touchend', e => { 
  isDragging = false; 
});

document.addEventListener('touchmove', e => {
  if (!isDragging) return;
  if (e.touches.length > 0) {
    moveSlider(e.touches[0].clientX);
  }
  e.preventDefault();
});

container.addEventListener('touchstart', e => {
  if (e.target === sliderBar || e.target.closest('.player-icons')) return;
  if (e.touches.length > 0) {
    moveSlider(e.touches[0].clientX);
    isDragging = true;
  }
  e.preventDefault();
});

// Fullscreen
document.getElementById('fullscreen-btn').onclick = function() {
  if (!document.fullscreenElement) {
    container.requestFullscreen().catch(err => {
      console.log('Fullscreen error:', err);
    });
  } else {
    document.exitFullscreen();
  }
};

// Miniplayer (Picture-in-Picture API)
document.getElementById('miniplayer-btn').onclick = function() {
  const mainVideo = container.querySelector('.video');
  if (mainVideo.requestPictureInPicture) {
    mainVideo.requestPictureInPicture().catch(err => {
      console.log('PiP error:', err);
    });
  } else {
    alert('Picture-in-Picture not supported in this browser');
  }
};

// Keep the two videos in sync
const video1 = container.querySelector('.video');
const video2 = container.querySelector('.video-top');

// Sync the second video to the first one
video1.addEventListener('timeupdate', () => {
  if (Math.abs(video1.currentTime - video2.currentTime) > 0.1) {
    video2.currentTime = video1.currentTime;
  }
});

// Smooth scroll for CTA button
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});