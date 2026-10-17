const samples = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];
const labels = {
  "1": "T2P",
  "2": "EMPMP",
  "3": "SimpliHuMoN wide",
  "4": "SimpliHuMoN deep",
};

let activeSample = samples[0];
let videosPlaying = true;

function mediaPath(sample, model) {
  return `assets/media/sample_${sample}_model_${model}.mp4`;
}

function loadSample(sample) {
  activeSample = sample;
  const videos = document.querySelectorAll(".video-open video");
  videos.forEach((video) => {
    const model = video.closest(".video-open").dataset.model;
    video.src = mediaPath(sample, model);
    video.load();
    if (videosPlaying) {
      video.play().catch(() => {});
    }
  });

  document.querySelectorAll(".sample-button").forEach((button) => {
    const selected = button.dataset.sample === sample;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function setPlayback(playing) {
  videosPlaying = playing;
  document.querySelectorAll(".video-open video").forEach((video) => {
    if (playing) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  });

  const toggle = document.getElementById("play-toggle");
  toggle.innerHTML = `<i data-lucide="${playing ? "pause" : "play"}" aria-hidden="true"></i>`;
  toggle.title = playing ? "Pause all videos" : "Play all videos";
  toggle.setAttribute("aria-label", toggle.title);
  window.lucide?.createIcons();
}

function synchronizeVideos() {
  const videos = [...document.querySelectorAll(".video-open video")];
  const reference = videos.find((video) => video.readyState >= 2);
  if (!reference) return;

  videos.forEach((video) => {
    if (video !== reference && video.readyState >= 2 && Math.abs(video.currentTime - reference.currentTime) > 0.12) {
      video.currentTime = reference.currentTime;
    }
  });
}

function openVideo(model) {
  const dialog = document.getElementById("video-dialog");
  const video = document.getElementById("dialog-video");
  document.getElementById("dialog-title").textContent = labels[model];
  video.src = mediaPath(activeSample, model);
  dialog.showModal();
  video.play().catch(() => {});
}

function closeVideo() {
  const dialog = document.getElementById("video-dialog");
  const video = document.getElementById("dialog-video");
  video.pause();
  video.removeAttribute("src");
  video.load();
  dialog.close();
}

document.addEventListener("DOMContentLoaded", () => {
  window.lucide?.createIcons();
  loadSample(activeSample);
  window.setInterval(synchronizeVideos, 250);

  document.querySelectorAll(".sample-button").forEach((button) => {
    button.addEventListener("click", () => loadSample(button.dataset.sample));
  });

  document.getElementById("play-toggle").addEventListener("click", () => {
    setPlayback(!videosPlaying);
  });

  document.querySelectorAll(".video-open").forEach((button) => {
    button.addEventListener("click", () => openVideo(button.dataset.model));
  });

  const dialog = document.getElementById("video-dialog");
  document.getElementById("dialog-close").addEventListener("click", closeVideo);
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) closeVideo();
  });

  document.getElementById("copy-citation").addEventListener("click", async () => {
    const citation = document.getElementById("bibtex").textContent;
    await navigator.clipboard.writeText(citation);
    const button = document.getElementById("copy-citation");
    button.innerHTML = '<i data-lucide="check" aria-hidden="true"></i>';
    button.title = "Copied";
    window.lucide?.createIcons();
    window.setTimeout(() => {
      button.innerHTML = '<i data-lucide="copy" aria-hidden="true"></i>';
      button.title = "Copy BibTeX";
      window.lucide?.createIcons();
    }, 1600);
  });
});
