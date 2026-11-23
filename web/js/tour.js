/**
 * ZERO-TEXT: INTERACTIVE PRESENTATION & DEMO TOUR SYSTEM (tour.js)
 *
 * Designed for evaluations, client demos, mentor walkthroughs, and presentations.
 * Features:
 * - Dynamic spotlight backdrop with glowing target outline
 * - Smooth step-by-step guidance across all major features:
 *   1. System Overview & Live Telemetry
 *   2. 3D WebGL Celestial Galaxy (Orbit, Constellations, Fly-To)
 *   3. 2D UMAP Semantic Star Map (Dimensionality Reduction)
 *   4. Card Gallery Grid (High-speed browsing)
 *   5. Multimodal Search Hub (Natural language text & OCR)
 *   6. Dynamic Smart Vocab Suggestions
 *   7. AI Auto-Weight Decider & Tri-Modal Sliders (CLIP, OCR, Color)
 *   8. Interactive Sketch Search & Multi-Color Canvas
 *   9. Drag-and-Drop Image Probe Telemetry
 *   10. Offline INT8 Turbo Mode (Faster inference)
 *   11. Local System Folder Ingestion & Browse
 *   12. Built-in Benchmark & Performance Analytics
 *   13. Secret Computer-Vision AR Arcade Bay
 * - Live interactive actions (auto switches tabs, triggers views, opens modals on cue)
 * - Keyboard navigation (Arrow keys, Space, Escape)
 * - Auto-play presenter mode with adjustable countdown timer
 * - Interactive step jump list for instant access to any feature during Q&A
 */

(function () {
  'use strict';

  const TOUR_STEPS = [
    {
      id: 'welcome',
      title: 'Welcome to ZERO-TEXT 2.0',
      tag: 'ARCHITECTURE & OVERVIEW',
      description:
        'ZERO-TEXT is an end-to-end multimodal search & visual discovery engine. It unifies <b>OpenCLIP visual embeddings</b>, <b>EasyOCR in-image text detection</b>, <b>K-Means dominant color extraction</b>, and <b>HDBSCAN / UMAP clustering</b> into a 3D celestial galaxy.',
      target: '.brand',
      placement: 'bottom',
      action: () => {
        const tab = document.getElementById('tab-galaxy3d');
        if (tab) tab.click();
      }
    },
    {
      id: 'telemetry',
      title: 'Live Catalog Telemetry',
      tag: 'DATA ENGINE',
      description:
        'Real-time status badges tracking total indexed media corpus, unsupervised clusters formed via HDBSCAN density clustering, and images containing recognized OCR text.',
      target: '.header-stats',
      placement: 'bottom'
    },
    {
      id: 'galaxy3d',
      title: '3D Celestial Visual Galaxy',
      tag: '3D WEBGL MANIFOLD',
      description:
        'High-dimensional 512-D CLIP embeddings projected into interactive 3D space using <b>stratified UMAP</b>. Images in the same semantic cluster share glowing constellation lines. You can rotate, pan, and zoom freely with your mouse.',
      target: '#view-galaxy3d',
      placement: 'center',
      action: () => {
        const tab = document.getElementById('tab-galaxy3d');
        if (tab) tab.click();
      }
    },
    {
      id: 'galaxy3d_controls',
      title: 'Galaxy Controls & Constellations',
      tag: 'VISUAL INTERACTION',
      description:
        'Toggle celestial orbit spin, show/hide inter-cluster constellation mesh lines, or reset the camera orientation. When searching, the camera automatically flies right in front of the top semantic matches.',
      target: '.galaxy-hud',
      placement: 'top'
    },
    {
      id: 'galaxy2d',
      title: '2D Semantic Star Map',
      tag: 'PLANAR EMBEDDING',
      description:
        'A continuous planar 2D manifold ideal for spatial cartography of your image library. Clusters form distinct semantic islands with smooth zoom, pan, and hover inspections.',
      target: '#tab-galaxy2d',
      placement: 'bottom',
      action: () => {
        const tab = document.getElementById('tab-galaxy2d');
        if (tab) tab.click();
      }
    },
    {
      id: 'grid_view',
      title: 'Card Gallery Grid',
      tag: 'FAST CATALOGING',
      description:
        'A high-density gallery view displaying full image cards with cluster tags, extracted dominant color swatches, and relevance score badges for rapid bulk reviewing.',
      target: '#tab-grid',
      placement: 'bottom',
      action: () => {
        const tab = document.getElementById('tab-grid');
        if (tab) tab.click();
      }
    },
    {
      id: 'search_hub',
      title: 'Multimodal Hybrid Search Hub',
      tag: 'QUERY PROCESSING',
      description:
        'Search across visual concepts, exact text inside photos, or colors (e.g., <i>"red sports car"</i>, <i>"neural network architecture"</i>, <i>"handwritten receipt"</i>). Press Enter to execute a hybrid search with sub-millisecond FAISS vector retrieval.',
      target: '.search-box-wrap',
      placement: 'bottom',
      action: () => {
        const tab = document.getElementById('tab-galaxy3d');
        if (tab) tab.click();
        const input = document.getElementById('search-input');
        if (input && !input.value) {
          input.placeholder = "Try 'purple nebula', 'pizza', 'receipt', 'vintage camera'...";
        }
      }
    },
    {
      id: 'smart_vocab',
      title: 'Dynamic Smart Vocab Chips',
      tag: 'DATASET INSIGHTS',
      description:
        'Automatically extracted dataset vocabulary tags. Clicking any chip instantly triggers a precision search for concepts known to exist within your indexed images.',
      target: '#smart-vocab-bar',
      placement: 'bottom'
    },
    {
      id: 'ai_decider',
      title: 'AI Weight Decider & Presets',
      tag: 'INTELLIGENT FUSION',
      description:
        'The intelligent query decider analyzes user search intent in real time. Queries asking for text bias toward OCR; visual descriptions bias toward CLIP; color words boost color harmony.',
      target: '#auto-decider-bar',
      placement: 'bottom'
    },
    {
      id: 'weights_drawer',
      title: 'Tri-Modal Weight Sliders',
      tag: 'FINE-GRAINED TUNING',
      description:
        'Fine-tune the mathematical balance between <b>CLIP semantic similarity</b>, <b>OCR text matching</b>, and <b>Lab-space color distance</b> to tailor results precisely to user needs.',
      target: '#weights-toggle-btn',
      placement: 'bottom',
      action: () => {
        const drawer = document.getElementById('weights-drawer');
        if (drawer && !drawer.classList.contains('open')) {
          const btn = document.getElementById('weights-toggle-btn');
          if (btn) btn.click();
        }
      }
    },
    {
      id: 'sketch_search',
      title: 'Sketch Search & Multimodal Canvas',
      tag: 'REVERSE IMAGE SEARCH',
      description:
        'Draw a rough sketch with custom brush sizes, color palette, or flood fill. A specialized projection head maps line drawings into CLIP embedding space to find matching real photos!',
      target: '#btn-open-sketch',
      placement: 'bottom',
      action: () => {
        const modal = document.getElementById('modal-sketch');
        const openBtn = document.getElementById('btn-open-sketch');
        if (modal && modal.style.display !== 'flex' && openBtn) {
          openBtn.click();
        }
      }
    },
    {
      id: 'probe_drop',
      title: 'Drag-and-Drop Image Probe',
      tag: 'MANIFOLD PROJECTION',
      description:
        'Drag any image directly into the galaxy viewport! The live test probe projects the image into the 3D space, highlights its nearest semantic neighbors, and outputs live cosine similarity telemetry.',
      target: '#view-galaxy3d',
      placement: 'center',
      action: () => {
        const closeSketch = document.getElementById('close-sketch-btn');
        if (closeSketch) closeSketch.click();
        const tab = document.getElementById('tab-galaxy3d');
        if (tab) tab.click();
      }
    },
    {
      id: 'add_ingestion',
      title: 'Local Folder & File Ingestion',
      tag: 'OFFLINE PIPELINE',
      description:
        'Index files locally or browse any directory using the native system folder picker. Zero cloud dependency—all embeddings, OCR, and FAISS indexing occur entirely on your local machine.',
      target: '#btn-open-add',
      placement: 'bottom',
      action: () => {
        const openBtn = document.getElementById('btn-open-add');
        if (openBtn) openBtn.click();
      }
    },
    {
      id: 'benchmarks',
      title: 'System Benchmarks & 100k Scaling',
      tag: 'PERFORMANCE RIG',
      description:
        'Quantitative latency benchmarks proving <b>sub-15ms FAISS search across 100,000 images</b>, <b>1.79× INT8 Turbo quantization speedup</b>, and complete PDF report generation for mentors and evaluations.',
      target: '#btn-tour-benchmark',
      placement: 'bottom',
      action: () => {
        const closeAdd = document.getElementById('close-add-btn');
        if (closeAdd) closeAdd.click();
        const modal = document.getElementById('modal-benchmark');
        if (modal) {
          const btn = document.getElementById('btn-tour-benchmark');
          if (btn) btn.click();
        }
      }
    },
    {
      id: 'arcade_portal',
      title: 'Shadowbox AR Kinetic Arcade',
      tag: 'BONUS AR INNOVATION',
      description:
        'A zero-controller computer-vision gaming suite powered by 60 FPS BlazePose and Kiai acoustic battle cries (Shadow Boxing AR & Laser Blade Fruit Ninja). Type <b>"SAM"</b> anywhere in the app to unlock the Arcade bay!',
      target: '.brand-badge',
      placement: 'bottom'
    },
    {
      id: 'finish',
      title: 'Ready to Explore & Present!',
      tag: 'DEMO READY',
      description:
        'You are now equipped with the full tour. You can replay this tour anytime with the <b>"🎓 Demo Tour"</b> button in the header, or click any step in the step navigator during your presentation.',
      target: '.brand',
      placement: 'bottom'
    }
  ];

  class DemoTour {
    constructor() {
      this.currentStep = 0;
      this.isPlaying = false;
      this.playInterval = null;
      this.playSpeed = 7000; // 7 seconds per slide in autoplay
      this.timerCountdown = null;
      this.timeLeft = 7;
      this.isOpen = false;

      this.initDom();
      this.bindEvents();
    }

    initDom() {
      // Backdrop overlay
      this.overlay = document.createElement('div');
      this.overlay.className = 'tour-backdrop';
      this.overlay.id = 'tour-backdrop';

      // Spotlight cutout
      this.spotlight = document.createElement('div');
      this.spotlight.className = 'tour-spotlight';
      this.spotlight.id = 'tour-spotlight';

      // Modal dialog
      this.card = document.createElement('div');
      this.card.className = 'tour-card';
      this.card.id = 'tour-card';

      this.card.innerHTML = `
        <div class="tour-header">
          <div class="tour-tag" id="tour-tag">OVERVIEW</div>
          <button class="tour-close-btn" id="tour-close-btn" title="End Tour (ESC)">✕</button>
        </div>
        <h3 class="tour-title" id="tour-title">Tour Step</h3>
        <p class="tour-description" id="tour-description">Step details...</p>
        
        <div class="tour-stepper" id="tour-stepper"></div>

        <div class="tour-footer">
          <div class="tour-left-controls">
            <button class="tour-btn tour-btn-secondary" id="tour-autoplay-btn" title="Auto-advance slides">
              <span id="tour-play-icon">▶</span> <span id="tour-play-text">Auto Play</span> <span id="tour-timer-badge" class="tour-timer-badge" style="display:none">7s</span>
            </button>
            <button class="tour-btn tour-btn-secondary" id="tour-menu-btn" title="Jump to any feature">
              📑 Topics
            </button>
          </div>
          <div class="tour-right-controls">
            <button class="tour-btn tour-btn-secondary" id="tour-prev-btn">◀ Back</button>
            <button class="tour-btn tour-btn-primary" id="tour-next-btn">Next ▶</button>
          </div>
        </div>

        <!-- Jump menu drawer -->
        <div class="tour-jump-menu" id="tour-jump-menu" style="display: none;">
          <div class="tour-jump-header">
            <strong>Presentation Chapters</strong>
            <button class="tour-jump-close" id="tour-jump-close">✕</button>
          </div>
          <div class="tour-jump-list" id="tour-jump-list"></div>
        </div>
      `;

      document.body.appendChild(this.overlay);
      document.body.appendChild(this.spotlight);
      document.body.appendChild(this.card);

      // Cache elements
      this.elTag = this.card.querySelector('#tour-tag');
      this.elTitle = this.card.querySelector('#tour-title');
      this.elDesc = this.card.querySelector('#tour-description');
      this.elStepper = this.card.querySelector('#tour-stepper');
      this.elPrev = this.card.querySelector('#tour-prev-btn');
      this.elNext = this.card.querySelector('#tour-next-btn');
      this.elPlay = this.card.querySelector('#tour-autoplay-btn');
      this.elPlayIcon = this.card.querySelector('#tour-play-icon');
      this.elPlayText = this.card.querySelector('#tour-play-text');
      this.elTimerBadge = this.card.querySelector('#tour-timer-badge');
      this.elMenu = this.card.querySelector('#tour-menu-btn');
      this.elJumpMenu = this.card.querySelector('#tour-jump-menu');
      this.elJumpClose = this.card.querySelector('#tour-jump-close');
      this.elJumpList = this.card.querySelector('#tour-jump-list');
      this.elClose = this.card.querySelector('#tour-close-btn');

      this.buildStepper();
      this.buildJumpList();
    }

    buildStepper() {
      this.elStepper.innerHTML = '';
      TOUR_STEPS.forEach((step, idx) => {
        const dot = document.createElement('button');
        dot.className = 'tour-dot' + (idx === this.currentStep ? ' active' : '');
        dot.title = `Step ${idx + 1}: ${step.title}`;
        dot.addEventListener('click', () => {
          this.pauseAutoPlay();
          this.goToStep(idx);
        });
        this.elStepper.appendChild(dot);
      });
    }

    buildJumpList() {
      this.elJumpList.innerHTML = '';
      TOUR_STEPS.forEach((step, idx) => {
        const item = document.createElement('div');
        item.className = 'tour-jump-item' + (idx === this.currentStep ? ' active' : '');
        item.innerHTML = `
          <span class="jump-num">${idx + 1}</span>
          <div class="jump-info">
            <strong class="jump-title">${step.title}</strong>
            <span class="jump-tag">${step.tag}</span>
          </div>
        `;
        item.addEventListener('click', () => {
          this.pauseAutoPlay();
          this.goToStep(idx);
          this.elJumpMenu.style.display = 'none';
        });
        this.elJumpList.appendChild(item);
      });
    }

    bindEvents() {
      this.elPrev.addEventListener('click', () => {
        this.pauseAutoPlay();
        this.prevStep();
      });

      this.elNext.addEventListener('click', () => {
        this.pauseAutoPlay();
        this.nextStep();
      });

      this.elPlay.addEventListener('click', () => {
        this.toggleAutoPlay();
      });

      this.elMenu.addEventListener('click', () => {
        const isHidden = this.elJumpMenu.style.display === 'none';
        this.elJumpMenu.style.display = isHidden ? 'flex' : 'none';
      });

      this.elJumpClose.addEventListener('click', () => {
        this.elJumpMenu.style.display = 'none';
      });

      this.elClose.addEventListener('click', () => {
        this.stopTour();
      });

      this.overlay.addEventListener('click', () => {
        // Clicking backdrop closes menu if open, or advances
        if (this.elJumpMenu.style.display !== 'none') {
          this.elJumpMenu.style.display = 'none';
        }
      });

      // Keyboard navigation
      window.addEventListener('keydown', (e) => {
        if (!this.isOpen) return;

        if (e.key === 'Escape') {
          this.stopTour();
        } else if (e.key === 'ArrowRight' || e.key === 'PageDown') {
          e.preventDefault();
          this.pauseAutoPlay();
          this.nextStep();
        } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
          e.preventDefault();
          this.pauseAutoPlay();
          this.prevStep();
        } else if (e.key === ' ' && e.target.tagName !== 'INPUT') {
          e.preventDefault();
          this.toggleAutoPlay();
        }
      });

      // Window resize re-positions spotlight
      window.addEventListener('resize', () => {
        if (this.isOpen) {
          this.positionTarget();
        }
      });
    }

    startTour() {
      this.isOpen = true;
      document.body.classList.add('tour-active');
      this.overlay.classList.add('visible');
      this.spotlight.classList.add('visible');
      this.card.classList.add('visible');
      this.currentStep = 0;
      this.goToStep(0);
    }

    stopTour() {
      this.isOpen = false;
      this.pauseAutoPlay();
      document.body.classList.remove('tour-active');
      this.overlay.classList.remove('visible');
      this.spotlight.classList.remove('visible');
      this.card.classList.remove('visible');
      this.elJumpMenu.style.display = 'none';

      const modalBenchmark = document.getElementById('modal-benchmark');
      if (modalBenchmark) modalBenchmark.style.display = 'none';

      // Remove highlight from any active element
      document.querySelectorAll('.tour-highlight-active').forEach((el) => {
        el.classList.remove('tour-highlight-active');
      });
    }

    goToStep(index) {
      if (index < 0 || index >= TOUR_STEPS.length) return;
      this.currentStep = index;
      const step = TOUR_STEPS[this.currentStep];

      // Execute custom action if configured
      if (typeof step.action === 'function') {
        try {
          step.action();
        } catch (err) {
          console.warn('[Tour] Step action error:', err);
        }
      }

      // Update content
      this.elTag.textContent = step.tag || 'FEATURE';
      this.elTitle.textContent = step.title;
      this.elDesc.innerHTML = step.description;

      // Update buttons
      this.elPrev.disabled = this.currentStep === 0;
      if (this.currentStep === TOUR_STEPS.length - 1) {
        this.elNext.textContent = 'Finish Tour ✓';
      } else {
        this.elNext.textContent = 'Next ▶';
      }

      // Update stepper & jump list
      const dots = this.elStepper.querySelectorAll('.tour-dot');
      dots.forEach((dot, idx) => {
        dot.classList.toggle('active', idx === this.currentStep);
      });

      const items = this.elJumpList.querySelectorAll('.tour-jump-item');
      items.forEach((item, idx) => {
        item.classList.toggle('active', idx === this.currentStep);
      });

      // Position spotlight and card after DOM adjusts
      setTimeout(() => {
        this.positionTarget();
      }, 120);

      // Reset autoplay countdown if active
      if (this.isPlaying) {
        this.resetCountdown();
      }
    }

    nextStep() {
      if (this.currentStep < TOUR_STEPS.length - 1) {
        this.goToStep(this.currentStep + 1);
      } else {
        this.stopTour();
      }
    }

    prevStep() {
      if (this.currentStep > 0) {
        this.goToStep(this.currentStep - 1);
      }
    }

    toggleAutoPlay() {
      if (this.isPlaying) {
        this.pauseAutoPlay();
      } else {
        this.startAutoPlay();
      }
    }

    startAutoPlay() {
      this.isPlaying = true;
      this.elPlayIcon.textContent = '⏸';
      this.elPlayText.textContent = 'Pause';
      this.elTimerBadge.style.display = 'inline-block';
      this.resetCountdown();
    }

    pauseAutoPlay() {
      this.isPlaying = false;
      this.elPlayIcon.textContent = '▶';
      this.elPlayText.textContent = 'Auto Play';
      this.elTimerBadge.style.display = 'none';
      if (this.timerCountdown) {
        clearInterval(this.timerCountdown);
        this.timerCountdown = null;
      }
    }

    resetCountdown() {
      if (this.timerCountdown) clearInterval(this.timerCountdown);
      this.timeLeft = Math.round(this.playSpeed / 1000);
      this.elTimerBadge.textContent = `${this.timeLeft}s`;

      this.timerCountdown = setInterval(() => {
        this.timeLeft--;
        if (this.timeLeft <= 0) {
          if (this.currentStep >= TOUR_STEPS.length - 1) {
            this.pauseAutoPlay();
          } else {
            this.nextStep();
          }
        } else {
          this.elTimerBadge.textContent = `${this.timeLeft}s`;
        }
      }, 1000);
    }

    positionTarget() {
      const step = TOUR_STEPS[this.currentStep];
      let targetEl = null;

      if (step.target) {
        targetEl = document.querySelector(step.target);
      }

      // Clear previous active highlight
      document.querySelectorAll('.tour-highlight-active').forEach((el) => {
        el.classList.remove('tour-highlight-active');
      });

      if (targetEl && step.placement !== 'center') {
        targetEl.classList.add('tour-highlight-active');
        const rect = targetEl.getBoundingClientRect();

        const padding = 8;
        const top = Math.max(0, rect.top - padding);
        const left = Math.max(0, rect.left - padding);
        const width = rect.width + padding * 2;
        const height = rect.height + padding * 2;

        this.spotlight.style.display = 'block';
        this.spotlight.style.top = `${top}px`;
        this.spotlight.style.left = `${left}px`;
        this.spotlight.style.width = `${width}px`;
        this.spotlight.style.height = `${height}px`;
        this.spotlight.style.borderRadius = '12px';

        // Position Card
        this.positionCardRelative(rect, step.placement);
      } else {
        // Center placement or fallback
        this.spotlight.style.display = 'none';
        this.card.style.top = '50%';
        this.card.style.left = '50%';
        this.card.style.transform = 'translate(-50%, -50%)';
      }
    }

    positionCardRelative(targetRect, placement) {
      const cardWidth = Math.min(460, window.innerWidth - 32);
      this.card.style.width = `${cardWidth}px`;
      this.card.style.transform = 'none';

      const gap = 16;
      let top = 0;
      let left = 0;

      if (placement === 'bottom') {
        top = targetRect.bottom + gap;
        left = targetRect.left + targetRect.width / 2 - cardWidth / 2;
      } else if (placement === 'top') {
        top = targetRect.top - this.card.offsetHeight - gap;
        left = targetRect.left + targetRect.width / 2 - cardWidth / 2;
      } else if (placement === 'left') {
        top = targetRect.top;
        left = targetRect.left - cardWidth - gap;
      } else if (placement === 'right') {
        top = targetRect.top;
        left = targetRect.right + gap;
      } else {
        top = targetRect.bottom + gap;
        left = targetRect.left;
      }

      // Constrain within viewport bounds
      const maxLeft = window.innerWidth - cardWidth - 16;
      left = Math.max(16, Math.min(left, maxLeft));

      const cardHeight = this.card.offsetHeight || 320;
      const maxTop = window.innerHeight - cardHeight - 16;

      if (top > maxTop) {
        // Flip to above target if it spills off bottom
        top = Math.max(16, targetRect.top - cardHeight - gap);
      }

      this.card.style.top = `${Math.max(16, top)}px`;
      this.card.style.left = `${left}px`;
    }
  }

  // Export tour instance globally
  window.demoTour = new DemoTour();
})();
