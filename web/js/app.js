/**
 * ZERO-TEXT: MAIN APPLICATION CONTROLLER (v2.5)
 * Coordinates:
 * - 3D Visual Galaxy with Camera Fly-To & Search Spotlight
 * - 2D Semantic Star Map with Smooth Centering
 * - Card Gallery Grid
 * - Intelligent AI Weight Decider (CLIP vs OCR vs Dominant Colour)
 * - 1-Click Scoring Presets
 * - Full Galaxy / Search Bar Visibility Toggle
 * - Rapid Ingestion & Directory Ingestion Pipeline
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    allImages: [],
    displayedImages: [],
    selectedItem: null,
    activeView: 'galaxy3d', // 'galaxy3d' | 'galaxy2d' | 'grid'
    activeClusterFilter: null,
    autoWeights: true,
    weights: {
      clip: 0.50,
      ocr: 0.25,
      colour: 0.25,
    },
    currentMatchIndex: 0,
    searchDebounceTimer: null,
  };

  // UI Elements
  const el = {
    // Stats
    statTotal: document.getElementById('stat-total'),
    statClusters: document.getElementById('stat-clusters'),
    statText: document.getElementById('stat-text'),

    // Loading banner
    loadingBanner: document.getElementById('loading-banner'),
    loadingBannerMsg: document.getElementById('loading-banner-msg'),

    // Full Galaxy toggle
    btnToggleSearchBar: document.getElementById('btn-toggle-search-bar'),
    searchHub: document.querySelector('.search-hub'),

    // View tabs
    tabGalaxy3d: document.getElementById('tab-galaxy3d'),
    tabGalaxy2d: document.getElementById('tab-galaxy2d'),
    tabGrid: document.getElementById('tab-grid'),
    viewGalaxy3d: document.getElementById('view-galaxy3d'),
    viewGalaxy2d: document.getElementById('view-galaxy2d'),
    viewGrid: document.getElementById('view-grid'),

    // Search
    searchInput: document.getElementById('search-input'),
    searchBtn: document.getElementById('search-btn'),
    searchClearBtn: document.getElementById('search-clear-btn'),
    weightsToggleBtn: document.getElementById('weights-toggle-btn'),
    weightsDrawer: document.getElementById('weights-drawer'),
    weightClipSlider: document.getElementById('weight-clip'),
    weightOcrSlider: document.getElementById('weight-ocr'),
    weightColourSlider: document.getElementById('weight-colour'),
    weightClipVal: document.getElementById('val-clip'),
    weightOcrVal: document.getElementById('val-ocr'),
    weightColourVal: document.getElementById('val-colour'),
    clusterPillsContainer: document.getElementById('cluster-pills'),
    queryInsights: document.getElementById('query-insights'),

    // AI Decider & Presets
    deciderLabel: document.getElementById('decider-label'),
    deciderReason: document.getElementById('decider-reason'),
    presetPills: document.querySelectorAll('.preset-pill'),

    // 3D & 2D Spotlight HUDs
    spotlight3dHud: document.getElementById('spotlight-3d-hud'),
    spotlight3dTitle: document.getElementById('spotlight-3d-title'),
    spotlight3dScore: document.getElementById('spotlight-3d-score'),
    spotlight3dPrev: document.getElementById('spotlight-3d-prev'),
    spotlight3dNext: document.getElementById('spotlight-3d-next'),

    spotlight2dHud: document.getElementById('spotlight-2d-hud'),
    spotlight2dTitle: document.getElementById('spotlight-2d-title'),
    spotlight2dScore: document.getElementById('spotlight-2d-score'),
    spotlight2dNext: document.getElementById('spotlight-2d-next'),

    // Grid
    gridWrapper: document.getElementById('grid-wrapper'),

    // Tooltip
    tooltip: document.getElementById('galaxy-tooltip'),
    tooltipThumb: document.getElementById('tooltip-thumb'),
    tooltipTitle: document.getElementById('tooltip-title'),
    tooltipCluster: document.getElementById('tooltip-cluster'),
    tooltipScore: document.getElementById('tooltip-score'),

    // Inspector
    drawer: document.getElementById('inspector-drawer'),
    drawerCloseBtn: document.getElementById('drawer-close-btn'),
    drawerTitle: document.getElementById('drawer-title'),
    drawerPreview: document.getElementById('drawer-preview'),
    drawerScoreFinal: document.getElementById('score-final'),
    drawerScoreClip: document.getElementById('score-clip'),
    drawerScoreOcr: document.getElementById('score-ocr'),
    drawerScoreColor: document.getElementById('score-color'),
    drawerClusterTag: document.getElementById('drawer-cluster-tag'),
    drawerMembership: document.getElementById('drawer-membership'),
    drawerCoords2d: document.getElementById('drawer-coords-2d'),
    drawerCoords3d: document.getElementById('drawer-coords-3d'),
    drawerPalette: document.getElementById('drawer-palette'),
    drawerOcrText: document.getElementById('drawer-ocr-text'),
    drawerCopyOcrBtn: document.getElementById('drawer-copy-ocr-btn'),
    drawerSimilarBtn: document.getElementById('drawer-similar-btn'),
    drawerCopyPathBtn: document.getElementById('drawer-copy-path-btn'),

    // Sketch Modal
    btnOpenSketch: document.getElementById('btn-open-sketch'),
    modalSketch: document.getElementById('modal-sketch'),
    closeSketchBtn: document.getElementById('close-sketch-btn'),
    btnSearchSketch: document.getElementById('btn-search-sketch'),
    btnClearSketch: document.getElementById('btn-clear-sketch'),
    toolPen: document.getElementById('tool-pen'),
    toolEraser: document.getElementById('tool-eraser'),

    // Add Images Modal
    btnOpenAdd: document.getElementById('btn-open-add'),
    modalAdd: document.getElementById('modal-add'),
    closeAddBtn: document.getElementById('close-add-btn'),
    tabUploadFiles: document.getElementById('tab-upload-files'),
    tabScanDir: document.getElementById('tab-scan-dir'),
    bodyUploadFiles: document.getElementById('body-upload-files'),
    bodyScanDir: document.getElementById('body-scan-dir'),
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('file-input'),
    dirInput: document.getElementById('dir-input'),
    btnScanDir: document.getElementById('btn-scan-dir'),
    progressContainer: document.getElementById('progress-container'),
    progressFill: document.getElementById('progress-fill'),
    progressPercent: document.getElementById('progress-percent'),
    progressStep: document.getElementById('progress-step'),
    progressLogs: document.getElementById('progress-logs'),

    // 3D HUD
    hudReset: document.getElementById('hud-reset'),
    hudSpin: document.getElementById('hud-spin'),
    hudLines: document.getElementById('hud-lines'),

    // 2D Controls
    mapZoomIn: document.getElementById('map-zoom-in'),
    mapZoomOut: document.getElementById('map-zoom-out'),
    mapFit: document.getElementById('map-fit'),

    // Recluster
    btnRecluster: document.getElementById('btn-recluster'),
    toast: document.getElementById('toast'),
  };

  // -------------------------------------------------------------
  // INITIALIZE GALAXY ENGINES
  // -------------------------------------------------------------
  const galaxy3d = new Galaxy3D('galaxy3d-canvas', {
    onSelect: item => selectItem(item),
    onHover: (item, x, y) => showTooltip(item, x, y),
  });

  const galaxy2d = new Galaxy2D('galaxy2d-canvas', {
    onSelect: item => selectItem(item),
    onHover: (item, x, y) => showTooltip(item, x, y),
  });

  const sketchStudio = new SketchStudio('sketch-canvas');

  // -------------------------------------------------------------
  // TOAST NOTIFICATIONS
  // -------------------------------------------------------------
  function showToast(message, duration = 3000) {
    if (!el.toast) return;
    el.toast.textContent = message;
    el.toast.style.display = 'block';
    setTimeout(() => {
      el.toast.style.display = 'none';
    }, duration);
  }

  // -------------------------------------------------------------
  // FULL GALAXY VIEW TOGGLE
  // -------------------------------------------------------------
  if (el.btnToggleSearchBar && el.searchHub) {
    let isSearchVisible = true;
    el.btnToggleSearchBar.addEventListener('click', () => {
      isSearchVisible = !isSearchVisible;
      el.searchHub.style.display = isSearchVisible ? 'block' : 'none';
      el.btnToggleSearchBar.innerHTML = isSearchVisible ? '<span>🌌</span> Full Galaxy Mode' : '<span>🔍</span> Show Search Bar';
      galaxy3d.resize();
      galaxy2d.resize();
    });
  }

  // -------------------------------------------------------------
  // DATA FETCHING & SYNCHRONIZATION
  // -------------------------------------------------------------
  async function loadInitialData() {
    try {
      const [statusRes, imagesRes] = await Promise.all([
        fetch('/api/status').then(r => r.json()),
        fetch('/api/images').then(r => r.json()),
      ]);

      updateStats(statusRes);
      state.allImages = imagesRes;
      state.displayedImages = [...imagesRes];

      if (el.loadingBanner) el.loadingBanner.style.display = 'none';
      buildClusterPills();
      updateAllViews();
    } catch (err) {
      console.error('Failed to load initial data:', err);
      showToast('Could not connect to ZERO-TEXT backend.');
    }
  }

  function updateStats(status) {
    if (el.statTotal) el.statTotal.textContent = status.total_images;
    if (el.statClusters) el.statClusters.textContent = status.clusters_count;
    if (el.statText) el.statText.textContent = status.text_bearing_images;
  }

  function updateAllViews() {
    if (state.selectedItem) {
      galaxy3d.selectedItem = state.selectedItem;
      galaxy2d.selectedItem = state.selectedItem;
    }
    galaxy3d.setItems(state.displayedImages);
    galaxy2d.setItems(state.displayedImages);
    renderGrid(state.displayedImages);
  }

  // -------------------------------------------------------------
  // CLUSTER FILTER PILLS
  // -------------------------------------------------------------
  function buildClusterPills() {
    const container = el.clusterPillsContainer;
    if (!container) return;
    container.innerHTML = '';

    const allPill = document.createElement('span');
    allPill.className = `cluster-pill ${state.activeClusterFilter === null ? 'active' : ''}`;
    allPill.innerHTML = `All (${state.allImages.length})`;
    allPill.addEventListener('click', () => filterByCluster(null));
    container.appendChild(allPill);

    const clustersMap = {};
    state.allImages.forEach(it => {
      const c = it.cluster ?? -1;
      if (!clustersMap[c]) clustersMap[c] = [];
      clustersMap[c].push(it);
    });

    const sortedClusterIds = Object.keys(clustersMap).map(Number).sort((a, b) => a - b);
    sortedClusterIds.forEach(cId => {
      if (cId === -1) return;
      const count = clustersMap[cId].length;
      const pill = document.createElement('span');
      pill.className = `cluster-pill ${state.activeClusterFilter === cId ? 'active' : ''}`;
      const color = clustersMap[cId][0].cluster_color || '#4cc9f0';
      pill.innerHTML = `<span class="cluster-dot" style="background:${color}"></span> Cluster ${cId} (${count})`;
      pill.addEventListener('click', () => filterByCluster(cId));
      container.appendChild(pill);
    });

    if (clustersMap[-1]) {
      const noisePill = document.createElement('span');
      noisePill.className = `cluster-pill ${state.activeClusterFilter === -1 ? 'active' : ''}`;
      noisePill.innerHTML = `<span class="cluster-dot" style="background:#8b96a8"></span> Noise (${clustersMap[-1].length})`;
      noisePill.addEventListener('click', () => filterByCluster(-1));
      container.appendChild(noisePill);
    }
  }

  function filterByCluster(cId) {
    state.activeClusterFilter = cId;
    buildClusterPills();

    if (cId === null) {
      state.displayedImages = [...state.allImages];
    } else {
      state.displayedImages = state.allImages.filter(it => it.cluster === cId);
    }
    updateAllViews();
  }

  // -------------------------------------------------------------
  // VIEW SWITCHER
  // -------------------------------------------------------------
  function setView(viewName) {
    state.activeView = viewName;

    el.tabGalaxy3d.classList.toggle('active', viewName === 'galaxy3d');
    el.tabGalaxy2d.classList.toggle('active', viewName === 'galaxy2d');
    el.tabGrid.classList.toggle('active', viewName === 'grid');

    el.viewGalaxy3d.classList.toggle('active', viewName === 'galaxy3d');
    el.viewGalaxy2d.classList.toggle('active', viewName === 'galaxy2d');
    el.viewGrid.classList.toggle('active', viewName === 'grid');

    if (viewName === 'galaxy3d') galaxy3d.resize();
    if (viewName === 'galaxy2d') galaxy2d.resize();
  }

  el.tabGalaxy3d.addEventListener('click', () => setView('galaxy3d'));
  el.tabGalaxy2d.addEventListener('click', () => setView('galaxy2d'));
  el.tabGrid.addEventListener('click', () => setView('grid'));

  // -------------------------------------------------------------
  // GALLERY GRID RENDERING
  // -------------------------------------------------------------
  function renderGrid(items) {
    const wrapper = el.gridWrapper;
    wrapper.innerHTML = '';

    if (!items.length) {
      wrapper.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px;color:var(--text-muted)">No images match your query.</div>';
      return;
    }

    items.forEach(item => {
      const card = document.createElement('div');
      card.className = `image-card ${state.selectedItem && state.selectedItem.filename === item.filename ? 'active' : ''}`;

      const clusterLabel = item.cluster === -1 ? 'Noise' : `C${item.cluster}`;
      const clusterColor = item.cluster_color || '#4cc9f0';
      const scorePct = item.final !== undefined && item.final > 0 ? Math.round(item.final * 100) : null;

      let paletteHtml = '';
      if (item.dominant_colours && item.dominant_colours.length) {
        paletteHtml = '<div class="card-palette">';
        item.dominant_colours.forEach(c => {
          paletteHtml += `<div class="palette-slice" style="width:${c.percentage}%;background:${c.hex}"></div>`;
        });
        paletteHtml += '</div>';
      }

      card.innerHTML = `
        <div class="card-img-wrap">
          <img class="card-img" src="/api/image/${encodeURIComponent(item.filename)}" loading="lazy" alt="${item.filename}" />
          <span class="card-cluster-badge" style="background:${clusterColor}">${clusterLabel}</span>
          ${scorePct !== null ? `<span class="card-score-badge">${scorePct}%</span>` : ''}
          ${item.has_text || (item.full_text && item.full_text.trim()) ? '<span class="card-ocr-tag">OCR</span>' : ''}
        </div>
        <div class="card-body">
          <div class="card-title" title="${item.filename}">${item.filename}</div>
          <div class="card-ocr-preview">${item.full_text || 'No text detected in image'}</div>
          ${paletteHtml}
        </div>
      `;

      card.addEventListener('click', () => selectItem(item));
      wrapper.appendChild(card);
    });
  }

  // -------------------------------------------------------------
  // TOOLTIP RENDERING
  // -------------------------------------------------------------
  function showTooltip(item, mouseX, mouseY) {
    if (!item) {
      el.tooltip.style.display = 'none';
      return;
    }

    el.tooltipThumb.src = `/api/image/${encodeURIComponent(item.filename)}`;
    el.tooltipTitle.textContent = item.filename;

    const clusterLabel = item.cluster === -1 ? 'Noise / Outlier' : `Cluster ${item.cluster}`;
    el.tooltipCluster.textContent = clusterLabel;
    el.tooltipCluster.style.color = item.cluster_color || '#4cc9f0';

    if (item.final !== undefined && item.final > 0) {
      el.tooltipScore.textContent = `${Math.round(item.final * 100)}% match`;
      el.tooltipScore.style.display = 'inline-block';
    } else {
      el.tooltipScore.style.display = 'none';
    }

    const TW = 190, TH = 145;
    let tx = mouseX + 14;
    let ty = mouseY - TH / 2;
    if (tx + TW > window.innerWidth) tx = mouseX - TW - 14;
    if (ty < 0) ty = 8;
    if (ty + TH > window.innerHeight) ty = window.innerHeight - TH - 8;

    el.tooltip.style.left = `${tx}px`;
    el.tooltip.style.top = `${ty}px`;
    el.tooltip.style.display = 'block';
  }

  // -------------------------------------------------------------
  // IMAGE INSPECTION DRAWER
  // -------------------------------------------------------------
  function selectItem(item) {
    if (!item) return;
    state.selectedItem = item;

    galaxy3d.selectedItem = item;
    galaxy2d.selectedItem = item;
    if (galaxy2d.render) galaxy2d.render();
    if (galaxy3d.useThree) galaxy3d.buildThreeScene();

    document.querySelectorAll('.image-card').forEach(c => {
      const title = c.querySelector('.card-title');
      if (title && title.textContent === item.filename) {
        c.classList.add('active');
        c.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        c.classList.remove('active');
      }
    });

    // Populate Drawer
    el.drawerTitle.textContent = item.filename;
    el.drawerPreview.src = `/api/image/${encodeURIComponent(item.filename)}`;

    const finalPct = Math.round((item.final ?? 0) * 100);
    const clipPct = Math.round((item.clip ?? 0) * 100);
    const ocrPct = Math.round((item.ocr ?? 0) * 100);
    const colPct = Math.round((item.colour ?? 0) * 100);

    el.drawerScoreFinal.textContent = `${finalPct}%`;
    el.drawerScoreClip.textContent = `${clipPct}%`;
    el.drawerScoreOcr.textContent = `${ocrPct}%`;
    el.drawerScoreColor.textContent = `${colPct}%`;

    const cId = item.cluster ?? -1;
    el.drawerClusterTag.textContent = cId === -1 ? 'Noise' : `Cluster ${cId}`;
    el.drawerClusterTag.style.background = item.cluster_color || '#4cc9f0';
    el.drawerMembership.textContent = `${Math.round((item.membership_probability ?? 0) * 100)}%`;

    const x2 = (item.x ?? 0).toFixed(2);
    const y2 = (item.y ?? 0).toFixed(2);
    el.drawerCoords2d.textContent = `(${x2}, ${y2})`;

    const x3 = (item.x3d ?? item.x ?? 0).toFixed(2);
    const y3 = (item.y3d ?? item.y ?? 0).toFixed(2);
    const z3 = (item.z3d ?? item.z ?? 0).toFixed(2);
    el.drawerCoords3d.textContent = `(${x3}, ${y3}, ${z3})`;

    el.drawerPalette.innerHTML = '';
    const dominant = item.dominant_colours || [];
    if (dominant.length) {
      dominant.forEach(c => {
        const chip = document.createElement('div');
        chip.className = 'color-chip';
        chip.innerHTML = `
          <span class="color-chip-swatch" style="background:${c.hex}"></span>
          <span>${c.hex} (${c.percentage}%)</span>
        `;
        chip.title = `Search by color ${c.hex}`;
        chip.addEventListener('click', () => searchByColor(c.hex));
        el.drawerPalette.appendChild(chip);
      });
    } else {
      el.drawerPalette.innerHTML = '<span style="color:var(--text-dim);font-size:0.75rem">No palette extracted</span>';
    }

    const ocrText = (item.full_text || '').trim();
    el.drawerOcrText.textContent = ocrText || '[No readable text detected in this image]';

    el.drawerCopyOcrBtn.onclick = () => {
      navigator.clipboard.writeText(ocrText);
      showToast('OCR text copied to clipboard!');
    };

    el.drawerCopyPathBtn.onclick = () => {
      navigator.clipboard.writeText(item.path);
      showToast('Image path copied!');
    };

    el.drawerSimilarBtn.onclick = () => {
      searchSimilar(item.filename);
    };

    el.drawer.classList.add('open');
  }

  el.drawerCloseBtn.addEventListener('click', () => {
    el.drawer.classList.remove('open');
  });

  // -------------------------------------------------------------
  // AI PRESET BUTTONS
  // -------------------------------------------------------------
  function setPreset(preset) {
    el.presetPills.forEach(p => p.classList.toggle('active', p.dataset.preset === preset));

    if (preset === 'auto') {
      state.autoWeights = true;
      el.deciderLabel.textContent = 'Auto Balancing';
      el.deciderReason.textContent = '(Dynamic query-adaptive decider)';
    } else {
      state.autoWeights = false;
      if (preset === 'semantic') {
        applyWeightsUI(0.75, 0.15, 0.10, '🧠 Semantic Focus', 'CLIP visual semantic weighting');
      } else if (preset === 'ocr') {
        applyWeightsUI(0.25, 0.65, 0.10, '📝 Text & OCR Focus', 'RapidOCR in-image text weighting');
      } else if (preset === 'colour') {
        applyWeightsUI(0.20, 0.05, 0.75, '🎨 Colour Harmony', 'Dominant colour palette weighting');
      } else if (preset === 'balanced') {
        applyWeightsUI(0.50, 0.30, 0.20, '⚖️ Balanced Hybrid', 'Even split across all features');
      }
    }

    if (el.searchInput.value.trim()) {
      performHybridSearch();
    }
  }

  function applyWeightsUI(clip, ocr, col, label, reason) {
    state.weights.clip = clip;
    state.weights.ocr = ocr;
    state.weights.colour = col;

    el.weightClipSlider.value = Math.round(clip * 100);
    el.weightOcrSlider.value = Math.round(ocr * 100);
    el.weightColourSlider.value = Math.round(col * 100);

    el.weightClipVal.textContent = `${Math.round(clip * 100)}%`;
    el.weightOcrVal.textContent = `${Math.round(ocr * 100)}%`;
    el.weightColourVal.textContent = `${Math.round(col * 100)}%`;

    el.deciderLabel.textContent = label;
    el.deciderReason.textContent = `(${reason})`;
  }

  el.presetPills.forEach(btn => {
    btn.addEventListener('click', () => setPreset(btn.dataset.preset));
  });

  // -------------------------------------------------------------
  // HYBRID SEARCH
  // -------------------------------------------------------------
  async function performHybridSearch() {
    const query = el.searchInput.value.trim();
    if (!query) {
      clearSearch();
      return;
    }

    el.searchClearBtn.style.display = 'flex';
    if (el.searchBtn) {
      el.searchBtn.textContent = '…';
      el.searchBtn.disabled = true;
    }

    try {
      const res = await fetch('/api/search/hybrid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          clip_weight: state.autoWeights ? null : state.weights.clip,
          ocr_weight: state.autoWeights ? null : state.weights.ocr,
          colour_weight: state.autoWeights ? null : state.weights.colour,
          auto_weights: state.autoWeights,
          cluster_filter: state.activeClusterFilter,
          top_k: 40,
        }),
      });

      const data = await res.json();
      state.displayedImages = data.results;
      state.currentMatchIndex = 0;

      // Update AI Recommendation Display
      if (data.recommendation) {
        const rec = data.recommendation;
        el.deciderLabel.textContent = rec.label || 'Active';
        el.deciderReason.textContent = `(${rec.reason || 'Auto-balanced'})`;

        if (state.autoWeights) {
          applyWeightsUI(rec.clip, rec.ocr, rec.colour, rec.label, rec.reason);
        }
      }

      updateAllViews();
      renderQueryInsights(data.detected_words, data.detected_colours);

      if (data.results.length > 0) {
        const topItem = data.results[0];
        selectItem(topItem);

        // Update 3D spotlight HUD & fly camera
        updateSpotlightHUD(topItem, data.results.length);
        if (state.activeView === 'galaxy3d') {
          galaxy3d.flyToItem(topItem);
        } else if (state.activeView === 'galaxy2d') {
          galaxy2d.focusItem(topItem);
        }
      } else {
        hideSpotlightHUD();
      }

      showToast(`Found ${data.total_results} matching images.`);
    } catch (err) {
      console.error('Search failed:', err);
      showToast('Search error. See console for details.');
    } finally {
      if (el.searchBtn) {
        el.searchBtn.textContent = '🔍';
        el.searchBtn.disabled = false;
      }
    }
  }

  function updateSpotlightHUD(item, total) {
    if (el.spotlight3dHud) {
      el.spotlight3dTitle.textContent = item.filename;
      el.spotlight3dScore.textContent = `${Math.round((item.final || 0) * 100)}% match`;
      el.spotlight3dHud.style.display = 'flex';
    }
    if (el.spotlight2dHud) {
      el.spotlight2dTitle.textContent = item.filename;
      el.spotlight2dScore.textContent = `${Math.round((item.final || 0) * 100)}% match`;
      el.spotlight2dHud.style.display = 'flex';
    }
  }

  function hideSpotlightHUD() {
    if (el.spotlight3dHud) el.spotlight3dHud.style.display = 'none';
    if (el.spotlight2dHud) el.spotlight2dHud.style.display = 'none';
  }

  // 3D / 2D Spotlight Navigator Click Handlers
  if (el.spotlight3dNext) {
    el.spotlight3dNext.addEventListener('click', () => {
      if (!state.displayedImages.length) return;
      state.currentMatchIndex = (state.currentMatchIndex + 1) % state.displayedImages.length;
      const target = state.displayedImages[state.currentMatchIndex];
      selectItem(target);
      updateSpotlightHUD(target, state.displayedImages.length);
      galaxy3d.flyToItem(target);
    });
  }

  if (el.spotlight3dPrev) {
    el.spotlight3dPrev.addEventListener('click', () => {
      if (!state.displayedImages.length) return;
      state.currentMatchIndex = (state.currentMatchIndex - 1 + state.displayedImages.length) % state.displayedImages.length;
      const target = state.displayedImages[state.currentMatchIndex];
      selectItem(target);
      updateSpotlightHUD(target, state.displayedImages.length);
      galaxy3d.flyToItem(target);
    });
  }

  if (el.spotlight2dNext) {
    el.spotlight2dNext.addEventListener('click', () => {
      if (!state.displayedImages.length) return;
      state.currentMatchIndex = (state.currentMatchIndex + 1) % state.displayedImages.length;
      const target = state.displayedImages[state.currentMatchIndex];
      selectItem(target);
      updateSpotlightHUD(target, state.displayedImages.length);
      galaxy2d.focusItem(target);
    });
  }

  function renderQueryInsights(words, colours) {
    const container = el.queryInsights;
    if (!container) return;
    container.innerHTML = '';
    if (!words.length && !colours.length) {
      container.style.display = 'none';
      return;
    }
    container.style.display = 'flex';
    container.innerHTML = '<span style="font-weight:700">Insights:</span>';

    words.forEach(w => {
      const tag = document.createElement('span');
      tag.className = 'insight-tag';
      tag.textContent = `Text: ${w}`;
      container.appendChild(tag);
    });

    colours.forEach(c => {
      const tag = document.createElement('span');
      tag.className = 'insight-tag';
      tag.style.borderColor = 'var(--accent-pink)';
      tag.style.color = 'var(--accent-pink)';
      tag.textContent = `Color: ${c}`;
      container.appendChild(tag);
    });
  }

  function clearSearch() {
    el.searchInput.value = '';
    el.searchClearBtn.style.display = 'none';
    if (el.queryInsights) el.queryInsights.style.display = 'none';
    hideSpotlightHUD();
    filterByCluster(state.activeClusterFilter);
  }

  el.searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') performHybridSearch();
  });
  if (el.searchBtn) el.searchBtn.addEventListener('click', performHybridSearch);
  el.searchClearBtn.addEventListener('click', clearSearch);

  // Manual Slider Adjustments
  if (el.weightsToggleBtn) {
    el.weightsToggleBtn.addEventListener('click', () => {
      el.weightsDrawer.classList.toggle('open');
    });
  }

  function onSliderChange() {
    state.autoWeights = false;
    el.presetPills.forEach(p => p.classList.remove('active'));

    state.weights.clip = parseFloat(el.weightClipSlider.value) / 100;
    state.weights.ocr = parseFloat(el.weightOcrSlider.value) / 100;
    state.weights.colour = parseFloat(el.weightColourSlider.value) / 100;

    el.weightClipVal.textContent = `${Math.round(state.weights.clip * 100)}%`;
    el.weightOcrVal.textContent = `${Math.round(state.weights.ocr * 100)}%`;
    el.weightColourVal.textContent = `${Math.round(state.weights.colour * 100)}%`;

    el.deciderLabel.textContent = 'Custom Manual';
    el.deciderReason.textContent = '(Adjusted via sliders)';

    if (el.searchInput.value.trim()) {
      clearTimeout(state.searchDebounceTimer);
      state.searchDebounceTimer = setTimeout(performHybridSearch, 300);
    }
  }

  el.weightClipSlider.addEventListener('input', onSliderChange);
  el.weightOcrSlider.addEventListener('input', onSliderChange);
  el.weightColourSlider.addEventListener('input', onSliderChange);

  // -------------------------------------------------------------
  // COLOR & SIMILARITY SEARCH
  // -------------------------------------------------------------
  async function searchByColor(hexCode) {
    try {
      const res = await fetch('/api/search/color', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hex: hexCode }),
      });
      const data = await res.json();
      state.displayedImages = data.results;
      updateAllViews();
      if (data.results.length > 0) {
        selectItem(data.results[0]);
        updateSpotlightHUD(data.results[0], data.results.length);
        if (state.activeView === 'galaxy3d') galaxy3d.flyToItem(data.results[0]);
        if (state.activeView === 'galaxy2d') galaxy2d.focusItem(data.results[0]);
      }
      showToast(`Showing images matching color ${hexCode}`);
    } catch (err) {
      console.error('Color search failed:', err);
    }
  }

  async function searchSimilar(filename) {
    try {
      const res = await fetch('/api/search/similar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      const data = await res.json();
      state.displayedImages = data.results;
      updateAllViews();
      showToast(`Showing nearest visual neighbors for ${filename}`);
    } catch (err) {
      console.error('Similar search failed:', err);
    }
  }

  // -------------------------------------------------------------
  // SKETCH STUDIO INTEGRATION
  // -------------------------------------------------------------
  el.btnOpenSketch.addEventListener('click', () => {
    el.modalSketch.classList.add('open');
    sketchStudio.initCanvas();
  });

  el.closeSketchBtn.addEventListener('click', () => {
    el.modalSketch.classList.remove('open');
  });

  el.btnClearSketch.addEventListener('click', () => {
    sketchStudio.clear();
  });

  el.toolPen.addEventListener('click', () => {
    sketchStudio.setTool('pen');
    el.toolPen.classList.add('active');
    el.toolEraser.classList.remove('active');
  });

  el.toolEraser.addEventListener('click', () => {
    sketchStudio.setTool('eraser');
    el.toolEraser.classList.add('active');
    el.toolPen.classList.remove('active');
  });

  document.querySelectorAll('.brush-size-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.brush-size-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      sketchStudio.setBrushSize(parseInt(btn.dataset.size));
    });
  });

  el.btnSearchSketch.addEventListener('click', async () => {
    const dataUrl = sketchStudio.toDataURL();
    el.btnSearchSketch.textContent = 'Analyzing sketch…';
    try {
      const res = await fetch('/api/search/sketch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sketch_data: dataUrl }),
      });
      const data = await res.json();

      el.modalSketch.classList.remove('open');
      state.displayedImages = data.results;
      updateAllViews();
      if (data.results.length > 0) {
        selectItem(data.results[0]);
        updateSpotlightHUD(data.results[0], data.results.length);
        if (state.activeView === 'galaxy3d') galaxy3d.flyToItem(data.results[0]);
        if (state.activeView === 'galaxy2d') galaxy2d.focusItem(data.results[0]);
      }
      showToast(`Sketch search complete: ${data.total_results} matches found.`);
    } catch (err) {
      console.error('Sketch search error:', err);
      showToast('Sketch search failed.');
    } finally {
      el.btnSearchSketch.textContent = 'Search Galaxy by Sketch';
    }
  });

  // -------------------------------------------------------------
  // INGESTION & INDEXING PIPELINE MODAL
  // -------------------------------------------------------------
  el.btnOpenAdd.addEventListener('click', () => {
    el.modalAdd.classList.add('open');
  });

  el.closeAddBtn.addEventListener('click', () => {
    el.modalAdd.classList.remove('open');
  });

  el.tabUploadFiles.addEventListener('click', () => {
    el.tabUploadFiles.classList.add('active');
    el.tabScanDir.classList.remove('active');
    el.bodyUploadFiles.style.display = 'block';
    el.bodyScanDir.style.display = 'none';
  });

  el.tabScanDir.addEventListener('click', () => {
    el.tabScanDir.classList.add('active');
    el.tabUploadFiles.classList.remove('active');
    el.bodyScanDir.style.display = 'block';
    el.bodyUploadFiles.style.display = 'none';
  });

  el.dropzone.addEventListener('click', () => el.fileInput.click());
  el.dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    el.dropzone.classList.add('dragover');
  });
  el.dropzone.addEventListener('dragleave', () => {
    el.dropzone.classList.remove('dragover');
  });
  el.dropzone.addEventListener('drop', e => {
    e.preventDefault();
    el.dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      uploadFiles(e.dataTransfer.files);
    }
  });
  el.fileInput.addEventListener('change', e => {
    if (e.target.files.length) {
      uploadFiles(e.target.files);
    }
  });

  async function uploadFiles(files) {
    startProgressTracker();
    try {
      const fileItems = [];
      for (const f of files) {
        const dataUrl = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = ev => resolve(ev.target.result);
          reader.onerror = reject;
          reader.readAsDataURL(f);
        });
        fileItems.push({ filename: f.name, data: dataUrl });
      }

      const res = await fetch('/api/index/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: fileItems }),
      });
      const data = await res.json();
      showToast(data.message);
    } catch (err) {
      console.error('Upload failed:', err);
      showToast('File upload failed.');
    }
  }

  el.btnScanDir.addEventListener('click', async () => {
    const dirPath = el.dirInput.value.trim();
    if (!dirPath) {
      showToast('Please enter a directory path.');
      return;
    }

    startProgressTracker();
    try {
      const res = await fetch('/api/index/directory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directory_path: dirPath }),
      });
      const data = await res.json();
      showToast(data.message);
    } catch (err) {
      console.error('Directory index failed:', err);
      showToast('Could not index directory: ' + err.message);
    }
  });

  function startProgressTracker() {
    el.progressContainer.style.display = 'flex';
    const pollTimer = setInterval(async () => {
      try {
        const res = await fetch('/api/index/status');
        const s = await res.json();

        el.progressFill.style.width = `${s.progress}%`;
        el.progressPercent.textContent = `${s.progress}%`;
        el.progressStep.textContent = s.message;

        if (s.logs && s.logs.length) {
          el.progressLogs.innerHTML = s.logs.map(l => `<div>[${l.time}] ${l.message}</div>`).join('');
          el.progressLogs.scrollTop = el.progressLogs.scrollHeight;
        }

        if (s.status === 'idle' && s.progress === 100) {
          clearInterval(pollTimer);
          showToast('Indexing completed! Reloading galaxy…');
          setTimeout(() => {
            loadInitialData();
            el.modalAdd.classList.remove('open');
            el.progressContainer.style.display = 'none';
          }, 1500);
        }
      } catch (e) {
        console.error('Error polling status:', e);
      }
    }, 1000);
  }

  // -------------------------------------------------------------
  // RECLUSTER BUTTON
  // -------------------------------------------------------------
  el.btnRecluster.addEventListener('click', async () => {
    el.btnRecluster.textContent = 'Re-clustering…';
    try {
      const res = await fetch('/api/recluster', { method: 'POST' });
      await res.json();
      showToast('Reclustering complete!');
      await loadInitialData();
    } catch (err) {
      console.error('Reclustering failed:', err);
      showToast('Reclustering failed.');
    } finally {
      el.btnRecluster.textContent = '🔄 Recluster';
    }
  });

  // -------------------------------------------------------------
  // 3D HUD & 2D CONTROLS
  // -------------------------------------------------------------
  el.hudReset.addEventListener('click', () => galaxy3d.resetCamera());
  el.hudSpin.addEventListener('click', () => {
    const isSpinning = galaxy3d.toggleAutoRotate();
    el.hudSpin.classList.toggle('active', isSpinning);
  });
  el.hudLines.addEventListener('click', () => {
    const show = galaxy3d.toggleConstellations();
    el.hudLines.classList.toggle('active', show);
  });

  el.mapZoomIn.addEventListener('click', () => galaxy2d.zoomIn());
  el.mapZoomOut.addEventListener('click', () => galaxy2d.zoomOut());
  el.mapFit.addEventListener('click', () => galaxy2d.fitView());

  window.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      el.drawer.classList.remove('open');
      el.modalSketch.classList.remove('open');
      el.modalAdd.classList.remove('open');
    }
  });

  loadInitialData();
});
