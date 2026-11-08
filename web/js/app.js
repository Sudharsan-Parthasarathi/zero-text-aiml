/**
 * ZERO-TEXT: MAIN FRONTEND APPLICATION CONTROLLER
 * Coordinates 3D Galaxy, 2D Map, Gallery Grid, Multi-modal Search,
 * Ingestion Pipeline, and Image Inspector.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    allImages: [],
    displayedImages: [],
    selectedItem: null,
    activeView: 'galaxy3d', // 'galaxy3d' | 'galaxy2d' | 'grid'
    activeClusterFilter: null,
    weights: {
      clip: 0.55,
      ocr: 0.30,
      colour: 0.15,
    },
    isIndexing: false,
    pollTimer: null,
  };

  // UI Elements
  const el = {
    // Stats
    statTotal: document.getElementById('stat-total'),
    statClusters: document.getElementById('stat-clusters'),
    statText: document.getElementById('stat-text'),

    // View tabs
    tabGalaxy3d: document.getElementById('tab-galaxy3d'),
    tabGalaxy2d: document.getElementById('tab-galaxy2d'),
    tabGrid: document.getElementById('tab-grid'),
    viewGalaxy3d: document.getElementById('view-galaxy3d'),
    viewGalaxy2d: document.getElementById('view-galaxy2d'),
    viewGrid: document.getElementById('view-grid'),

    // Search
    searchInput: document.getElementById('search-input'),
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
    el.toast.textContent = message;
    el.toast.style.display = 'block';
    setTimeout(() => {
      el.toast.style.display = 'none';
    }, duration);
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

      buildClusterPills();
      updateAllViews();
    } catch (err) {
      console.error('Failed to load initial data:', err);
      showToast('Could not load data from ZERO-TEXT server.');
    }
  }

  function updateStats(status) {
    if (el.statTotal) el.statTotal.textContent = status.total_images;
    if (el.statClusters) el.statClusters.textContent = status.clusters_count;
    if (el.statText) el.statText.textContent = status.text_bearing_images;
  }

  function updateAllViews() {
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

    // "All" Pill
    const allPill = document.createElement('span');
    allPill.className = `cluster-pill ${state.activeClusterFilter === null ? 'active' : ''}`;
    allPill.innerHTML = `All (${state.allImages.length})`;
    allPill.addEventListener('click', () => filterByCluster(null));
    container.appendChild(allPill);

    // Discover unique clusters
    const clustersMap = {};
    state.allImages.forEach(it => {
      const c = it.cluster ?? -1;
      if (!clustersMap[c]) clustersMap[c] = [];
      clustersMap[c].push(it);
    });

    const sortedClusterIds = Object.keys(clustersMap).map(Number).sort((a, b) => a - b);

    sortedClusterIds.forEach(cId => {
      if (cId === -1) return; // noise handled last
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
      const scorePct = item.final !== undefined ? Math.round(item.final * 100) : null;

      // Palette slices
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

    if (item.final !== undefined) {
      el.tooltipScore.textContent = `${Math.round(item.final * 100)}% match`;
      el.tooltipScore.style.display = 'inline-block';
    } else {
      el.tooltipScore.style.display = 'none';
    }

    el.tooltip.style.left = `${mouseX}px`;
    el.tooltip.style.top = `${mouseY}px`;
    el.tooltip.style.display = 'block';
  }

  // -------------------------------------------------------------
  // IMAGE INSPECTION DRAWER
  // -------------------------------------------------------------
  function selectItem(item) {
    state.selectedItem = item;
    galaxy3d.selectItem(item);
    galaxy2d.selectItem(item);

    // Update grid selection highlights
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

    // Scores
    const finalPct = Math.round((item.final ?? 0) * 100);
    const clipPct = Math.round((item.clip ?? 0) * 100);
    const ocrPct = Math.round((item.ocr ?? 0) * 100);
    const colPct = Math.round((item.colour ?? 0) * 100);

    el.drawerScoreFinal.textContent = `${finalPct}%`;
    el.drawerScoreClip.textContent = `${clipPct}%`;
    el.drawerScoreOcr.textContent = `${ocrPct}%`;
    el.drawerScoreColor.textContent = `${colPct}%`;

    // Cluster info
    const cId = item.cluster ?? -1;
    el.drawerClusterTag.textContent = cId === -1 ? 'Noise' : `Cluster ${cId}`;
    el.drawerClusterTag.style.background = item.cluster_color || '#4cc9f0';
    el.drawerMembership.textContent = `${Math.round((item.membership_probability ?? 0) * 100)}%`;

    // Coordinates
    const x2 = (item.x ?? 0).toFixed(2);
    const y2 = (item.y ?? 0).toFixed(2);
    el.drawerCoords2d.textContent = `(${x2}, ${y2})`;

    const x3 = (item.x3d ?? item.x ?? 0).toFixed(2);
    const y3 = (item.y3d ?? item.y ?? 0).toFixed(2);
    const z3 = (item.z3d ?? item.z ?? 0).toFixed(2);
    el.drawerCoords3d.textContent = `(${x3}, ${y3}, ${z3})`;

    // Dominant Colours
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

    // OCR Text
    const ocrText = (item.full_text || '').trim();
    el.drawerOcrText.textContent = ocrText || '[No readable text detected in this image]';

    // Actions
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

    // Open Drawer
    el.drawer.classList.add('open');
  }

  el.drawerCloseBtn.addEventListener('click', () => {
    el.drawer.classList.remove('open');
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

    try {
      const res = await fetch('/api/search/hybrid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          clip_weight: state.weights.clip,
          ocr_weight: state.weights.ocr,
          colour_weight: state.weights.colour,
          cluster_filter: state.activeClusterFilter,
        }),
      });
      const data = await res.json();

      state.displayedImages = data.results;
      updateAllViews();

      // Update Query Insights
      renderQueryInsights(data.detected_words, data.detected_colours);

      if (data.results.length > 0) {
        selectItem(data.results[0]);
      }
      showToast(`Found ${data.total_results} matching images.`);
    } catch (err) {
      console.error('Search failed:', err);
      showToast('Search error. See console for details.');
    }
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
    filterByCluster(state.activeClusterFilter);
  }

  el.searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') performHybridSearch();
  });
  el.searchClearBtn.addEventListener('click', clearSearch);

  // Weights Slider Handling
  el.weightsToggleBtn.addEventListener('click', () => {
    el.weightsDrawer.classList.toggle('open');
  });

  function updateWeights() {
    state.weights.clip = parseFloat(el.weightClipSlider.value) / 100;
    state.weights.ocr = parseFloat(el.weightOcrSlider.value) / 100;
    state.weights.colour = parseFloat(el.weightColourSlider.value) / 100;

    el.weightClipVal.textContent = `${Math.round(state.weights.clip * 100)}%`;
    el.weightOcrVal.textContent = `${Math.round(state.weights.ocr * 100)}%`;
    el.weightColourVal.textContent = `${Math.round(state.weights.colour * 100)}%`;

    if (el.searchInput.value.trim()) {
      performHybridSearch();
    }
  }

  el.weightClipSlider.addEventListener('input', updateWeights);
  el.weightOcrSlider.addEventListener('input', updateWeights);
  el.weightColourSlider.addEventListener('input', updateWeights);

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
      if (data.results.length > 0) selectItem(data.results[0]);
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
    el.btnSearchSketch.textContent = 'Analyzing sketch...';
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
      if (data.results.length > 0) selectItem(data.results[0]);
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

  // Drag & drop
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
          reader.onload = e => resolve(e.target.result);
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
    if (state.pollTimer) clearInterval(state.pollTimer);

    state.pollTimer = setInterval(async () => {
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
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          showToast('Indexing completed! Reloading galaxy...');
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
    el.btnRecluster.textContent = 'Re-clustering...';
    try {
      const res = await fetch('/api/recluster', { method: 'POST' });
      const data = await res.json();
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

  // Close modals on escape
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      el.drawer.classList.remove('open');
      el.modalSketch.classList.remove('open');
      el.modalAdd.classList.remove('open');
    }
  });

  // Initial load
  loadInitialData();
});
