/**
 * ZERO-TEXT: 2D VISUAL STAR MAP EXPLORER (v2.5 Radiant Spotlight & Centering)
 * Interactive high-DPI HTML5 canvas displaying UMAP 2D coordinates,
 * cluster groups, density centroids, and pan-zoom navigation.
 */

class Galaxy2D {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.container = this.canvas.parentElement;
    this.options = Object.assign({
      onSelect: null,
      onHover: null,
    }, options);

    this.items = [];
    this.selectedItem = null;

    // View transform
    this.offsetX = 0;
    this.offsetY = 0;
    this.scale = 1.0;
    this.isDragging = false;
    this.mouseDownX = 0;
    this.mouseDownY = 0;
    this.lastX = 0;
    this.lastY = 0;

    this.points = [];
    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.bindEvents();
    this.render();
  }

  resize() {
    this.width = this.container.clientWidth || window.innerWidth;
    this.height = this.container.clientHeight || window.innerHeight;
    this.canvas.width = this.width * window.devicePixelRatio;
    this.canvas.height = this.height * window.devicePixelRatio;
    this.canvas.style.width = this.width + 'px';
    this.canvas.style.height = this.height + 'px';
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this.render();
  }

  setItems(items) {
    this.items = items || [];
    if (this.selectedItem) {
      const still = this.items.find(it => it.filename === this.selectedItem.filename);
      if (!still) this.selectedItem = null;
    }
    this.fitView();
    this.render();
  }

  fitView() {
    if (!this.items.length) {
      this.offsetX = this.width / 2;
      this.offsetY = this.height / 2;
      this.scale = 1.0;
      return;
    }

    const xs = this.items.map(it => it.x ?? 0);
    const ys = this.items.map(it => it.y ?? 0);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;

    const margin = 90;
    const scaleX = (this.width - margin * 2) / spanX;
    const scaleY = (this.height - margin * 2) / spanY;
    this.scale = Math.min(scaleX, scaleY, 70);

    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    this.offsetX = this.width / 2 - midX * this.scale;
    this.offsetY = this.height / 2 + midY * this.scale;
  }

  focusItem(item) {
    if (!item) return;
    const target = this.points.find(p => p.filename === item.filename);
    if (!target) return;

    // Smooth center on target
    const targetOffsetX = this.width / 2 - (item.x ?? 0) * this.scale;
    const targetOffsetY = this.height / 2 + (item.y ?? 0) * this.scale;
    this.offsetX = targetOffsetX;
    this.offsetY = targetOffsetY;
    this.render();
  }

  zoomIn() {
    this.scale *= 1.25;
    this.render();
  }

  zoomOut() {
    this.scale = Math.max(5, this.scale / 1.25);
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);

    // Subtle background grid
    ctx.strokeStyle = 'rgba(112, 225, 209, 0.04)';
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = (this.offsetX % gridSize); x < w; x += gridSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = (this.offsetY % gridSize); y < h; y += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    if (!this.items.length) {
      ctx.fillStyle = '#5a6b85';
      ctx.font = '14px Segoe UI, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No images in view', w / 2, h / 2);
      return;
    }

    this.points = [];
    const clusters = {};
    const hasMatches = this.items.some(it => (it.final || 0) > 0.2);

    this.items.forEach(item => {
      const px = this.offsetX + (item.x ?? 0) * this.scale;
      const py = this.offsetY - (item.y ?? 0) * this.scale;
      this.points.push({ ...item, px, py });

      const cId = item.cluster ?? -1;
      if (cId !== -1) {
        if (!clusters[cId]) clusters[cId] = [];
        clusters[cId].push({ px, py, color: item.cluster_color || '#4cc9f0' });
      }
    });

    // Draw cluster constellation lines & halos
    Object.values(clusters).forEach(group => {
      if (group.length < 2) return;
      const color = group[0].color;

      ctx.strokeStyle = color + '22';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < group.length; i++) {
        for (let j = i + 1; j < group.length; j++) {
          ctx.moveTo(group[i].px, group[i].py);
          ctx.lineTo(group[j].px, group[j].py);
        }
      }
      ctx.stroke();

      const avgX = group.reduce((acc, p) => acc + p.px, 0) / group.length;
      const avgY = group.reduce((acc, p) => acc + p.py, 0) / group.length;
      const radius = 70;
      const grad = ctx.createRadialGradient(avgX, avgY, 5, avgX, avgY, radius);
      grad.addColorStop(0, color + '1a');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(avgX, avgY, radius, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw Image Nodes
    this.points.forEach(p => {
      const isSelected = this.selectedItem && this.selectedItem.filename === p.filename;
      const score = p.final || 0;
      const isTopMatch = hasMatches && score > 0.4;
      const isDimmed = hasMatches && score <= 0.15 && !isSelected;

      const color = p.cluster_color || '#4cc9f0';
      const radius = isSelected ? 15 : (isTopMatch ? 12 : 7);

      if (isDimmed) {
        ctx.globalAlpha = 0.22;
      }

      // Outer glow for matches or selected
      ctx.beginPath();
      ctx.arc(p.px, p.py, radius * (isSelected ? 2.4 : (isTopMatch ? 2.0 : 1.6)), 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? 'rgba(112, 225, 209, 0.45)' : (isTopMatch ? 'rgba(255, 209, 102, 0.35)' : color + '22');
      ctx.fill();

      // Node core
      ctx.beginPath();
      ctx.arc(p.px, p.py, radius, 0, Math.PI * 2);
      ctx.fillStyle = isTopMatch ? '#ffd166' : color;
      ctx.fill();
      ctx.strokeStyle = isSelected ? '#70e1d1' : (isTopMatch ? '#ffffff' : 'rgba(255, 255, 255, 0.7)');
      ctx.lineWidth = isSelected ? 3 : (isTopMatch ? 2 : 1);
      ctx.stroke();

      // Label & Match Badge
      ctx.fillStyle = isSelected ? '#70e1d1' : (isTopMatch ? '#ffd166' : '#e2eaf5');
      ctx.font = isSelected || isTopMatch ? 'bold 11px Segoe UI, sans-serif' : '10px Segoe UI, sans-serif';
      ctx.textAlign = 'left';
      const labelText = isTopMatch ? `${p.filename.split('.')[0]} (${Math.round(score * 100)}%)` : p.filename.split('.')[0];
      ctx.fillText(labelText, p.px + radius + 6, p.py + 4);

      ctx.globalAlpha = 1.0;
    });
  }

  bindEvents() {
    this.canvas.addEventListener('mousedown', e => {
      this.isDragging = true;
      this.mouseDownX = e.clientX;
      this.mouseDownY = e.clientY;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('mousemove', e => {
      if (this.isDragging) {
        const dx = e.clientX - this.lastX;
        const dy = e.clientY - this.lastY;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
        this.offsetX += dx;
        this.offsetY += dy;
        this.render();
      } else {
        this.handleHover(e);
      }
    });

    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const factor = e.deltaY < 0 ? 1.15 : 0.85;
      const newScale = Math.max(5, Math.min(300, this.scale * factor));

      this.offsetX = mouseX - (mouseX - this.offsetX) * (newScale / this.scale);
      this.offsetY = mouseY - (mouseY - this.offsetY) * (newScale / this.scale);
      this.scale = newScale;
      this.render();
    }, { passive: false });

    this.canvas.addEventListener('click', e => {
      const dx = Math.abs(e.clientX - this.mouseDownX);
      const dy = Math.abs(e.clientY - this.mouseDownY);
      if (dx > 6 || dy > 6) return;

      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let closest = null;
      let closestDist = 32 * 32;
      for (const p of this.points) {
        const ddx = p.px - mouseX;
        const ddy = p.py - mouseY;
        const dist2 = ddx * ddx + ddy * ddy;
        if (dist2 < closestDist) {
          closestDist = dist2;
          closest = p;
        }
      }
      if (closest) {
        this.selectItem(closest);
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      if (this.options.onHover) {
        this.options.onHover(null, 0, 0);
      }
    });
  }

  handleHover(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let hit = null;
    let closestDist = 26 * 26;
    for (const p of this.points) {
      const dx = p.px - mouseX;
      const dy = p.py - mouseY;
      const dist2 = dx * dx + dy * dy;
      if (dist2 < closestDist) {
        closestDist = dist2;
        hit = p;
      }
    }

    if (this.options.onHover) {
      this.options.onHover(hit, e.clientX, e.clientY);
    }
  }

  selectItem(item) {
    this.selectedItem = item;
    this.render();
    if (this.options.onSelect) {
      this.options.onSelect(item);
    }
  }
}

window.Galaxy2D = Galaxy2D;
