/**
 * ZERO-TEXT: 2D VISUAL STAR MAP EXPLORER
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

    const margin = 80;
    const scaleX = (this.width - margin * 2) / spanX;
    const scaleY = (this.height - margin * 2) / spanY;
    this.scale = Math.min(scaleX, scaleY, 60);

    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    this.offsetX = this.width / 2 - midX * this.scale;
    this.offsetY = this.height / 2 + midY * this.scale;
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

    // Group by cluster for centroid halos and connections
    const clusters = {};
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

      ctx.strokeStyle = color + '2b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < group.length; i++) {
        for (let j = i + 1; j < group.length; j++) {
          ctx.moveTo(group[i].px, group[i].py);
          ctx.lineTo(group[j].px, group[j].py);
        }
      }
      ctx.stroke();

      // Cluster centroid aura
      const avgX = group.reduce((acc, p) => acc + p.px, 0) / group.length;
      const avgY = group.reduce((acc, p) => acc + p.py, 0) / group.length;
      const radius = 60;
      const grad = ctx.createRadialGradient(avgX, avgY, 5, avgX, avgY, radius);
      grad.addColorStop(0, color + '22');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(avgX, avgY, radius, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw Image Nodes
    this.points.forEach(p => {
      const isSelected = this.selectedItem && this.selectedItem.filename === p.filename;
      const color = p.cluster_color || '#4cc9f0';
      const radius = isSelected ? 12 : 7;

      // Outer glow
      ctx.beginPath();
      ctx.arc(p.px, p.py, radius * 2, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? 'rgba(112, 225, 209, 0.35)' : (color + '1f');
      ctx.fill();

      // Node core
      ctx.beginPath();
      ctx.arc(p.px, p.py, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = isSelected ? '#ffffff' : 'rgba(255, 255, 255, 0.7)';
      ctx.lineWidth = isSelected ? 2.5 : 1;
      ctx.stroke();

      // Label
      ctx.fillStyle = isSelected ? '#70e1d1' : '#e2eaf5';
      ctx.font = isSelected ? 'bold 11px Segoe UI, sans-serif' : '10px Segoe UI, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(p.filename.split('.')[0], p.px + radius + 6, p.py + 4);
    });
  }

  bindEvents() {
    this.canvas.addEventListener('mousedown', e => {
      this.isDragging = true;
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
      const newScale = Math.max(5, Math.min(200, this.scale * factor));

      // Zoom towards mouse pointer
      this.offsetX = mouseX - (mouseX - this.offsetX) * (newScale / this.scale);
      this.offsetY = mouseY - (mouseY - this.offsetY) * (newScale / this.scale);
      this.scale = newScale;
      this.render();
    });

    this.canvas.addEventListener('click', e => {
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      for (const p of this.points) {
        const dx = p.px - mouseX;
        const dy = p.py - mouseY;
        if (dx * dx + dy * dy < 22 * 22) {
          this.selectItem(p);
          return;
        }
      }
    });
  }

  handleHover(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let hit = null;
    for (const p of this.points) {
      const dx = p.px - mouseX;
      const dy = p.py - mouseY;
      if (dx * dx + dy * dy < 18 * 18) {
        hit = p;
        break;
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
