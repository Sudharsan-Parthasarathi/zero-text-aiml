/**
 * ZERO-TEXT: INTERACTIVE SKETCH-TO-IMAGE STUDIO
 * Interactive drawing canvas with brush tools, eraser, sample sketches,
 * and direct integration with the CLIP sketch search engine.
 */

class SketchStudio {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.options = Object.assign({
      onSearch: null,
    }, options);

    this.isDrawing = false;
    this.tool = 'pen'; // 'pen' or 'eraser'
    this.brushSize = 6;
    this.brushColor = '#000000';
    this.lastX = 0;
    this.lastY = 0;

    this.initCanvas();
    this.bindEvents();
    this.loadSampleSketches();
  }

  initCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width || 560;
    this.canvas.height = rect.height || 340;
    this.clear();
  }

  clear() {
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  setTool(tool) {
    this.tool = tool;
  }

  setBrushSize(size) {
    this.brushSize = size;
  }

  setBrushColor(color) {
    this.brushColor = color;
    this.tool = 'pen';
  }

  bindEvents() {
    const getPos = e => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.canvas.width / rect.width;
      const scaleY = this.canvas.height / rect.height;
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      };
    };

    // Mouse events
    this.canvas.addEventListener('mousedown', e => {
      this.isDrawing = true;
      const pos = getPos(e);
      this.lastX = pos.x;
      this.lastY = pos.y;
      this.draw(pos.x, pos.y, false);
    });

    window.addEventListener('mouseup', () => {
      this.isDrawing = false;
    });

    this.canvas.addEventListener('mousemove', e => {
      if (!this.isDrawing) return;
      const pos = getPos(e);
      this.draw(pos.x, pos.y, true);
      this.lastX = pos.x;
      this.lastY = pos.y;
    });

    // Touch events for tablets / laptops
    this.canvas.addEventListener('touchstart', e => {
      e.preventDefault();
      if (e.touches.length === 1) {
        this.isDrawing = true;
        const pos = getPos(e.touches[0]);
        this.lastX = pos.x;
        this.lastY = pos.y;
        this.draw(pos.x, pos.y, false);
      }
    });

    this.canvas.addEventListener('touchmove', e => {
      e.preventDefault();
      if (!this.isDrawing || e.touches.length !== 1) return;
      const pos = getPos(e.touches[0]);
      this.draw(pos.x, pos.y, true);
      this.lastX = pos.x;
      this.lastY = pos.y;
    });

    this.canvas.addEventListener('touchend', e => {
      e.preventDefault();
      this.isDrawing = false;
    });
  }

  draw(x, y, isMove) {
    const ctx = this.ctx;
    ctx.lineWidth = this.brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (this.tool === 'eraser') {
      ctx.strokeStyle = '#ffffff';
    } else {
      ctx.strokeStyle = this.brushColor;
    }

    ctx.beginPath();
    if (isMove) {
      ctx.moveTo(this.lastX, this.lastY);
      ctx.lineTo(x, y);
      ctx.stroke();
    } else {
      ctx.arc(x, y, this.brushSize / 2, 0, Math.PI * 2);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fill();
    }
  }

  loadSample(url) {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      this.clear();
      // Draw centered with aspect fit
      const cw = this.canvas.width;
      const ch = this.canvas.height;
      const scale = Math.min((cw - 40) / img.width, (ch - 40) / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      const dx = (cw - dw) / 2;
      const dy = (ch - dh) / 2;
      this.ctx.drawImage(img, dx, dy, dw, dh);
    };
    img.src = url;
  }

  loadSampleSketches() {
    fetch('/api/sketches')
      .then(r => r.json())
      .then(sketches => {
        const row = document.getElementById('sample-sketches-row');
        if (!row) return;
        row.innerHTML = '';
        sketches.forEach(s => {
          const chip = document.createElement('span');
          chip.className = 'sample-sketch-chip';
          chip.textContent = s.filename.split('.')[0];
          chip.addEventListener('click', () => {
            this.loadSample(s.url);
          });
          row.appendChild(chip);
        });
      })
      .catch(err => console.log('Sample sketches not available:', err));
  }

  toDataURL() {
    return this.canvas.toDataURL('image/png');
  }
}

window.SketchStudio = SketchStudio;
