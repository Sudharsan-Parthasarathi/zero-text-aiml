/**
 * ZERO-TEXT: INTERACTIVE SKETCH-TO-IMAGE STUDIO
 * Interactive drawing canvas with brush tools, bucket fill, multi-color palette,
 * undo/redo history, eraser, sample sketches, and direct integration with CLIP.
 */

class SketchStudio {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.options = Object.assign({
      onSearch: null,
    }, options);

    this.isDrawing = false;
    this.strokeDrawn = false;
    this.tool = 'pen'; // 'pen', 'bucket', or 'eraser'
    this.brushSize = 7;
    this.brushColor = '#000000';
    this.lastX = 0;
    this.lastY = 0;
    this.uploadedDataUrl = null;
    this.hasUploadedImage = false;
    this.hasDrawnOverUpload = false;

    // Undo / Redo history
    this.undoStack = [];
    this.redoStack = [];
    this.maxHistory = 35;

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
    this.uploadedDataUrl = null;
    this.hasUploadedImage = false;
    this.hasDrawnOverUpload = false;
    this.undoStack = [];
    this.redoStack = [];
    this.pushState();
  }

  setTool(tool) {
    this.tool = tool;
    if (this.tool === 'bucket') {
      this.canvas.style.cursor = 'crosshair';
    } else if (this.tool === 'eraser') {
      this.canvas.style.cursor = 'cell';
    } else {
      this.canvas.style.cursor = 'crosshair';
    }
  }

  setBrushSize(size) {
    this.brushSize = size;
  }

  setBrushColor(color) {
    this.brushColor = color;
    // If eraser was active, switch to pen so color applies immediately
    if (this.tool === 'eraser') {
      this.setTool('pen');
      const toolPen = document.getElementById('tool-pen');
      const toolEraser = document.getElementById('tool-eraser');
      const toolBucket = document.getElementById('tool-bucket');
      if (toolPen) toolPen.classList.add('active');
      if (toolEraser) toolEraser.classList.remove('active');
      if (toolBucket) toolBucket.classList.remove('active');
    }
  }

  hexToRgb(hex) {
    let c = hex.replace('#', '');
    if (c.length === 3) {
      c = c.split('').map(x => x + x).join('');
    }
    const num = parseInt(c, 16);
    return {
      r: (num >> 16) & 255,
      g: (num >> 8) & 255,
      b: num & 255,
    };
  }

  // --- UNDO / REDO HISTORY ---
  pushState() {
    try {
      const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
      this.undoStack.push(imgData);
      if (this.undoStack.length > this.maxHistory) {
        this.undoStack.shift();
      }
      this.redoStack = [];
      this.updateUndoRedoUI();
    } catch (e) {
      console.warn('Canvas pushState warning:', e);
    }
  }

  undo() {
    if (this.undoStack.length <= 1) return;
    const current = this.undoStack.pop();
    this.redoStack.push(current);
    const prev = this.undoStack[this.undoStack.length - 1];
    this.ctx.putImageData(prev, 0, 0);
    this.hasDrawnOverUpload = true;
    this.updateUndoRedoUI();
  }

  redo() {
    if (this.redoStack.length === 0) return;
    const next = this.redoStack.pop();
    this.undoStack.push(next);
    this.ctx.putImageData(next, 0, 0);
    this.hasDrawnOverUpload = true;
    this.updateUndoRedoUI();
  }

  updateUndoRedoUI() {
    const btnUndo = document.getElementById('btn-undo-sketch');
    const btnRedo = document.getElementById('btn-redo-sketch');
    if (btnUndo) {
      const canUndo = this.undoStack.length > 1;
      btnUndo.disabled = !canUndo;
      btnUndo.classList.toggle('disabled', !canUndo);
    }
    if (btnRedo) {
      const canRedo = this.redoStack.length > 0;
      btnRedo.disabled = !canRedo;
      btnRedo.classList.toggle('disabled', !canRedo);
    }
  }

  // --- BUCKET FILL (FLOOD FILL) ---
  floodFill(startX, startY) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    if (startX < 0 || startX >= w || startY < 0 || startY >= h) return;

    const imgData = this.ctx.getImageData(0, 0, w, h);
    const data = imgData.data;

    const startIdx = (startY * w + startX) * 4;
    const startR = data[startIdx];
    const startG = data[startIdx + 1];
    const startB = data[startIdx + 2];
    const startA = data[startIdx + 3];

    const fillRgb = this.hexToRgb(this.brushColor);
    const fr = fillRgb.r;
    const fg = fillRgb.g;
    const fb = fillRgb.b;
    const fa = 255;

    // If click color already matches fill color, skip
    if (Math.abs(startR - fr) < 4 && Math.abs(startG - fg) < 4 && Math.abs(startB - fb) < 4 && Math.abs(startA - fa) < 4) {
      return;
    }

    const tolerance = 28;
    const colorMatches = (idx) => {
      return (
        Math.abs(data[idx] - startR) <= tolerance &&
        Math.abs(data[idx + 1] - startG) <= tolerance &&
        Math.abs(data[idx + 2] - startB) <= tolerance &&
        Math.abs(data[idx + 3] - startA) <= tolerance
      );
    };

    const visited = new Uint8Array(w * h);
    const stack = [startX, startY];
    visited[startY * w + startX] = 1;

    while (stack.length > 0) {
      const cy = stack.pop();
      const cx = stack.pop();
      const idx = (cy * w + cx) * 4;

      data[idx] = fr;
      data[idx + 1] = fg;
      data[idx + 2] = fb;
      data[idx + 3] = fa;

      const neighbors = [
        [cx + 1, cy],
        [cx - 1, cy],
        [cx, cy + 1],
        [cx, cy - 1],
      ];

      for (let i = 0; i < 4; i++) {
        const nx = neighbors[i][0];
        const ny = neighbors[i][1];
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
          const nPos = ny * w + nx;
          if (!visited[nPos]) {
            visited[nPos] = 1;
            const nIdx = nPos * 4;
            if (colorMatches(nIdx)) {
              stack.push(nx, ny);
            }
          }
        }
      }
    }

    this.ctx.putImageData(imgData, 0, 0);
    this.hasDrawnOverUpload = true;
    this.pushState();
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
      const pos = getPos(e);
      if (this.tool === 'bucket') {
        this.floodFill(Math.round(pos.x), Math.round(pos.y));
        return;
      }
      this.isDrawing = true;
      this.strokeDrawn = false;
      this.lastX = pos.x;
      this.lastY = pos.y;
      this.draw(pos.x, pos.y, false);
    });

    window.addEventListener('mouseup', () => {
      if (this.isDrawing) {
        this.isDrawing = false;
        if (this.strokeDrawn) {
          this.pushState();
          this.strokeDrawn = false;
        }
      }
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
        const pos = getPos(e.touches[0]);
        if (this.tool === 'bucket') {
          this.floodFill(Math.round(pos.x), Math.round(pos.y));
          return;
        }
        this.isDrawing = true;
        this.strokeDrawn = false;
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
      if (this.isDrawing) {
        this.isDrawing = false;
        if (this.strokeDrawn) {
          this.pushState();
          this.strokeDrawn = false;
        }
      }
    });

    // Keyboard shortcuts for Undo (Ctrl+Z) and Redo (Ctrl+Y / Ctrl+Shift+Z)
    window.addEventListener('keydown', e => {
      const modal = document.getElementById('modal-sketch');
      if (!modal || !modal.classList.contains('open')) return;

      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
        if (e.shiftKey) {
          e.preventDefault();
          this.redo();
        } else {
          e.preventDefault();
          this.undo();
        }
      } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) {
        e.preventDefault();
        this.redo();
      }
    });
  }

  draw(x, y, isMove) {
    this.hasDrawnOverUpload = true;
    this.strokeDrawn = true;
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
    this.uploadedDataUrl = null;
    this.hasUploadedImage = false;
    this.hasDrawnOverUpload = false;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      this.ctx.fillStyle = '#ffffff';
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      const cw = this.canvas.width;
      const ch = this.canvas.height;
      const scale = Math.min((cw - 40) / img.width, (ch - 40) / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      const dx = (cw - dw) / 2;
      const dy = (ch - dh) / 2;
      this.ctx.drawImage(img, dx, dy, dw, dh);
      this.undoStack = [];
      this.redoStack = [];
      this.pushState();
    };
    img.src = url;
  }

  loadImageFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      this.uploadedDataUrl = e.target.result;
      this.hasUploadedImage = true;
      this.hasDrawnOverUpload = false;
      const img = new Image();
      img.onload = () => {
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        const scale = Math.min((cw - 20) / img.width, (ch - 20) / img.height);
        const dw = img.width * scale;
        const dh = img.height * scale;
        const dx = (cw - dw) / 2;
        const dy = (ch - dh) / 2;
        this.ctx.drawImage(img, dx, dy, dw, dh);
        this.undoStack = [];
        this.redoStack = [];
        this.pushState();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
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
    if (this.hasUploadedImage && !this.hasDrawnOverUpload && this.uploadedDataUrl) {
      return this.uploadedDataUrl;
    }
    return this.canvas.toDataURL('image/png');
  }
}

window.SketchStudio = SketchStudio;
