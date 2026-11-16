/**
 * ZERO-TEXT: 3D INTERACTIVE VISUAL GALAXY (v2.5 Cosmic Spotlight & Smooth Camera Navigation)
 * Renders semantic embeddings in 3D cosmic space.
 * Features:
 * - Three.js WebGL with orbital camera, starfield, glowing constellation webs
 * - Search Spotlight: matching stars radiate bright pulsing halos; non-matches dim
 * - Smooth Camera Fly-To: camera flies and focuses directly on the top search match
 * - Next / Prev search match traversal
 * - Built-in high-performance Canvas 3D fallback
 */

class Galaxy3D {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    this.container = this.canvas.parentElement;
    this.options = Object.assign({
      onSelect: null,
      onHover: null,
    }, options);

    this.items = [];
    this.selectedItem = null;
    this.autoRotate = true;
    this.showConstellations = true;
    this.starSize = 1.0;

    // Interaction state
    this.rotX = 0.2;
    this.rotY = 0.4;
    this.zoom = 1.0;
    this.isDragging = false;
    this.mouseDownX = 0;
    this.mouseDownY = 0;
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    this.hoveredItem = null;

    // Camera animation target
    this.targetCameraPos = null;
    this.targetCameraLookAt = null;

    // Three.js instances if available
    this.useThree = typeof window.THREE !== 'undefined';
    this.threeScene = null;
    this.threeCamera = null;
    this.threeRenderer = null;
    this.threeNodes = [];
    this.raycaster = null;
    this.mouseVec = null;
    this.galaxyGroup = null;

    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());

    if (this.useThree) {
      this.initThreeJS();
    } else {
      this.initCanvasFallback();
    }
  }

  resize() {
    this.width = this.container.clientWidth || window.innerWidth;
    this.height = this.container.clientHeight || window.innerHeight;
    this.canvas.width = this.width * window.devicePixelRatio;
    this.canvas.height = this.height * window.devicePixelRatio;
    this.canvas.style.width = this.width + 'px';
    this.canvas.style.height = this.height + 'px';

    if (this.threeRenderer && this.threeCamera) {
      this.threeCamera.aspect = this.width / this.height;
      this.threeCamera.updateProjectionMatrix();
      this.threeRenderer.setSize(this.width, this.height);
    }
  }

  setItems(items) {
    this.items = items || [];
    if (this.selectedItem) {
      const still = this.items.find(it => it.filename === this.selectedItem.filename);
      if (!still) this.selectedItem = null;
    }
    this.normalizeCoordinates();
    if (this.useThree) {
      this.buildThreeScene();
    }
  }

  normalizeCoordinates() {
    if (!this.items.length) {
      this.normalizedItems = [];
      return;
    }
    const xs = this.items.map(it => it.x3d ?? it.x ?? 0);
    const ys = this.items.map(it => it.y3d ?? it.y ?? 0);
    const zs = this.items.map(it => it.z3d ?? it.z ?? 0);

    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const minZ = Math.min(...zs), maxZ = Math.max(...zs);

    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const spanZ = maxZ - minZ || 1;

    const scale = 260; // Galaxy radius
    this.normalizedItems = this.items.map(it => {
      const origX = it.x3d ?? it.x ?? 0;
      const origY = it.y3d ?? it.y ?? 0;
      const origZ = it.z3d ?? it.z ?? 0;
      return {
        ...it,
        nx: ((origX - minX) / spanX - 0.5) * scale * 2,
        ny: ((origY - minY) / spanY - 0.5) * scale * 2,
        nz: ((origZ - minZ) / spanZ - 0.5) * scale * 2,
      };
    });
  }

  // -------------------------------------------------------------
  // THREE.JS IMPLEMENTATION
  // -------------------------------------------------------------
  initThreeJS() {
    const THREE = window.THREE;
    this.threeScene = new THREE.Scene();
    this.threeScene.fog = new THREE.FogExp2(0x060913, 0.0010);

    this.threeCamera = new THREE.PerspectiveCamera(45, this.width / this.height, 1, 3500);
    this.threeCamera.position.set(0, 70, 560);

    this.threeRenderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });
    this.threeRenderer.setPixelRatio(window.devicePixelRatio);
    this.threeRenderer.setSize(this.width, this.height);

    this.raycaster = new THREE.Raycaster();
    this.mouseVec = new THREE.Vector2();

    const ambient = new THREE.AmbientLight(0xffffff, 0.9);
    this.threeScene.add(ambient);

    const pointLight = new THREE.PointLight(0x70e1d1, 2.5, 900);
    pointLight.position.set(0, 150, 150);
    this.threeScene.add(pointLight);

    this.buildStarfield();
    this.bindEvents();

    let clock = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      clock += 0.02;

      // Smooth camera fly-to interpolation
      if (this.targetCameraPos && this.threeCamera) {
        this.threeCamera.position.lerp(this.targetCameraPos, 0.06);
        if (this.threeCamera.position.distanceTo(this.targetCameraPos) < 2) {
          this.targetCameraPos = null;
          if (this.onFlyComplete) {
            const cb = this.onFlyComplete;
            this.onFlyComplete = null;
            cb();
          }
        }
      }

      // Gentle auto-rotate
      if (this.autoRotate && !this.isDragging && !this.targetCameraPos) {
        if (this.galaxyGroup) {
          this.galaxyGroup.rotation.y += 0.0018;
        }
      }

      // Pulse halos of top matches
      if (this.threeNodes) {
        this.threeNodes.forEach(node => {
          if (node.userData && node.userData.haloMesh) {
            const isMatch = (node.userData.item.final || 0) > 0.4;
            const pulse = 1.0 + Math.sin(clock * 3) * (isMatch ? 0.2 : 0.08);
            node.userData.haloMesh.scale.set(pulse, pulse, 1);
          }
        });
      }

      // Animate breathing volumetric nebula clouds
      if (this.nebulaMeshes && this.nebulaMeshes.length) {
        this.nebulaMeshes.forEach((neb, idx) => {
          const breath = 1.0 + Math.sin(clock * 0.9 + idx * 0.8) * 0.08;
          neb.scale.set(neb.userData.baseScale * breath, neb.userData.baseScale * breath, 1);
          neb.rotation.z += neb.userData.rotSpeed;
          neb.lookAt(this.threeCamera.position);
        });
      }

      // Animate cosmic stardust drift & twinkle
      if (this.stardustPoints && this.stardustPoints.geometry) {
        const positions = this.stardustPoints.geometry.attributes.position.array;
        for (let i = 1; i < positions.length; i += 3) {
          positions[i] += Math.sin(clock * 1.2 + i * 0.01) * 0.10;
        }
        this.stardustPoints.geometry.attributes.position.needsUpdate = true;
      }

      this.threeRenderer.render(this.threeScene, this.threeCamera);
    };
    animate();
  }

  buildStarfield() {
    const THREE = window.THREE;
    const count = 1500;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      const r = 900 + Math.random() * 900;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      positions[i3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = r * Math.cos(phi);

      const tint = Math.random();
      if (tint > 0.8) {
        colors[i3] = 0.44; colors[i3 + 1] = 0.88; colors[i3 + 2] = 0.82; // cyan
      } else if (tint > 0.6) {
        colors[i3] = 0.78; colors[i3 + 1] = 0.49; colors[i3 + 2] = 1.0; // purple
      } else {
        colors[i3] = 0.85; colors[i3 + 1] = 0.90; colors[i3 + 2] = 0.98; // white
      }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 2.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
    });

    const starPoints = new THREE.Points(geometry, material);
    this.threeScene.add(starPoints);
  }

  buildThreeScene() {
    const THREE = window.THREE;
    if (this.galaxyGroup) {
      this.threeScene.remove(this.galaxyGroup);
    }
    this.galaxyGroup = new THREE.Group();
    this.threeNodes = [];

    if (!this.normalizedItems || !this.normalizedItems.length) {
      this.threeScene.add(this.galaxyGroup);
      return;
    }

    const sphereGeo = new THREE.SphereGeometry(1, 20, 20);
    const hasSearchMatches = this.normalizedItems.some(it => (it.final || 0) > 0.2);

    this.normalizedItems.forEach(item => {
      const clusterColor = item.cluster_color || '#4cc9f0';
      const colorHex = parseInt(clusterColor.replace('#', '0x'));

      const isSelected = this.selectedItem && this.selectedItem.filename === item.filename;
      const score = item.final || 0;
      const isTopMatch = hasSearchMatches && score > 0.4;
      const isDimmed = hasSearchMatches && score <= 0.15 && !isSelected;

      // Base radius scaled by relevance
      let baseRadius = isSelected ? 15 : (isTopMatch ? 11 : 7);

      const mat = new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: isSelected ? 1.0 : (isTopMatch ? 0.8 : (isDimmed ? 0.15 : 0.45)),
        roughness: 0.2,
        metalness: 0.8,
        transparent: true,
        opacity: isDimmed ? 0.25 : 1.0,
      });

      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.scale.setScalar(baseRadius * this.starSize);
      mesh.position.set(item.nx, item.ny, item.nz);
      mesh.userData = { item };

      // Radiant Halo
      const haloGeo = new THREE.RingGeometry(baseRadius * 1.2, baseRadius * (isSelected ? 2.3 : (isTopMatch ? 2.0 : 1.6)), 32);
      const haloMat = new THREE.MeshBasicMaterial({
        color: isSelected ? 0x70e1d1 : colorHex,
        transparent: true,
        opacity: isSelected ? 0.85 : (isTopMatch ? 0.6 : (isDimmed ? 0.08 : 0.25)),
        side: THREE.DoubleSide,
      });
      const halo = new THREE.Mesh(haloGeo, haloMat);
      halo.lookAt(this.threeCamera.position);
      mesh.add(halo);
      mesh.userData.haloMesh = halo;

      this.galaxyGroup.add(mesh);
      this.threeNodes.push(mesh);
    });

    if (this.showConstellations) {
      this.buildConstellationLines();
    }

    this.buildCosmicStardust();
    this.buildNebulaClouds();

    this.threeScene.add(this.galaxyGroup);
  }

  getNebulaTexture() {
    if (this._nebulaTexture) return this._nebulaTexture;
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(255, 255, 255, 0.85)');
    grad.addColorStop(0.25, 'rgba(255, 255, 255, 0.50)');
    grad.addColorStop(0.55, 'rgba(255, 255, 255, 0.16)');
    grad.addColorStop(0.85, 'rgba(255, 255, 255, 0.03)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    const THREE = window.THREE;
    this._nebulaTexture = new THREE.CanvasTexture(canvas);
    return this._nebulaTexture;
  }

  buildCosmicStardust() {
    const THREE = window.THREE;
    const dustCount = 2000;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(dustCount * 3);
    const col = new Float32Array(dustCount * 3);

    for (let i = 0; i < dustCount; i++) {
      const i3 = i * 3;
      // Scatter throughout the active galaxy core
      pos[i3] = (Math.random() - 0.5) * 720;
      pos[i3 + 1] = (Math.random() - 0.5) * 720;
      pos[i3 + 2] = (Math.random() - 0.5) * 720;

      const pick = Math.random();
      if (pick > 0.7) {
        // Cyan stardust
        col[i3] = 0.44; col[i3 + 1] = 0.88; col[i3 + 2] = 0.85;
      } else if (pick > 0.4) {
        // Magenta / Rose stardust
        col[i3] = 0.85; col[i3 + 1] = 0.35; col[i3 + 2] = 0.65;
      } else if (pick > 0.2) {
        // Electric Gold stardust
        col[i3] = 1.0; col[i3 + 1] = 0.82; col[i3 + 2] = 0.45;
      } else {
        // Soft white starlight
        col[i3] = 0.9; col[i3 + 1] = 0.95; col[i3 + 2] = 1.0;
      }
    }

    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));

    const mat = new THREE.PointsMaterial({
      size: 1.8,
      vertexColors: true,
      transparent: true,
      opacity: 0.60,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    this.stardustPoints = new THREE.Points(geo, mat);
    this.galaxyGroup.add(this.stardustPoints);
  }

  buildNebulaClouds() {
    const THREE = window.THREE;
    this.nebulaMeshes = [];
    if (!this.normalizedItems || !this.normalizedItems.length) return;

    // Group items by cluster
    const clusters = {};
    this.normalizedItems.forEach(it => {
      const c = it.cluster ?? -1;
      if (c === -1) return;
      if (!clusters[c]) clusters[c] = [];
      clusters[c].push(it);
    });

    const texture = this.getNebulaTexture();
    const planeGeo = new THREE.PlaneGeometry(1, 1);

    Object.entries(clusters).forEach(([cId, items]) => {
      if (!items.length) return;
      // Calculate cluster centroid & radius
      let sumX = 0, sumY = 0, sumZ = 0;
      items.forEach(it => {
        sumX += it.nx; sumY += it.ny; sumZ += it.nz;
      });
      const cx = sumX / items.length;
      const cy = sumY / items.length;
      const cz = sumZ / items.length;

      let maxDist = 70;
      items.forEach(it => {
        const dx = it.nx - cx, dy = it.ny - cy, dz = it.nz - cz;
        const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (d > maxDist) maxDist = d;
      });

      const clusterColor = items[0].cluster_color || '#4cc9f0';
      const colorHex = parseInt(clusterColor.replace('#', '0x'), 16);

      // Create overlapping volumetric nebula clouds
      const puffCount = Math.min(3, Math.max(2, Math.floor(items.length / 3) + 1));
      for (let p = 0; p < puffCount; p++) {
        const mat = new THREE.MeshBasicMaterial({
          map: texture,
          color: colorHex,
          transparent: true,
          opacity: 0.12 + (p === 0 ? 0.05 : 0),
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          side: THREE.DoubleSide
        });

        const mesh = new THREE.Mesh(planeGeo, mat);
        const offsetX = (Math.random() - 0.5) * (maxDist * 0.4);
        const offsetY = (Math.random() - 0.5) * (maxDist * 0.4);
        const offsetZ = (Math.random() - 0.5) * (maxDist * 0.4);
        mesh.position.set(cx + offsetX, cy + offsetY, cz + offsetZ);

        const baseScale = maxDist * (1.8 + Math.random() * 0.6);
        mesh.scale.set(baseScale, baseScale, 1);
        mesh.rotation.z = Math.random() * Math.PI * 2;
        mesh.userData = {
          baseScale,
          rotSpeed: (Math.random() - 0.5) * 0.0010
        };

        this.galaxyGroup.add(mesh);
        this.nebulaMeshes.push(mesh);
      }
    });
  }

  buildConstellationLines() {
    const THREE = window.THREE;
    const clusters = {};
    this.normalizedItems.forEach(item => {
      if (item.cluster === -1) return;
      if (!clusters[item.cluster]) clusters[item.cluster] = [];
      clusters[item.cluster].push(item);
    });

    Object.entries(clusters).forEach(([cId, group]) => {
      if (group.length < 2) return;
      const clusterColor = group[0].cluster_color || '#4cc9f0';
      const colorHex = parseInt(clusterColor.replace('#', '0x'));
      const lineMat = new THREE.LineBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.20,
      });

      const lineGeo = new THREE.BufferGeometry();
      const points = [];
      for (let i = 0; i < group.length; i++) {
        for (let j = i + 1; j < group.length; j++) {
          points.push(new THREE.Vector3(group[i].nx, group[i].ny, group[i].nz));
          points.push(new THREE.Vector3(group[j].nx, group[j].ny, group[j].nz));
        }
      }
      lineGeo.setFromPoints(points);
      const lines = new THREE.LineSegments(lineGeo, lineMat);
      this.galaxyGroup.add(lines);
    });
  }

  flyToItem(item, onComplete = null) {
    if (!item) return;
    this.onFlyComplete = typeof onComplete === 'function' ? onComplete : null;
    const targetItem = this.normalizedItems.find(it => it.filename === item.filename);
    if (!targetItem) {
      if (this.onFlyComplete) {
        const cb = this.onFlyComplete;
        this.onFlyComplete = null;
        cb();
      }
      return;
    }

    if (this.useThree && this.threeCamera) {
      const THREE = window.THREE;
      // Position camera in front of node
      this.targetCameraPos = new THREE.Vector3(
        targetItem.nx,
        targetItem.ny + 30,
        targetItem.nz + 190
      );
      this.autoRotate = false;
    } else {
      this.rotX = 0.1;
      this.rotY = 0.2;
      this.zoom = 1.6;
      if (this.renderCanvasFallback) this.renderCanvasFallback();
      if (this.onFlyComplete) {
        setTimeout(() => {
          if (this.onFlyComplete) {
            const cb = this.onFlyComplete;
            this.onFlyComplete = null;
            cb();
          }
        }, 800);
      }
    }
  }

  // -------------------------------------------------------------
  // CANVAS 3D FALLBACK
  // -------------------------------------------------------------
  initCanvasFallback() {
    this.ctx = this.canvas.getContext('2d');
    this.bindEvents();

    const render = () => {
      this.renderCanvasFallback();
      if (this.autoRotate && !this.isDragging) {
        this.rotY += 0.0025;
      }
      requestAnimationFrame(render);
    };
    render();
  }

  renderCanvasFallback() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(w / 2, h / 2);

    const fov = 450;
    const cosY = Math.cos(this.rotY), sinY = Math.sin(this.rotY);
    const cosX = Math.cos(this.rotX), sinX = Math.sin(this.rotX);

    const projected = (this.normalizedItems || []).map(item => {
      let x1 = item.nx * cosY - item.nz * sinY;
      let z1 = item.nx * sinY + item.nz * cosY;
      let y2 = item.ny * cosX - z1 * sinX;
      let z2 = item.ny * sinX + z1 * cosX + 600 / this.zoom;

      const scale = fov / (fov + z2);
      return {
        ...item,
        px: x1 * scale,
        py: y2 * scale,
        pz: z2,
        scale: Math.max(0.2, scale),
      };
    });

    projected.sort((a, b) => b.pz - a.pz);

    if (this.showConstellations) {
      ctx.lineWidth = 1;
      const clusters = {};
      projected.forEach(p => {
        if (p.cluster === -1) return;
        if (!clusters[p.cluster]) clusters[p.cluster] = [];
        clusters[p.cluster].push(p);
      });

      Object.values(clusters).forEach(group => {
        if (group.length < 2) return;
        ctx.strokeStyle = (group[0].cluster_color || '#4cc9f0') + '2e';
        ctx.beginPath();
        for (let i = 0; i < group.length; i++) {
          for (let j = i + 1; j < group.length; j++) {
            ctx.moveTo(group[i].px, group[i].py);
            ctx.lineTo(group[j].px, group[j].py);
          }
        }
        ctx.stroke();
      });
    }

    const hasMatches = projected.some(p => (p.final || 0) > 0.2);

    projected.forEach(p => {
      const isSelected = this.selectedItem && this.selectedItem.filename === p.filename;
      const score = p.final || 0;
      const isTopMatch = hasMatches && score > 0.4;
      const isDimmed = hasMatches && score <= 0.15 && !isSelected;

      const radius = Math.max(5, (isSelected ? 16 : (isTopMatch ? 12 : 7)) * p.scale * this.starSize);

      // Glow halo
      ctx.beginPath();
      ctx.arc(p.px, p.py, radius * 2.2, 0, Math.PI * 2);
      ctx.fillStyle = (p.cluster_color || '#4cc9f0') + (isSelected ? '77' : (isTopMatch ? '55' : (isDimmed ? '11' : '22')));
      ctx.fill();

      // Core
      ctx.beginPath();
      ctx.arc(p.px, p.py, radius, 0, Math.PI * 2);
      ctx.fillStyle = p.cluster_color || '#4cc9f0';
      ctx.globalAlpha = isDimmed ? 0.25 : 1.0;
      ctx.fill();
      ctx.strokeStyle = isSelected ? '#70e1d1' : (isTopMatch ? '#ffffff' : 'rgba(255,255,255,0.7)');
      ctx.lineWidth = isSelected ? 2.8 : (isTopMatch ? 2 : 1);
      ctx.stroke();
      ctx.globalAlpha = 1.0;

      // Label & Match Badge
      if (p.scale > 0.35 || isSelected || isTopMatch) {
        ctx.fillStyle = isSelected ? '#70e1d1' : (isTopMatch ? '#ffd166' : '#e0e8f5');
        ctx.font = `${Math.round(isSelected || isTopMatch ? 12 : 10 * p.scale)}px Segoe UI, sans-serif`;
        const text = isTopMatch ? `${p.filename.split('.')[0]} (${Math.round(score * 100)}%)` : p.filename.split('.')[0];
        ctx.fillText(text, p.px + radius + 5, p.py + 4);
      }
    });

    this.fallbackProjected = projected;
    ctx.restore();
  }

  // -------------------------------------------------------------
  // USER INTERACTION & CONTROLS
  // -------------------------------------------------------------
  bindEvents() {
    this.canvas.addEventListener('mousedown', e => {
      this.isDragging = true;
      this.mouseDownX = e.clientX;
      this.mouseDownY = e.clientY;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('mousemove', e => {
      if (this.isDragging) {
        const dx = e.clientX - this.lastMouseX;
        const dy = e.clientY - this.lastMouseY;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;

        if (this.useThree && this.galaxyGroup) {
          this.galaxyGroup.rotation.y += dx * 0.005;
          this.galaxyGroup.rotation.x += dy * 0.005;
        } else {
          this.rotY += dx * 0.005;
          this.rotX += dy * 0.005;
        }
      } else {
        this.handleHover(e);
      }
    });

    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      if (this.useThree && this.threeCamera) {
        this.threeCamera.position.z = Math.max(80, Math.min(1800, this.threeCamera.position.z / delta));
      } else {
        this.zoom = Math.max(0.4, Math.min(3.5, this.zoom * delta));
      }
    }, { passive: false });

    this.canvas.addEventListener('click', e => {
      const dx = Math.abs(e.clientX - this.mouseDownX);
      const dy = Math.abs(e.clientY - this.mouseDownY);
      if (dx > 6 || dy > 6) return;
      this.handleClick(e);
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

    let hitItem = null;

    if (this.useThree && this.raycaster && this.threeNodes.length) {
      this.mouseVec.x = (mouseX / this.width) * 2 - 1;
      this.mouseVec.y = -(mouseY / this.height) * 2 + 1;
      this.raycaster.setFromCamera(this.mouseVec, this.threeCamera);
      const intersects = this.raycaster.intersectObjects(this.threeNodes, true);
      if (intersects.length > 0) {
        let obj = intersects[0].object;
        while (obj && !obj.userData.item) obj = obj.parent;
        if (obj && obj.userData.item) hitItem = obj.userData.item;
      }
    } else if (this.fallbackProjected) {
      const centerX = this.width / 2;
      const centerY = this.height / 2;
      for (const p of this.fallbackProjected) {
        const ddx = p.px + centerX - mouseX;
        const ddy = p.py + centerY - mouseY;
        if (ddx * ddx + ddy * ddy < 28 * 28) {
          hitItem = p;
          break;
        }
      }
    }

    if (this.options.onHover) {
      this.options.onHover(hitItem, e.clientX, e.clientY);
    }
  }

  handleClick(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let hitItem = null;

    if (this.useThree && this.raycaster && this.threeNodes.length) {
      this.mouseVec.x = (mouseX / this.width) * 2 - 1;
      this.mouseVec.y = -(mouseY / this.height) * 2 + 1;
      this.raycaster.setFromCamera(this.mouseVec, this.threeCamera);
      const intersects = this.raycaster.intersectObjects(this.threeNodes, true);
      if (intersects.length > 0) {
        let obj = intersects[0].object;
        while (obj && !obj.userData.item) obj = obj.parent;
        if (obj && obj.userData.item) hitItem = obj.userData.item;
      }
    } else if (this.fallbackProjected) {
      const centerX = this.width / 2;
      const centerY = this.height / 2;
      for (const p of this.fallbackProjected) {
        const ddx = p.px + centerX - mouseX;
        const ddy = p.py + centerY - mouseY;
        if (ddx * ddx + ddy * ddy < 32 * 32) {
          hitItem = p;
          break;
        }
      }
    }

    if (hitItem) {
      if (this.selectedItem && this.selectedItem.filename === hitItem.filename) {
        // Toggle selection off if clicking the already selected item
        this.deselectItem();
      } else {
        this.selectItem(hitItem);
      }
    } else {
      // Clicking empty 3D space unselects
      if (this.selectedItem) {
        this.deselectItem();
      }
    }
  }

  selectItem(item) {
    this.selectedItem = item;
    if (this.useThree) {
      this.buildThreeScene();
    } else if (this.renderCanvasFallback) {
      this.renderCanvasFallback();
    }
    if (this.options.onSelect) {
      this.options.onSelect(item);
    }
  }

  deselectItem() {
    this.selectedItem = null;
    if (this.useThree) {
      this.buildThreeScene();
    } else if (this.renderCanvasFallback) {
      this.renderCanvasFallback();
    }
    if (this.options.onSelect) {
      this.options.onSelect(null);
    }
  }

  resetCamera() {
    this.rotX = 0.2;
    this.rotY = 0.4;
    this.zoom = 1.0;
    this.targetCameraPos = null;
    if (this.useThree && this.threeCamera && this.galaxyGroup) {
      this.threeCamera.position.set(0, 70, 560);
      this.galaxyGroup.rotation.set(0, 0, 0);
    }
  }

  toggleAutoRotate() {
    this.autoRotate = !this.autoRotate;
    return this.autoRotate;
  }

  toggleConstellations() {
    this.showConstellations = !this.showConstellations;
    if (this.useThree) {
      this.buildThreeScene();
    }
    return this.showConstellations;
  }
}

window.Galaxy3D = Galaxy3D;
