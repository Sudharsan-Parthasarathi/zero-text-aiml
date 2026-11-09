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

    this.threeScene.add(this.galaxyGroup);
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

  flyToItem(item) {
    if (!item) return;
    const targetItem = this.normalizedItems.find(it => it.filename === item.filename);
    if (!targetItem) return;

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
      this.selectItem(hitItem);
    }
  }

  selectItem(item) {
    this.selectedItem = item;
    if (this.useThree) {
      this.buildThreeScene();
    }
    if (this.options.onSelect) {
      this.options.onSelect(item);
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
