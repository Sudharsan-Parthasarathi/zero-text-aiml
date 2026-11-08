/**
 * ZERO-TEXT: 3D INTERACTIVE VISUAL GALAXY
 * Renders semantic embeddings in 3D cosmic space.
 * Supports Three.js WebGL with orbital camera, starfield, glowing constellation webs,
 * and built-in Canvas 3D perspective projection fallback.
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
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    this.hoveredItem = null;

    // Three.js instances if available
    this.useThree = typeof window.THREE !== 'undefined';
    this.threeScene = null;
    this.threeCamera = null;
    this.threeRenderer = null;
    this.threeNodes = [];
    this.threeLinesGroup = null;
    this.raycaster = null;
    this.mouseVec = null;

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
    this.normalizeCoordinates();
    if (this.useThree) {
      this.buildThreeScene();
    }
  }

  normalizeCoordinates() {
    if (!this.items.length) return;
    const xs = this.items.map(it => it.x3d ?? it.x ?? 0);
    const ys = this.items.map(it => it.y3d ?? it.y ?? 0);
    const zs = this.items.map(it => it.z3d ?? it.z ?? 0);

    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const minZ = Math.min(...zs), maxZ = Math.max(...zs);

    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const spanZ = maxZ - minZ || 1;

    const scale = 240; // Galaxy radius
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
    this.threeScene.fog = new THREE.FogExp2(0x060913, 0.0012);

    this.threeCamera = new THREE.PerspectiveCamera(45, this.width / this.height, 1, 3000);
    this.threeCamera.position.set(0, 80, 520);

    this.threeRenderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });
    this.threeRenderer.setPixelRatio(window.devicePixelRatio);
    this.threeRenderer.setSize(this.width, this.height);

    this.raycaster = new THREE.Raycaster();
    this.mouseVec = new THREE.Vector2();

    // Cosmic Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.85);
    this.threeScene.add(ambient);
    const pointLight = new THREE.PointLight(0x70e1d1, 2, 800);
    pointLight.position.set(0, 100, 100);
    this.threeScene.add(pointLight);

    // Starfield Background
    this.buildStarfield();

    // Interaction Events
    this.bindEvents();

    // Animation Loop
    const animate = () => {
      requestAnimationFrame(animate);
      if (this.autoRotate && !this.isDragging) {
        if (this.galaxyGroup) {
          this.galaxyGroup.rotation.y += 0.002;
        }
      }
      this.threeRenderer.render(this.threeScene, this.threeCamera);
    };
    animate();
  }

  buildStarfield() {
    const THREE = window.THREE;
    const count = 1200;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      const r = 800 + Math.random() * 800;
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
        colors[i3] = 0.8; colors[i3 + 1] = 0.85; colors[i3 + 2] = 0.95; // white
      }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 2,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
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

    const sphereGeo = new THREE.SphereGeometry(1, 16, 16);

    // Nodes
    this.normalizedItems.forEach(item => {
      const clusterColor = item.cluster_color || '#4cc9f0';
      const colorHex = parseInt(clusterColor.replace('#', '0x'));

      const isSelected = this.selectedItem && this.selectedItem.filename === item.filename;
      const baseRadius = isSelected ? 12 : 7;

      const mat = new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: isSelected ? 0.9 : 0.4,
        roughness: 0.2,
        metalness: 0.8,
      });

      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.scale.setScalar(baseRadius * this.starSize);
      mesh.position.set(item.nx, item.ny, item.nz);
      mesh.userData = { item };

      // Glowing Halo
      const haloGeo = new THREE.RingGeometry(baseRadius * 1.3, baseRadius * 1.8, 24);
      const haloMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: isSelected ? 0.8 : 0.25,
        side: THREE.DoubleSide,
      });
      const halo = new THREE.Mesh(haloGeo, haloMat);
      halo.lookAt(this.threeCamera.position);
      mesh.add(halo);

      this.galaxyGroup.add(mesh);
      this.threeNodes.push(mesh);
    });

    // Constellation lines connecting nodes in the same cluster
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
        opacity: 0.25,
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

  // -------------------------------------------------------------
  // CANVAS 3D FALLBACK (Works 100% offline without Three.js)
  // -------------------------------------------------------------
  initCanvasFallback() {
    this.ctx = this.canvas.getContext('2d');
    this.bindEvents();

    const render = () => {
      this.renderCanvasFallback();
      if (this.autoRotate && !this.isDragging) {
        this.rotY += 0.003;
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

    const fov = 400;
    const cosY = Math.cos(this.rotY), sinY = Math.sin(this.rotY);
    const cosX = Math.cos(this.rotX), sinX = Math.sin(this.rotX);

    // Project points
    const projected = (this.normalizedItems || []).map(item => {
      // Rotate Y
      let x1 = item.nx * cosY - item.nz * sinY;
      let z1 = item.nx * sinY + item.nz * cosY;
      // Rotate X
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

    // Depth sort (furthest to nearest)
    projected.sort((a, b) => b.pz - a.pz);

    // Constellations
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
        ctx.strokeStyle = (group[0].cluster_color || '#4cc9f0') + '33';
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

    // Draw Nodes
    projected.forEach(p => {
      const isSelected = this.selectedItem && this.selectedItem.filename === p.filename;
      const radius = Math.max(4, (isSelected ? 14 : 7) * p.scale * this.starSize);

      ctx.beginPath();
      ctx.arc(p.px, p.py, radius * 1.8, 0, Math.PI * 2);
      ctx.fillStyle = (p.cluster_color || '#4cc9f0') + (isSelected ? '55' : '22');
      ctx.fill();

      ctx.beginPath();
      ctx.arc(p.px, p.py, radius, 0, Math.PI * 2);
      ctx.fillStyle = p.cluster_color || '#4cc9f0';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();

      // Label
      if (p.scale > 0.4 || isSelected) {
        ctx.fillStyle = isSelected ? '#70e1d1' : '#e0e8f5';
        ctx.font = `${Math.round(11 * p.scale)}px Segoe UI, sans-serif`;
        ctx.fillText(p.filename.split('.')[0], p.px + radius + 4, p.py + 4);
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
        this.threeCamera.position.z = Math.max(120, Math.min(1600, this.threeCamera.position.z / delta));
      } else {
        this.zoom = Math.max(0.4, Math.min(3.0, this.zoom * delta));
      }
    });

    this.canvas.addEventListener('click', e => {
      this.handleClick(e);
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
      const intersects = this.raycaster.intersectObjects(this.threeNodes);
      if (intersects.length > 0) {
        hitItem = intersects[0].object.userData.item;
      }
    } else if (this.fallbackProjected) {
      const centerX = this.width / 2;
      const centerY = this.height / 2;
      for (const p of this.fallbackProjected) {
        const dx = p.px + centerX - mouseX;
        const dy = p.py + centerY - mouseY;
        if (dx * dx + dy * dy < 20 * 20) {
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
      const intersects = this.raycaster.intersectObjects(this.threeNodes);
      if (intersects.length > 0) {
        hitItem = intersects[0].object.userData.item;
      }
    } else if (this.fallbackProjected) {
      const centerX = this.width / 2;
      const centerY = this.height / 2;
      for (const p of this.fallbackProjected) {
        const dx = p.px + centerX - mouseX;
        const dy = p.py + centerY - mouseY;
        if (dx * dx + dy * dy < 24 * 24) {
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
    if (this.useThree && this.threeCamera && this.galaxyGroup) {
      this.threeCamera.position.set(0, 80, 520);
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
