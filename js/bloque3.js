/* ===========================================================================
 * bloque3.js — Regla de aprendizaje del perceptrón.
 * También lo usa bloque4.html: si la página tiene #dataset-select, #stats o
 * #btn-learn-many, se activan el selector de datasets no lineales, las
 * estadísticas por época y el entrenamiento de varias épocas.
 * =========================================================================== */
(function () {
  "use strict";

  /* ----------------------------------------------------------------------
   * Configuración inicial del modelo.
   * -------------------------------------------------------------------- */
  const INITIAL_INPUTS  = [1.0, -2.0];
  const INITIAL_WEIGHTS = [0.8, -1.2];
  const INITIAL_BIAS    = -0.5;
  const INITIAL_ACTIVATION = "binaryStep"; // regla clásica del perceptrón

  // Rango de visualización en el plano cartesiano
  const xMin = -6, xMax = 6;
  const yMin = -6, yMax = 6;

  // Relleno del área de graficado (padding)
  const padL = 36;
  const padR = 16;
  const padT = 16;
  const padB = 28;

  // Parámetros geométricos para el SVG de la neurona
  const DIAGRAM = {
    W: 900,
    xInputs: 110,
    xSum: 430,
    xAct: 660,
    xOut: 840,
    rInput: 26,
    rBias: 26,
    rSum: 54,
    rOut: 30,
    actW: 120,
    actH: 90,
    biasGap: 70,
    padTop: 60,
    padBottom: 30,
    step: 70
  };

  const height = 360;
  const centerY = 180;
  const inputY = [145, 215];
  const biasY = 304;

  let neuron = null;
  let dataset = [];
  let flowTimeouts = [];

  // Parámetros de aprendizaje
  let learningRate = 0.1;
  let currentTrainingIndex = 0;
  let highlightedPoint = null;
  let epochTimer = null;

  // Almacenar referencias de sincronización de los controles
  let inputControls = [];
  let weightControls = [];
  let biasControl = null;
  let lrControl = null;

  /* ----------------------------------------------------------------------
   * Generación de números aleatorios con distribución normal (Box-Muller).
   * -------------------------------------------------------------------- */
  function randomNormal(mean = 0, stdDev = 1) {
    const u1 = Math.random();
    const u2 = Math.random();
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + stdDev * randStdNormal;
  }

  /** Crea un punto recortado al área visible del plano. */
  function point(x, y, label) {
    return {
      x: Math.max(xMin + 0.5, Math.min(xMax - 0.5, x)),
      y: Math.max(yMin + 0.5, Math.min(yMax - 0.5, y)),
      label
    };
  }

  /* Conjuntos de datos (40 puntos cada uno). Clase 0 = rojos, 1 = azules.
   * Solo "separable" puede separarse con una recta; el resto ilustra el
   * límite del perceptrón (Bloque 4). */
  const DATASETS = {
    // Dos nubes: rojos en (-2,-2), azules en (2,2).
    separable: () => {
      const pts = [];
      for (let i = 0; i < 20; i++) pts.push(point(randomNormal(-2, 1.2), randomNormal(-2, 1.2), 0));
      for (let i = 0; i < 20; i++) pts.push(point(randomNormal(2, 1.2), randomNormal(2, 1.2), 1));
      return pts;
    },
    // XOR: azules en los cuadrantes II y IV, rojos en I y III.
    xor: () => {
      const pts = [];
      [[-2.5, 2.5, 1], [2.5, -2.5, 1], [2.5, 2.5, 0], [-2.5, -2.5, 0]].forEach(([cx, cy, label]) => {
        for (let i = 0; i < 10; i++) pts.push(point(randomNormal(cx, 0.9), randomNormal(cy, 0.9), label));
      });
      return pts;
    },
    // Círculos: azules en un disco central, rojos en un anillo alrededor.
    circles: () => {
      const pts = [];
      for (let i = 0; i < 40; i++) {
        const inner = i < 20;
        const r = inner ? Math.random() * 1.8 : 3.4 + Math.random() * 1.6;
        const t = Math.random() * 2 * Math.PI;
        pts.push(point(r * Math.cos(t), r * Math.sin(t), inner ? 1 : 0));
      }
      return pts;
    },
    // Lunas: dos medias lunas entrelazadas.
    moons: () => {
      const pts = [];
      for (let i = 0; i < 20; i++) {
        const t = Math.PI * Math.random();
        pts.push(point(3 * Math.cos(t) - 1.5 + randomNormal(0, 0.3), 3 * Math.sin(t) - 0.75 + randomNormal(0, 0.3), 0));
        pts.push(point(3 * (1 - Math.cos(t)) - 1.5 + randomNormal(0, 0.3), 3 * (0.5 - Math.sin(t)) - 0.75 + randomNormal(0, 0.3), 1));
      }
      return pts;
    }
  };

  function generateDataset() {
    const select = document.getElementById("dataset-select");
    dataset = DATASETS[select ? select.value : "separable"]();
    currentTrainingIndex = 0;
    epochCount = 0;
    epochLog = [];
  }

  /* ----------------------------------------------------------------------
   * Estadísticas de entrenamiento (solo si la página tiene #stats).
   * -------------------------------------------------------------------- */
  let epochCount = 0;
  let epochLog = [];  // % de aciertos al final de cada época

  /** Nº de puntos clasificados correctamente (mismo criterio que la frontera). */
  function countCorrect() {
    const [w1, w2] = neuron.weights;
    return dataset.filter((p) => (p.x * w1 + p.y * w2 + neuron.bias >= 0 ? 1 : 0) === p.label).length;
  }

  function updateStats() {
    const el = document.getElementById("stats");
    if (!el || !neuron) return;
    const ok = countCorrect();
    el.innerHTML =
      `<div>Época: <strong>${epochCount}</strong> · Aciertos: <strong>${ok}/${dataset.length}</strong> ` +
      `(${Math.round((100 * ok) / dataset.length)}%)</div>` +
      `<div class="epoch-log">${epochLog.map((a, i) => `<span>Ép ${i + 1}: ${a}%</span>`).join("")}</div>`;
  }

  /** Registra el fin de una época. Devuelve true si clasifica todo bien. */
  function finishEpoch() {
    const ok = countCorrect();
    epochCount++;
    epochLog.push(Math.round((100 * ok) / dataset.length));
    updateStats();
    return ok === dataset.length;
  }

  /* ----------------------------------------------------------------------
   * Constructor dinámico de filas de control (Sliders + Inputs numéricos).
   * -------------------------------------------------------------------- */
  function buildControlRow(parent, cfg) {
    const row = document.createElement("div");
    row.className = "ctrl-row";
    row.style.setProperty("--accent", cfg.color);

    const header = document.createElement("div");
    header.className = "ctrl-head";
    header.innerHTML =
      `<span class="ctrl-label">${cfg.label}</span>` +
      `<input class="ctrl-num" type="number" ` +
      `min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" ` +
      `value="${Number(cfg.value.toFixed(cfg.decimals))}">`;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "ctrl-slider";
    slider.min = cfg.min;
    slider.max = cfg.max;
    slider.step = cfg.step;
    slider.value = cfg.value;

    row.appendChild(header);
    row.appendChild(slider);
    parent.appendChild(row);

    const numInput = header.querySelector(".ctrl-num");

    const updateSliderFill = () => {
      const val = parseFloat(slider.value);
      const pct = ((val - cfg.min) / (cfg.max - cfg.min)) * 100;
      slider.style.setProperty("--fill", `${Math.max(0, Math.min(100, pct))}%`);
    };

    const apply = (v, transient = false) => {
      const clamped = Math.min(cfg.max, Math.max(cfg.min, v));
      slider.value = clamped;
      numInput.value = Number(clamped.toFixed(cfg.decimals)).toString();
      updateSliderFill();
      cfg.onChange(clamped, transient);
    };

    slider.addEventListener("input", () => apply(parseFloat(slider.value), true));
    slider.addEventListener("change", () => apply(parseFloat(slider.value), false));
    numInput.addEventListener("input", () => {
      const v = parseFloat(numInput.value);
      if (!Number.isNaN(v)) apply(v, true);
    });
    numInput.addEventListener("change", () => {
      const v = parseFloat(numInput.value);
      apply(Number.isNaN(v) ? 0 : v, false);
    });

    updateSliderFill();

    return {
      setValue: (v) => {
        slider.value = v;
        numInput.value = Number(v.toFixed(cfg.decimals)).toString();
        updateSliderFill();
      }
    };
  }

  /* ----------------------------------------------------------------------
   * Constructor dinámico del selector de funciones de activación.
   * -------------------------------------------------------------------- */
  function buildActivationSelector(wrap, neuronInstance, onActivationChange) {
    wrap.innerHTML = "";
    const funcs = window.ActivationFunctions;

    Object.keys(funcs).forEach((key) => {
      const meta = funcs[key];
      const card = document.createElement("button");
      card.type = "button";
      card.className = "act-card";
      card.dataset.key = key;
      if (key === neuronInstance.activationFunction) card.classList.add("active");

      card.innerHTML =
        `<span class="act-name">${meta.name}</span>` +
        `<code class="act-eq">${meta.equation}</code>` +
        `<span class="act-desc">${meta.description}</span>`;

      card.addEventListener("click", () => {
        if (typeof onActivationChange === "function") {
          onActivationChange(key);
        }
      });
      wrap.appendChild(card);
    });
  }

  /**
   * Renderiza una expresión en LaTeX usando KaTeX, con un fallback en texto
   * legible por si KaTeX no está disponible.
   * @param {string} latex
   * @param {HTMLElement} element
   */
  function renderLatex(latex, element) {
    if (!element) return;
    if (window.katex) {
      try {
        window.katex.render(latex, element, { throwOnError: false, displayMode: false });
        return;
      } catch (e) {
        console.error("Error al renderizar KaTeX:", e);
      }
    }
    // Fallback: Quitar comandos LaTeX comunes para mostrar texto plano legible
    let plain = latex
      .replace(/\\text\{([^}]+)\}/g, "$1") // \text{hola} -> hola
      .replace(/\\mathbf\{([^}]+)\}/g, "$1") // \mathbf{1.2} -> 1.2
      .replace(/\\cdot/g, "·")             // \cdot -> ·
      .replace(/\\Delta/g, "Δ")            // \Delta -> Δ
      .replace(/\\eta/g, "η")              // \eta -> η
      .replace(/\\mathbf\{([^}]+)\}/g, "$1") // \mathbf{x} -> x
      .replace(/\\leftarrow/g, "←")        // \leftarrow -> ←
      .replace(/\\begin\{aligned\}/g, "")   // aligned env start
      .replace(/\\end\{aligned\}/g, "")     // aligned env end
      .replace(/\\\\/g, "\n")              // newline
      .replace(/&=/g, "=")                 // &= -> =
      .replace(/&/g, " ")                  // alignment ampersand
      .replace(/_(\d+)/g, "$1")            // subscript x_1 -> x1
      .replace(/_\{([^}]+)\}/g, "$1");     // subscript w_{ij} -> wij
    element.innerText = plain;
  }

  /* ----------------------------------------------------------------------
   * SVG Diagram: Neurona de 2 entradas.
   * -------------------------------------------------------------------- */
  function initNeuronDiagram(svg) {
    const D = DIAGRAM;
    svg.setAttribute("viewBox", `0 0 ${D.W} ${height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

    const rxAct = D.xAct - D.actW / 2;
    const ryAct = centerY - D.actH / 2;

    const defs =
      `<defs>
        <marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3"
                orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/>
        </marker>
        <clipPath id="actClip">
          <rect x="${rxAct + 8}" y="${ryAct + 8}" width="${D.actW - 16}" height="${D.actH - 16}" rx="6"/>
        </clipPath>
        <radialGradient id="actGrad" cx="38%" cy="32%" r="72%">
          <stop offset="0%"  stop-color="#bbf7d0"/>
          <stop offset="55%" stop-color="#4ade80"/>
          <stop offset="100%" stop-color="#15803d"/>
        </radialGradient>
      </defs>`;

    let svgInner = defs;

    // Conexiones de entrada
    for (let i = 0; i < 2; i++) {
      const x1 = D.xInputs + D.rInput;
      const y1 = inputY[i];
      const x2 = D.xSum - D.rSum;
      const y2 = centerY;
      const cx = (x1 + x2) / 2;
      svgInner +=
        `<path class="conn conn-input" data-conn="input-${i}"
               d="M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}"
               fill="none" stroke="#cbd5e1" stroke-width="2"
               marker-end="url(#ah)"/>`;
    }

    // Conexión Bias
    {
      const x1 = D.xSum;
      const y1 = biasY - D.rBias;
      const x2 = D.xSum;
      const y2 = centerY + D.rSum;
      svgInner +=
        `<path class="conn conn-bias" data-conn="bias"
               d="M${x1},${y1} L${x2},${y2}"
               fill="none" stroke="#cbd5e1" stroke-width="2"
               marker-end="url(#ah)"/>`;
    }

    // Conexión Sum -> Act
    svgInner +=
      `<path class="conn conn-sum" data-conn="sum"
             d="M${D.xSum + D.rSum},${centerY} L${D.xAct - D.actW / 2},${centerY}"
             fill="none" stroke="#cbd5e1" stroke-width="3"
             marker-end="url(#ah)"/>`;

    // Conexión Act -> Out
    svgInner +=
      `<path class="conn conn-act" data-conn="act"
             d="M${D.xAct + D.actW / 2},${centerY} L${D.xOut - D.rOut},${centerY}"
             fill="none" stroke="#cbd5e1" stroke-width="3"
             marker-end="url(#ah)"/>`;

    // Nodos Entrada
    for (let i = 0; i < 2; i++) {
      const cx = D.xInputs, cy = inputY[i];
      svgInner +=
        `<g class="node node-input" data-node="input-${i}">
           <circle cx="${cx}" cy="${cy}" r="${D.rInput}"
                   fill="#eff6ff" stroke="#3b82f6" stroke-width="3"/>
           <text x="${cx}" y="${cy + 5}" text-anchor="middle"
                 font-size="15" font-weight="700" fill="#3b82f6">x${i + 1}</text>
           <text class="node-value" x="${cx - D.rInput - 8}" y="${cy + 4}"
                 text-anchor="end" font-size="12" font-weight="600" fill="#1e3a8a">—</text>
         </g>`;
    }

    // Nodo Bias
    {
      const cx = D.xSum, cy = biasY;
      svgInner +=
        `<g class="node node-bias" data-node="bias">
           <circle cx="${cx}" cy="${cy}" r="${D.rBias}"
                   fill="#faf5ff" stroke="#a855f7" stroke-width="3"/>
           <text x="${cx}" y="${cy + 4}" text-anchor="middle"
                 font-size="12" font-weight="700" fill="#a855f7">b</text>
           <text class="node-value" x="${cx}" y="${cy + D.rBias + 16}"
                 text-anchor="middle" font-size="12" font-weight="600" fill="#6b21a8">—</text>
         </g>`;
    }

    // Nodo Sumatoria
    {
      const cx = D.xSum, cy = centerY;
      svgInner +=
        `<g class="node node-sum" data-node="sum">
           <circle cx="${cx}" cy="${cy}" r="${D.rSum}"
                   fill="#f8fafc" stroke="#374151" stroke-width="4"/>
           <text x="${cx}" y="${cy + 18}" text-anchor="middle"
                 font-size="46" font-weight="700" fill="#374151">Σ</text>
         </g>`;
    }

    // Etiqueta Sum
    {
      const mx = (D.xSum + D.rSum + D.xAct - D.actW / 2) / 2;
      const my = centerY;
      svgInner +=
        `<text class="sum-label" data-node="sum"
               x="${mx}" y="${my - 10}" text-anchor="middle"
               font-size="13" font-weight="700" fill="#111827">—</text>`;
    }

    // Nodo Activación
    {
      const cx = D.xAct, cy = centerY;
      const rx = cx - D.actW / 2, ry = cy - D.actH / 2;
      svgInner +=
        `<g class="node node-act" data-node="act">
           <rect x="${rx}" y="${ry}" width="${D.actW}" height="${D.actH}" rx="10"
                 fill="url(#actGrad)" stroke="#166534" stroke-width="3"/>
           <g id="act-mini-curve" clip-path="url(#actClip)"></g>
           <text class="node-value" x="${cx}" y="${cy + D.actH / 2 + 20}"
                 text-anchor="middle" font-size="13" font-weight="700" fill="#166534">—</text>
         </g>`;
    }

    // Nodo Salida
    {
      const cx = D.xOut, cy = centerY;
      svgInner +=
        `<g class="node node-out" data-node="out">
           <circle cx="${cx}" cy="${cy}" r="${D.rOut}"
                   fill="#fef2f2" stroke="#ef4444" stroke-width="3"/>
           <text x="${cx}" y="${cy + 5}" text-anchor="middle"
                 font-size="14" font-weight="700" fill="#ef4444">y</text>
           <text class="node-value" x="${cx}" y="${cy - D.rOut - 10}"
                 text-anchor="middle" font-size="13" font-weight="700" fill="#991b1b">—</text>
         </g>`;
    }

    svg.innerHTML = svgInner;
  }

  function renderWeightLabels(svg) {
    svg.querySelectorAll(".weight-label").forEach((e) => e.remove());
    const D = DIAGRAM;
    const cubic = (t, p0, p1, p2, p3) =>
      Math.pow(1 - t, 3) * p0 +
      3 * Math.pow(1 - t, 2) * t * p1 +
      3 * (1 - t) * t * t * p2 +
      t * t * t * p3;

    const t = 0.5;
    for (let i = 0; i < 2; i++) {
      const x1 = D.xInputs + D.rInput, y1 = inputY[i];
      const x2 = D.xSum - D.rSum, y2 = centerY;
      const cpx = (x1 + x2) / 2;

      const px = cubic(t, x1, cpx, cpx, x2);
      const py = cubic(t, y1, y1, y2, y2);

      const w = neuron.weights[i];
      const label =
        `<g class="weight-label">
           <rect x="${px - 27}" y="${py - 11}" width="54" height="18" rx="5"
                 fill="#fff7ed" stroke="#fdba74" stroke-width="1"/>
           <text x="${px}" y="${py + 2}" text-anchor="middle"
                 font-size="11" font-weight="700" fill="#c2410c">w${i + 1}=${w.toFixed(2)}</text>
         </g>`;
      svg.insertAdjacentHTML("beforeend", label);
    }
  }

  function renderActivationMiniCurve(svg) {
    const g = svg.querySelector("#act-mini-curve");
    if (!g) return;
    g.innerHTML = "";

    const meta = neuron.activationMeta;
    const range = meta.range;
    const D = DIAGRAM;
    const cx = D.xAct, cy = centerY;
    const innerW = D.actW - 16, innerH = D.actH - 16;
    const xToPx = (x) => cx - innerW / 2 + ((x - range.xMin) / (range.xMax - range.xMin)) * innerW;
    const yToPx = (y) => cy + innerH / 2 - ((y - range.yMin) / (range.yMax - range.yMin)) * innerH;

    let d = "";
    const steps = 80;
    for (let i = 0; i <= steps; i++) {
      const x = range.xMin + (i / steps) * (range.xMax - range.xMin);
      let y = meta.fn(x);
      if (!Number.isFinite(y)) continue;
      d += (d === "" ? "M" : " L") + xToPx(x).toFixed(1) + "," + yToPx(y).toFixed(1);
    }

    g.innerHTML =
      `<path d="${d}" fill="none" stroke="#ffffff" stroke-width="5" stroke-linecap="round" opacity="0.85"/>
       <path d="${d}" fill="none" stroke="#0f172a" stroke-width="2.2" stroke-linecap="round"/>`;

    const net = neuron.netInput, out = neuron.output;
    if (Number.isFinite(out)) {
      g.innerHTML +=
        `<circle cx="${xToPx(net).toFixed(1)}" cy="${yToPx(out).toFixed(1)}" r="3.5" fill="#ef4444" stroke="#fff" stroke-width="1"/>`;
    }
  }

  function animateFlow(svg) {
    const inputConns = svg.querySelectorAll(".conn-input");
    const biasConn = svg.querySelector(".conn-bias");
    const sumConn = svg.querySelector(".conn-sum");
    const actConn = svg.querySelector(".conn-act");
    const nodes = svg.querySelectorAll(".node");

    flowTimeouts.forEach(clearTimeout);
    flowTimeouts = [];

    svg.querySelectorAll(".lit").forEach((el) => el.classList.remove("lit"));

    const litMs = 420;
    const fire = (el, delay) => {
      if (!el) return;
      const id1 = setTimeout(() => {
        el.classList.add("lit");
        const id2 = setTimeout(() => el.classList.remove("lit"), litMs);
        flowTimeouts.push(id2);
      }, delay);
      flowTimeouts.push(id1);
    };

    nodes.forEach((node) => {
      if (node.classList.contains("node-input")) fire(node, 0);
    });
    inputConns.forEach((c) => fire(c, 30));
    fire(biasConn, 30);
    fire(svg.querySelector(".node-bias"), 0);

    fire(svg.querySelector(".node-sum"), 110);
    fire(sumConn, 170);
    fire(svg.querySelector(".node-act"), 220);
    fire(actConn, 260);
    fire(svg.querySelector(".node-out"), 300);
  }

  /* ----------------------------------------------------------------------
   * Renderizado del Plano Cartesiano (Canvas).
   * -------------------------------------------------------------------- */
  const canvas = document.getElementById("plane-canvas");
  const ctx = canvas.getContext("2d");
  let cssW = 0, cssH = 0;

  function resizeCanvas() {
    cssW = canvas.clientWidth || 400;
    cssH = canvas.clientHeight || 400;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function xToPx(x, plotW) {
    return padL + ((x - xMin) / (xMax - xMin)) * plotW;
  }

  function yToPx(y, plotH) {
    return padT + plotH - ((y - yMin) / (yMax - yMin)) * plotH;
  }

  function pxToX(px, plotW) {
    return xMin + ((px - padL) / plotW) * (xMax - xMin);
  }

  function pxToY(py, plotH) {
    return yMin + ((padT + plotH - py) / plotH) * (yMax - yMin);
  }

  function drawPlane() {
    if (cssW === 0 || cssH === 0) return;
    ctx.clearRect(0, 0, cssW, cssH);

    const plotW = cssW - padL - padR;
    const plotH = cssH - padT - padB;

    // 1. Mapa de calor (Zonas clasificadas en fondo)
    drawHeatmap(plotW, plotH);

    // 2. Grilla y Ejes Coordenados
    drawGridAndAxes(plotW, plotH);

    // 3. Recta de la Frontera de Decisión
    drawDecisionBoundary(plotW, plotH);

    // 4. Muestra de puntos de datos aleatorios + punto de entrenamiento activo
    drawDataset(plotW, plotH);

    // 5. Punto de Prueba (las entradas x1 y x2 representadas espacialmente)
    drawTestPoint(plotW, plotH);
  }

  function drawHeatmap(plotW, plotH) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT, plotW, plotH);
    ctx.clip();

    const gridCount = 55;
    const cellW = plotW / gridCount;
    const cellH = plotH / gridCount;

    const w1 = neuron.weights[0];
    const w2 = neuron.weights[1];
    const b = neuron.bias;

    for (let i = 0; i < gridCount; i++) {
      const px = padL + i * cellW;
      const x = xMin + (i + 0.5) * (xMax - xMin) / gridCount;
      for (let j = 0; j < gridCount; j++) {
        const py = padT + plotH - (j + 1) * cellH;
        const y = yMin + (j + 0.5) * (yMax - yMin) / gridCount;

        const net = x * w1 + y * w2 + b;
        // Sigmoide del net: 0.5 justo en la frontera, para cualquier activación.
        let val = 1 / (1 + Math.exp(-net));
        if (Number.isNaN(val) || !Number.isFinite(val)) val = 0.5;
        val = Math.max(0, Math.min(1, val));

        const rPart = Math.round(239 * (1 - val) + 59 * val);
        const gPart = Math.round(68 * (1 - val) + 130 * val);
        const bPart = Math.round(68 * (1 - val) + 246 * val);

        ctx.fillStyle = `rgba(${rPart}, ${gPart}, ${bPart}, 0.14)`;
        ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5);
      }
    }
    ctx.restore();
  }

  function drawGridAndAxes(plotW, plotH) {
    ctx.save();

    // Líneas de grilla
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    for (let val = -5; val <= 5; val++) {
      if (val === 0) continue;
      const px = xToPx(val, plotW);
      ctx.beginPath();
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();

      const py = yToPx(val, plotH);
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + plotW, py);
      ctx.stroke();
    }

    // Ejes cartesianos centrales
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1.5;

    const pxZero = xToPx(0, plotW);
    ctx.beginPath();
    ctx.moveTo(pxZero, padT);
    ctx.lineTo(pxZero, padT + plotH);
    ctx.stroke();

    const pyZero = yToPx(0, plotH);
    ctx.beginPath();
    ctx.moveTo(padL, pyZero);
    ctx.lineTo(padL + plotW, pyZero);
    ctx.stroke();

    // Borde exterior
    ctx.strokeStyle = "#e2e8f0";
    ctx.strokeRect(padL, padT, plotW, plotH);

    // Ticks y Etiquetas en Ejes
    ctx.fillStyle = "#64748b";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    [-4, -2, 2, 4].forEach((val) => {
      const px = xToPx(val, plotW);
      ctx.strokeStyle = "#94a3b8";
      ctx.beginPath();
      ctx.moveTo(px, pyZero - 3);
      ctx.lineTo(px, pyZero + 3);
      ctx.stroke();
      ctx.fillText(val.toString(), px, pyZero + 6);
    });

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    [-4, -2, 2, 4].forEach((val) => {
      const py = yToPx(val, plotH);
      ctx.strokeStyle = "#94a3b8";
      ctx.beginPath();
      ctx.moveTo(pxZero - 3, py);
      ctx.lineTo(pxZero + 3, py);
      ctx.stroke();
      ctx.fillText(val.toString(), pxZero - 6, py);
    });

    // Etiquetas indicativas de los ejes
    ctx.fillStyle = "#334155";
    ctx.font = "bold 11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("x₁", padL + plotW - 4, pyZero - 12);
    ctx.textAlign = "left";
    ctx.fillText("x₂", pxZero + 8, padT + 10);

    ctx.restore();
  }

  function drawDecisionBoundary(plotW, plotH) {
    const w1 = neuron.weights[0];
    const w2 = neuron.weights[1];
    const b = neuron.bias;

    if (Math.abs(w1) < 1e-6 && Math.abs(w2) < 1e-6) return;

    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT, plotW, plotH);
    ctx.clip();

    let p1, p2;
    if (Math.abs(w2) > 1e-6) {
      const getY = (x) => (-b - w1 * x) / w2;
      p1 = { x: -100, y: getY(-100) };
      p2 = { x: 100, y: getY(100) };
    } else {
      const xVal = -b / w1;
      p1 = { x: xVal, y: -100 };
      p2 = { x: xVal, y: 100 };
    }

    ctx.strokeStyle = "#10b981"; // Emerald green
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.moveTo(xToPx(p1.x, plotW), yToPx(p1.y, plotH));
    ctx.lineTo(xToPx(p2.x, plotW), yToPx(p2.y, plotH));
    ctx.stroke();

    ctx.restore();
  }

  function drawDataset(plotW, plotH) {
    ctx.save();
    dataset.forEach((p) => {
      const px = xToPx(p.x, plotW);
      const py = yToPx(p.y, plotH);

      // Si es el punto de entrenamiento activo, dibujar un círculo de realce parpadeante/halo
      if (highlightedPoint && highlightedPoint.x === p.x && highlightedPoint.y === p.y) {
        ctx.beginPath();
        const pulse = 11 + Math.sin(Date.now() / 120) * 2.5;
        ctx.arc(px, py, pulse, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(168, 85, 247, 0.85)"; // Morado
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.fillStyle = "rgba(168, 85, 247, 0.15)";
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 0 ? "#ef4444" : "#3b82f6";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawTestPoint(plotW, plotH) {
    const x1 = neuron.inputs[0];
    const x2 = neuron.inputs[1];

    const px = xToPx(x1, plotW);
    const py = yToPx(x2, plotH);

    ctx.save();

    // Círculo animado pulsante en el fondo
    const time = Date.now() / 250;
    const pulseRadius = 9 + Math.sin(time) * 2;
    ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, pulseRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Centro del punto de prueba
    ctx.beginPath();
    ctx.arc(px, py, 7.5, 0, Math.PI * 2);
    ctx.fillStyle = neuron.netInput >= 0 ? "#2563eb" : "#dc2626"; // mismo criterio que la frontera (net = 0)
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.5;
    ctx.fill();
    ctx.stroke();

    // Etiqueta flotante
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 10px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Prueba (x)", px, py - 12);

    ctx.restore();
  }

  /* ----------------------------------------------------------------------
   * Algoritmo y Regla de Aprendizaje del Perceptrón
   * -------------------------------------------------------------------- */

  function runLearningStep(index, transient = false) {
    if (dataset.length === 0) return;

    // 1. Obtener el punto del dataset
    const p = dataset[index];
    highlightedPoint = p;

    // 2. Establecer coordenadas en los inputs de la neurona de forma silenciosa
    neuron.batchUpdate(() => {
      neuron.setInput(0, p.x, true);
      neuron.setInput(1, p.y, true);
    });

    // 3. Obtener la salida e y el target d
    const y = neuron.output;
    const netBefore = neuron.netInput;
    const d = p.label;
    const e = d - y;

    // 4. Guardar valores anteriores de pesos y bias
    const w1Old = neuron.weights[0];
    const w2Old = neuron.weights[1];
    const bOld = neuron.bias;

    // 5. Aplicar la Regla del Perceptrón
    // Delta = eta * error * entrada
    const deltaW1 = learningRate * e * p.x;
    const deltaW2 = learningRate * e * p.y;
    const deltaB  = learningRate * e;

    const w1New = w1Old + deltaW1;
    const w2New = w2Old + deltaW2;
    const bNew  = bOld + deltaB;

    neuron.batchUpdate(() => {
      neuron.setWeight(0, w1New, transient);
      neuron.setWeight(1, w2New, transient);
      neuron.setBias(bNew, transient);
    });

    // 6. Actualizar paneles del DOM
    // El panel Predicción/Error muestra la evaluación ANTES del ajuste
    // (la que generó el error); render() ya pintó la salida posterior.
    document.getElementById("output-value").textContent = y.toFixed(4);
    document.getElementById("output-net").textContent = netBefore.toFixed(3);

    const targetValueEl = document.getElementById("target-value");
    if (targetValueEl) targetValueEl.textContent = d.toString();

    const outputErrorEl = document.getElementById("output-error");
    if (outputErrorEl) outputErrorEl.textContent = e.toFixed(4);

    // Fórmulas matemáticas
    const ruleWEl = document.getElementById("rule-weights");
    const ruleBEl = document.getElementById("rule-bias");
    const explanationEl = document.getElementById("learning-explanation");

    if (ruleWEl) {
      const ruleWText = 
        `\\begin{aligned}` +
        `\\Delta w_1 &= \\eta \\cdot e \\cdot x_1 \\\\` +
        `&= ${learningRate.toFixed(3)} \\cdot ${e.toFixed(3)} \\cdot ${p.x.toFixed(2)} = \\mathbf{${deltaW1.toFixed(4)}} \\\\` +
        `\\Delta w_2 &= \\eta \\cdot e \\cdot x_2 \\\\` +
        `&= ${learningRate.toFixed(3)} \\cdot ${e.toFixed(3)} \\cdot ${p.y.toFixed(2)} = \\mathbf{${deltaW2.toFixed(4)}} \\\\` +
        `w_1 &\\leftarrow ${w1Old.toFixed(3)} + (${deltaW1.toFixed(4)}) = \\mathbf{${w1New.toFixed(3)}} \\\\` +
        `w_2 &\\leftarrow ${w2Old.toFixed(3)} + (${deltaW2.toFixed(4)}) = \\mathbf{${w2New.toFixed(3)}}` +
        `\\end{aligned}`;
      renderLatex(ruleWText, ruleWEl);
    }

    if (ruleBEl) {
      const ruleBText = 
        `\\begin{aligned}` +
        `\\Delta b &= \\eta \\cdot e \\\\` +
        `&= ${learningRate.toFixed(3)} \\cdot ${e.toFixed(3)} = \\mathbf{${deltaB.toFixed(4)}} \\\\` +
        `b &\\leftarrow ${bOld.toFixed(3)} + (${deltaB.toFixed(4)}) = \\mathbf{${bNew.toFixed(3)}}` +
        `\\end{aligned}`;
      renderLatex(ruleBText, ruleBEl);
    }

    // Explicación didáctica
    if (explanationEl) {
      let explanationText = "";
      if (Math.abs(e) < 1e-4) {
        explanationText = `🎉 <strong>¡Predicción correcta!</strong> El error es cero. No se realizan cambios en los pesos ni en el bias para este ejemplo.`;
        explanationEl.style.borderColor = "#22c55e";
        explanationEl.style.background = "#f0fdf4";
      } else if (e > 0) {
        explanationText = `⚠️ <strong>Error positivo (d > y):</strong> La salida esperada era ${d} pero se obtuvo ${y.toFixed(3)}. Los pesos y el bias se incrementan para aumentar el valor del Net Input en futuras evaluaciones de esta zona azul.`;
        explanationEl.style.borderColor = "#3b82f6";
        explanationEl.style.background = "#eff6ff";
      } else {
        explanationText = `⚠️ <strong>Error negativo (d < y):</strong> La salida esperada era ${d} pero se obtuvo ${y.toFixed(3)}. Los pesos y el bias disminuyen para reducir el valor del Net Input en futuras evaluaciones de esta zona roja.`;
        explanationEl.style.borderColor = "#ef4444";
        explanationEl.style.background = "#fef2f2";
      }
      explanationEl.innerHTML = explanationText;
    }

    // Sincronizar sliders
    syncControlSliders();
  }

  function setSimButtonsState(disabled) {
    ["btn-learn-once", "btn-learn-epoch", "btn-learn-many", "btn-regenerate", "btn-reset", "dataset-select"]
      .forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.disabled = disabled;
      });
  }

  /**
   * Entrena época tras época (sin animar punto a punto) hasta clasificar
   * todo bien o agotar `maxEpochs`. Muestra el veredicto en la explicación.
   */
  function trainManyEpochs(maxEpochs) {
    if (epochTimer) clearInterval(epochTimer);
    setSimButtonsState(true);
    let done = 0;

    epochTimer = setInterval(() => {
      dataset.forEach((_, i) => runLearningStep(i, true));
      done++;
      const converged = finishEpoch();
      if (!converged && done < maxEpochs) return;

      clearInterval(epochTimer);
      epochTimer = null;
      setSimButtonsState(false);
      highlightedPoint = null;

      const explanationEl = document.getElementById("learning-explanation");
      if (explanationEl) {
        explanationEl.innerHTML = converged
          ? `✅ <strong>¡Convergió!</strong> Tras ${epochCount} épocas la recta separa todos los puntos.`
          : `❌ <strong>No converge.</strong> Tras ${done} épocas más sigue fallando puntos: la recta solo ` +
            `rota y se desplaza de un lado a otro. Ninguna recta puede separar estas clases.`;
        explanationEl.style.borderColor = converged ? "#22c55e" : "#ef4444";
        explanationEl.style.background = converged ? "#f0fdf4" : "#fef2f2";
      }
    }, 250);
  }

  function trainEpochAnimated() {
    // Si ya está corriendo un temporizador, lo paramos
    if (epochTimer) {
      clearInterval(epochTimer);
      epochTimer = null;
    }

    setSimButtonsState(true);
    let index = 0;
    const N = dataset.length;

    epochTimer = setInterval(() => {
      if (index >= N) {
        clearInterval(epochTimer);
        epochTimer = null;
        setSimButtonsState(false);
        highlightedPoint = null;
        finishEpoch();
        return;
      }
      runLearningStep(index, true); // transient = true durante el barrido
      index++;
    }, 100); // 100ms por punto resulta muy didáctico y fluido
  }

  /* ----------------------------------------------------------------------
   * Render principal (UI)
   * -------------------------------------------------------------------- */
  function render(svg, elements) {
    const w1 = neuron.weights[0];
    const w2 = neuron.weights[1];
    const x1 = neuron.inputs[0];
    const x2 = neuron.inputs[1];
    const b = neuron.bias;

    const eqText = `\\text{Net Input} = x_1 \\cdot w_1 + x_2 \\cdot w_2 + b`;
    const numText = `(${x1.toFixed(2)})(${w1.toFixed(3)}) + (${x2.toFixed(2)})(${w2.toFixed(3)}) + (${b.toFixed(3)}) = ${neuron.netInput.toFixed(3)}`;

    renderLatex(eqText, elements.equation);
    renderLatex(numText, elements.numeric);

    // Actualizar valores dentro de los nodos SVG
    svg.querySelector('[data-node="input-0"] .node-value').textContent = x1.toFixed(2);
    svg.querySelector('[data-node="input-1"] .node-value').textContent = x2.toFixed(2);
    svg.querySelector('[data-node="bias"] .node-value').textContent = b.toFixed(3);
    svg.querySelector(".sum-label").textContent = `Net = ${neuron.netInput.toFixed(3)}`;
    svg.querySelector('[data-node="act"] .node-value').textContent = `y = ${neuron.output.toFixed(3)}`;
    svg.querySelector('[data-node="out"] .node-value').textContent = neuron.output.toFixed(4);

    // Mini-curva en el nodo
    renderWeightLabels(svg);
    renderActivationMiniCurve(svg);

    // Panel de salida compacta
    if (elements.outputValue) elements.outputValue.textContent = neuron.output.toFixed(4);
    if (elements.outputActivation) elements.outputActivation.textContent = neuron.activationMeta.name;
    if (elements.outputNet) elements.outputNet.textContent = neuron.netInput.toFixed(3);

    // Resaltar la tarjeta de activación en el selector
    elements.activationList.querySelectorAll(".act-card").forEach((c) => {
      c.classList.toggle("active", c.dataset.key === neuron.activationFunction);
    });

    // Disparar animación visual de flujo de la neurona
    animateFlow(svg);

    // Aciertos con los pesos actuales (Bloque 4)
    updateStats();
  }

  function syncControlSliders() {
    inputControls[0].setValue(neuron.inputs[0]);
    inputControls[1].setValue(neuron.inputs[1]);
    weightControls[0].setValue(neuron.weights[0]);
    weightControls[1].setValue(neuron.weights[1]);
    biasControl.setValue(neuron.bias);
    if (lrControl) lrControl.setValue(learningRate);
  }

  /* ----------------------------------------------------------------------
   * Inicialización global del Bloque 3 al cargar el DOM.
   * -------------------------------------------------------------------- */
  function init() {
    // 1. Instanciar modelo
    neuron = new window.Neuron(
      INITIAL_INPUTS,
      INITIAL_WEIGHTS,
      INITIAL_BIAS,
      INITIAL_ACTIVATION
    );

    // 2. Referencias del DOM
    const elements = {
      inputsPanel:     document.getElementById("inputs-panel"),
      weightsPanel:    document.getElementById("weights-panel"),
      biasPanel:       document.getElementById("bias-panel"),
      lrPanel:         document.getElementById("lr-panel"),
      activationList:  document.getElementById("activation-list"),
      neuronSvg:       document.getElementById("neuron-svg"),
      equation:        document.getElementById("equation"),
      numeric:         document.getElementById("numeric"),
      outputValue:     document.getElementById("output-value"),
      outputActivation:document.getElementById("output-activation"),
      outputNet:       document.getElementById("output-net")
    };

    // 3. Inicializar diagrama SVG de la neurona
    initNeuronDiagram(elements.neuronSvg);

    // 4. Construir sliders de entradas
    inputControls.push(buildControlRow(elements.inputsPanel, {
      label: "x1", value: INITIAL_INPUTS[0], min: -6, max: 6, step: 0.01, decimals: 2,
      color: "#3b82f6", onChange: (v, transient) => {
        highlightedPoint = null;
        neuron.setInput(0, v, transient);
      }
    }));
    inputControls.push(buildControlRow(elements.inputsPanel, {
      label: "x2", value: INITIAL_INPUTS[1], min: -6, max: 6, step: 0.01, decimals: 2,
      color: "#3b82f6", onChange: (v, transient) => {
        highlightedPoint = null;
        neuron.setInput(1, v, transient);
      }
    }));

    // 5. Construir sliders de pesos
    weightControls.push(buildControlRow(elements.weightsPanel, {
      label: "w1", value: INITIAL_WEIGHTS[0], min: -5, max: 5, step: 0.001, decimals: 3,
      color: "#f97316", onChange: (v, transient) => {
        highlightedPoint = null;
        neuron.setWeight(0, v, transient);
      }
    }));
    weightControls.push(buildControlRow(elements.weightsPanel, {
      label: "w2", value: INITIAL_WEIGHTS[1], min: -5, max: 5, step: 0.001, decimals: 3,
      color: "#f97316", onChange: (v, transient) => {
        highlightedPoint = null;
        neuron.setWeight(1, v, transient);
      }
    }));

    // 6. Construir slider de bias
    biasControl = buildControlRow(elements.biasPanel, {
      label: "bias", value: INITIAL_BIAS, min: -5, max: 5, step: 0.001, decimals: 3,
      color: "#a855f7", onChange: (v, transient) => {
        highlightedPoint = null;
        neuron.setBias(v, transient);
      }
    });

    // 6b. Construir slider de tasa de aprendizaje (learning rate)
    lrControl = buildControlRow(elements.lrPanel, {
      label: "eta", value: learningRate, min: 0.001, max: 1.0, step: 0.001, decimals: 3,
      color: "#eab308", onChange: (v) => { learningRate = v; }
    });

    // 7. Construir selector de funciones de activación
    buildActivationSelector(elements.activationList, neuron, (key) => {
      neuron.setActivation(key);
    });

    // 8. Suscribir render a eventos del modelo
    neuron.subscribe(() => {
      render(elements.neuronSvg, elements);
    });

    // 9. Generar conjunto de puntos inicial
    generateDataset();

    // 10. Inicializar canvas del plano cartesiano
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // 10b. Interacciones de ratón/táctiles en el canvas para mover el punto de prueba (x)
    let isDragging = false;

    function handleCanvasInteraction(clientX, clientY, isTransient) {
      const rect = canvas.getBoundingClientRect();
      const mx = clientX - rect.left;
      const my = clientY - rect.top;

      const plotW = cssW - padL - padR;
      const plotH = cssH - padT - padB;

      let x1 = pxToX(mx, plotW);
      let x2 = pxToY(my, plotH);

      x1 = Math.max(xMin, Math.min(xMax, x1));
      x2 = Math.max(yMin, Math.min(yMax, x2));

      highlightedPoint = null;
      neuron.batchUpdate(() => {
        neuron.setInput(0, x1, isTransient);
        neuron.setInput(1, x2, isTransient);
      });
      syncControlSliders();
    }

    canvas.addEventListener("mousedown", (e) => {
      isDragging = true;
      handleCanvasInteraction(e.clientX, e.clientY, true);
    });

    window.addEventListener("mousemove", (e) => {
      if (isDragging) {
        handleCanvasInteraction(e.clientX, e.clientY, true);
      }
    });

    window.addEventListener("mouseup", (e) => {
      if (isDragging) {
        isDragging = false;
        handleCanvasInteraction(e.clientX, e.clientY, false);
      }
    });

    canvas.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        handleCanvasInteraction(e.touches[0].clientX, e.touches[0].clientY, true);
      }
    }, { passive: true });

    window.addEventListener("touchmove", (e) => {
      if (isDragging && e.touches.length === 1) {
        handleCanvasInteraction(e.touches[0].clientX, e.touches[0].clientY, true);
      }
    }, { passive: true });

    window.addEventListener("touchend", () => {
      if (isDragging) {
        isDragging = false;
        neuron.batchUpdate(() => {
          neuron.setInput(0, neuron.inputs[0], false);
          neuron.setInput(1, neuron.inputs[1], false);
        });
      }
    });

    // 11. Botones de la barra de herramientas
    const regenerate = () => {
      if (epochTimer) {
        clearInterval(epochTimer);
        epochTimer = null;
        setSimButtonsState(false);
      }
      highlightedPoint = null;
      generateDataset();
      updateStats();
    };
    document.getElementById("btn-regenerate").addEventListener("click", regenerate);
    const datasetSelect = document.getElementById("dataset-select");
    if (datasetSelect) datasetSelect.addEventListener("change", regenerate);

    document.getElementById("btn-reset").addEventListener("click", () => {
      if (epochTimer) {
        clearInterval(epochTimer);
        epochTimer = null;
        setSimButtonsState(false);
      }
      highlightedPoint = null;
      epochCount = 0;
      epochLog = [];
      neuron.batchUpdate(() => {
        neuron.setWeight(0, INITIAL_WEIGHTS[0]);
        neuron.setWeight(1, INITIAL_WEIGHTS[1]);
        neuron.setBias(INITIAL_BIAS);
        neuron.setActivation(INITIAL_ACTIVATION);
      });
      syncControlSliders();
    });

    document.getElementById("btn-learn-once").addEventListener("click", () => {
      if (epochTimer) {
        clearInterval(epochTimer);
        epochTimer = null;
        setSimButtonsState(false);
      }
      // Entrena sobre el elemento apuntado por currentTrainingIndex
      runLearningStep(currentTrainingIndex, false);
      // Avanzar el índice de forma cíclica
      currentTrainingIndex = (currentTrainingIndex + 1) % dataset.length;
      if (currentTrainingIndex === 0) finishEpoch();
    });

    document.getElementById("btn-learn-epoch").addEventListener("click", () => {
      trainEpochAnimated();
    });

    const btnMany = document.getElementById("btn-learn-many");
    if (btnMany) btnMany.addEventListener("click", () => trainManyEpochs(20));

    // 12. Pintado inicial completo
    render(elements.neuronSvg, elements);
    syncControlSliders();

    // 13. Loop de renderizado continuo para animaciones fluidas del plano cartesiano
    function animLoop() {
      drawPlane();
      requestAnimationFrame(animLoop);
    }
    requestAnimationFrame(animLoop);
  }

  // Arranque seguro cuando el DOM esté listo
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
