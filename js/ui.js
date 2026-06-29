/* ===========================================================================
 * ui.js
 * ---------------------------------------------------------------------------
 * Capa de presentación (Vista). No calcula nada: toma un objeto `Neuron` y
 * construye/refresca la interfaz.
 *
 * Responsabilidades:
 *   1. Construir dinámicamente los paneles de Entradas, Pesos y Bias a
 *      partir del estado de la neurona (escalable: nº de entradas = longitud
 *      de neuron.inputs, sin valores fijos).
 *   2. Dibujar el diagrama de NEURONA (SVG): nodos circulares separados
 *      conectados por líneas individuales (una por entrada), nodo Σ grande,
 *      nodo de bias, nodo de activación (con mini-curva) y nodo de salida.
 *   3. Animar el "flujo de información" iluminando conexiones y nodos.
 *   4. Dibujar en <canvas> la función de activación con un punto rojo y una
 *      línea vertical en el net input actual.
 *   5. Renderizar la ecuación simbólica, la sustitución numérica y la salida.
 *
 * El controlador (app.js) conecta el Neuron con esta vista mediante la API
 * pública de la clase `NeuronUI`.
 * =========================================================================== */

(function () {
  "use strict";

  /* ----------------------------------------------------------------------
   * Utilidades de formato numérico.
   * -------------------------------------------------------------------- */

  /**
   * Formatea un número de forma compacta y legible.
   * @param {number} n
   * @param {number} decimals Nº de decimales (por defecto 3).
   * @returns {string}
   */
  function fmt(n, decimals = 3) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Number(n.toFixed(decimals)).toString();
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
      .replace(/\\cdot/g, "·")             // \cdot -> ·
      .replace(/\\Delta/g, "Δ")            // \Delta -> Δ
      .replace(/\\eta/g, "η")              // \eta -> η
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
   * Geometría del diagrama de la neurona.
   * Las posiciones se calculan a partir del nº de entradas para que no haya
   * nada "hardcoded". Si hay 20 entradas, el diagrama crece en altura.
   * -------------------------------------------------------------------- */

  const DIAGRAM = {
    W: 900,                 // ancho lógico del viewBox
    xInputs: 110,           // columna de entradas
    xSum: 430,              // centro: nodo Σ (y del bias, debajo)
    xAct: 660,              // nodo de activación (rectángulo)
    xOut: 840,              // nodo de salida
    rInput: 26,             // radio entradas
    rBias: 26,              // radio bias
    rSum: 54,               // radio sumatoria (más grande)
    rOut: 30,               // radio salida
    actW: 120,              // ancho del rectángulo de activación
    actH: 90,               // alto del rectángulo de activación
    biasGap: 70,            // separación vertical entre Σ y el bias
    padTop: 60,             // margen superior para etiquetas
    padBottom: 30,          // margen inferior
    step: 70                // separación vertical mínima entre entradas
  };

  /* ----------------------------------------------------------------------
   * Clase principal: NeuronUI
   * -------------------------------------------------------------------- */

  class NeuronUI {
    /**
     * @param {Neuron} neuron
     * @param {Object} elements Referencias a contenedores del DOM.
     */
    constructor(neuron, elements) {
      this.neuron = neuron;
      this.el = elements;
      this._flowTimeouts = [];
      this._inputRows = [];
      this._weightRows = [];
      this._biasRow = null;
      this._lastNodeCount = -1;   // para detectar cambios en nº de entradas

      // Colores por etapa (coherentes con styles.css y el enunciado).
      this.colors = {
        input: "#3b82f6",     // azul
        weight: "#f97316",    // naranja
        bias: "#a855f7",      // morado
        sum: "#374151",       // gris oscuro
        activation: "#22c55e",// verde
        output: "#ef4444"     // rojo
      };

      this._buildAll();

      // Suscripción al modelo: cualquier cambio dispara repintado.
      this.neuron.subscribe(() => this.render());
    }

    /* ====================================================================
     * Construcción inicial de la vista.
     * =================================================================== */

    _buildAll() {
      this._buildActivationSelector();
      this._buildInputsPanel();
      this._buildWeightsPanel();
      this._buildBiasPanel();
      this._initCanvas();
      this._buildButtons();
    }

    /* --------------------------------------------------------------------
     * Selector de funciones de activación.
     * Se genera a partir del diccionario global → añadir una función nueva
     * la muestra automáticamente en el selector.
     * ------------------------------------------------------------------ */
    _buildActivationSelector() {
      const wrap = this.el.activationList;
      wrap.innerHTML = "";
      const funcs = window.ActivationFunctions;

      Object.keys(funcs).forEach((key) => {
        const meta = funcs[key];
        const card = document.createElement("button");
        card.type = "button";
        card.className = "act-card";
        card.dataset.key = key;
        if (key === this.neuron.activationFunction) card.classList.add("active");

        card.innerHTML =
          `<span class="act-name">${meta.name}</span>` +
          `<code class="act-eq">${meta.equation}</code>` +
          `<span class="act-desc">${meta.description}</span>`;

        card.addEventListener("click", () => {
          if (typeof this.onActivationChange === "function") {
            this.onActivationChange(key);
          }
        });
        wrap.appendChild(card);
      });
    }

    /* --------------------------------------------------------------------
     * Fila de control genérica (slider + número) para entradas/pesos/bias.
     * ------------------------------------------------------------------ */
    _buildControlRow(cfg) {
      const row = document.createElement("div");
      row.className = "ctrl-row";
      row.style.setProperty("--accent", cfg.color);

      const header = document.createElement("div");
      header.className = "ctrl-head";
      header.innerHTML =
        `<span class="ctrl-label">${cfg.label}</span>` +
        `<input class="ctrl-num" type="number" ` +
        `min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" ` +
        `value="${fmt(cfg.value, cfg.decimals)}">`;

      const slider = document.createElement("input");
      slider.type = "range";
      slider.className = "ctrl-slider";
      slider.min = cfg.min;
      slider.max = cfg.max;
      slider.step = cfg.step;
      slider.value = cfg.value;

      row.appendChild(header);
      row.appendChild(slider);

      const numInput = header.querySelector(".ctrl-num");

      /** Sincroniza slider <-> número y notifica el cambio. */
      const apply = (v, transient = false) => {
        const clamped = Math.min(cfg.max, Math.max(cfg.min, v));
        slider.value = clamped;
        numInput.value = fmt(clamped, cfg.decimals);
        this._updateSliderFill(slider, cfg.min, cfg.max);
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

      this._updateSliderFill(slider, cfg.min, cfg.max);

      // Permite actualizar el control desde fuera (presets) sin disparar callback.
      row._setValue = (v) => {
        slider.value = v;
        numInput.value = fmt(v, cfg.decimals);
        this._updateSliderFill(slider, cfg.min, cfg.max);
      };

      return row;
    }

    /**
     * Pinta el "relleno" del slider mediante una variable CSS --fill,
     * para que el rango avanzado tome el color de acento de la etapa.
     */
    _updateSliderFill(slider, min, max) {
      const pct = ((parseFloat(slider.value) - min) / (max - min)) * 100;
      slider.style.setProperty("--fill", `${Math.max(0, Math.min(100, pct))}%`);
    }

    /* --------------------------------------------------------------------
     * Paneles de Entradas / Pesos / Bias — dinámicos.
     * ------------------------------------------------------------------ */
    _buildInputsPanel() {
      const panel = this.el.inputsPanel;
      panel.innerHTML = "";
      this._inputRows = [];
      this.neuron.inputs.forEach((value, i) => {
        const row = this._buildControlRow({
          label: `x${i + 1}`,
          value, min: -10, max: 10, step: 0.01, decimals: 2,
          color: this.colors.input,
          onChange: (v) => this.neuron.setInput(i, v)
        });
        panel.appendChild(row);
        this._inputRows.push(row);
      });
    }

    _buildWeightsPanel() {
      const panel = this.el.weightsPanel;
      panel.innerHTML = "";
      this._weightRows = [];
      this.neuron.weights.forEach((value, i) => {
        const row = this._buildControlRow({
          label: `w${i + 1}`,
          value, min: -5, max: 5, step: 0.001, decimals: 3,
          color: this.colors.weight,
          onChange: (v) => this.neuron.setWeight(i, v)
        });
        panel.appendChild(row);
        this._weightRows.push(row);
      });
    }

    _buildBiasPanel() {
      const panel = this.el.biasPanel;
      panel.innerHTML = "";
      this._biasRow = this._buildControlRow({
        label: "bias",
        value: this.neuron.bias,
        min: -5, max: 5, step: 0.001, decimals: 3,
        color: this.colors.bias,
        onChange: (v) => this.neuron.setBias(v)
      });
      panel.appendChild(this._biasRow);
    }

    /* ====================================================================
     * Diagrama de la NEURONA (SVG con nodos circulares).
     *
     * Se reconstruye cuando cambia el nº de entradas. Las posiciones de los
     * nodos se calculan con _layout(). El contenido de texto y la mini-curva
     * de activación se refrescan en render().
     * =================================================================== */

    /**
     * Calcula la geometría: alto del SVG y posición 'y' de cada nodo de
     * entrada, además del bias.
     *
     * El bias se coloca JUSTO DEBAJO de la sumatoria (misma columna X que Σ),
     * con una separación `biasGap` suficiente para que el valor del bias sea
     * legible entre ambos nodos.
     * @returns {Object} { height, inputY:number[], biasY, centerY }
     */
    _layout() {
      const n = this.neuron.inputs.length;
      const contentH = Math.max(n - 1, 1) * DIAGRAM.step;
      // El alto debe acoger además el bias bajo Σ + su etiqueta.
      const height = Math.max(
        DIAGRAM.padTop + contentH + DIAGRAM.biasGap + DIAGRAM.rBias * 2 + 24 + DIAGRAM.padBottom,
        360
      );
      const centerY = height / 2;

      // Distribuir las entradas verticalmente, centradas alrededor de centerY.
      const inputY = [];
      const start = centerY - contentH / 2;
      for (let i = 0; i < n; i++) inputY.push(start + i * DIAGRAM.step);

      // El bias va DEBAJO de Σ: misma columna X, separado por biasGap.
      const biasY = centerY + DIAGRAM.rSum + DIAGRAM.biasGap;

      return { height, inputY, biasY, centerY };
    }

    /**
     * Construye (o reconstruye) el SVG de la neurona.
     * @private
     */
    buildNeuronDiagram() {
      const svg = this.el.neuronSvg;
      const { height, inputY, biasY, centerY } = this._layout();
      const n = this.neuron.inputs.length;
      this._lastNodeCount = n;

      const D = DIAGRAM;
      const rxAct = D.xAct - D.actW / 2;
      const ryAct = centerY - D.actH / 2;

      svg.setAttribute("viewBox", `0 0 ${D.W} ${height}`);
      svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

      // --- Definiciones: marcador de flecha + degradado de activación ----
      const defs =
        `<defs>
          <marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3"
                  orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/>
          </marker>
          <clipPath id="actClip">
            <rect x="${rxAct + 8}" y="${ryAct + 8}" width="${D.actW - 16}" height="${D.actH - 16}" rx="6"/>
          </clipPath>
          <!-- Degradado radial que da "volumen" al nodo de activación,
               simulando una esfera con un destello superior. -->
          <radialGradient id="actGrad" cx="38%" cy="32%" r="72%">
            <stop offset="0%"  stop-color="#bbf7d0"/>
            <stop offset="55%" stop-color="#4ade80"/>
            <stop offset="100%" stop-color="#15803d"/>
          </radialGradient>
        </defs>`;

      let svgInner = defs;

      /* --- 1) Conexiones de cada ENTRADA → Σ (individuales) -------------
         Cada entrada tiene su propia línea. Llevan etiqueta de peso. */
      for (let i = 0; i < n; i++) {
        const x1 = D.xInputs + D.rInput;
        const y1 = inputY[i];
        const x2 = D.xSum - D.rSum;
        const y2 = centerY;
        // Curva bezier suave hacia el centro.
        const cx = (x1 + x2) / 2;
        svgInner +=
          `<path class="conn conn-input" data-conn="input-${i}"
                 d="M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}"
                 fill="none" stroke="#cbd5e1" stroke-width="2"
                 marker-end="url(#ah)"/>`;
      }

      /* --- 2) Conexión BIAS → Σ -----------------------------------------
         El bias está DEBAJO de Σ (misma columna X), por lo que la conexión
         es VERTICAL hacia arriba. */
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

      /* --- 3) Conexión Σ → Activación (rectángulo) -------------------- */
      svgInner +=
        `<path class="conn conn-sum" data-conn="sum"
               d="M${D.xSum + D.rSum},${centerY} L${D.xAct - D.actW / 2},${centerY}"
               fill="none" stroke="#cbd5e1" stroke-width="3"
               marker-end="url(#ah)"/>`;

      /* --- 4) Conexión Activación → Salida ---------------------------- */
      svgInner +=
        `<path class="conn conn-act" data-conn="act"
               d="M${D.xAct + D.actW / 2},${centerY} L${D.xOut - D.rOut},${centerY}"
               fill="none" stroke="#cbd5e1" stroke-width="3"
               marker-end="url(#ah)"/>`;

      /* --- 5) Nodos de ENTRADA (uno por cada x_i) ----------------------
         El valor de cada entrada se muestra a la IZQUIERDA del círculo. */
      for (let i = 0; i < n; i++) {
        const cx = D.xInputs, cy = inputY[i];
        svgInner +=
          `<g class="node node-input" data-node="input-${i}">
             <circle cx="${cx}" cy="${cy}" r="${D.rInput}"
                     fill="#eff6ff" stroke="${this.colors.input}" stroke-width="3"/>
             <text x="${cx}" y="${cy + 5}" text-anchor="middle"
                   font-size="15" font-weight="700" fill="${this.colors.input}">x${i + 1}</text>
             <text class="node-value" x="${cx - D.rInput - 8}" y="${cy + 4}"
                   text-anchor="end" font-size="12" font-weight="600" fill="#1e3a8a">—</text>
           </g>`;
      }

      /* --- 6) Nodo BIAS (debajo de Σ, misma columna X) ------------------
         El valor del bias se muestra DEBAJO del círculo morado. */
      {
        const cx = D.xSum, cy = biasY;
        svgInner +=
          `<g class="node node-bias" data-node="bias">
             <circle cx="${cx}" cy="${cy}" r="${D.rBias}"
                     fill="#faf5ff" stroke="${this.colors.bias}" stroke-width="3"/>
             <text x="${cx}" y="${cy + 4}" text-anchor="middle"
                   font-size="12" font-weight="700" fill="${this.colors.bias}">b</text>
             <text class="node-value" x="${cx}" y="${cy + D.rBias + 16}"
                   text-anchor="middle" font-size="12" font-weight="600" fill="#6b21a8">—</text>
           </g>`;
      }

      /* --- 7) Nodo SUMATORIA (círculo grande con Σ) --------------------
         El valor del net input se muestra SOBRE la conexión Σ→activación,
         no dentro del nodo (ver etiqueta conn-sum-label más abajo). */
      {
        const cx = D.xSum, cy = centerY;
        svgInner +=
          `<g class="node node-sum" data-node="sum">
             <circle cx="${cx}" cy="${cy}" r="${D.rSum}"
                     fill="#f8fafc" stroke="${this.colors.sum}" stroke-width="4"/>
             <text x="${cx}" y="${cy + 18}" text-anchor="middle"
                   font-size="46" font-weight="700" fill="${this.colors.sum}">Σ</text>
           </g>`;
      }

      /* --- 7b) Etiqueta del net input SOBRE la conexión Σ→activación --- */
      {
        // Punto medio de la línea Σ → rectángulo de activación.
        const mx = (D.xSum + D.rSum + D.xAct - D.actW / 2) / 2;
        const my = centerY;
        svgInner +=
          `<text class="sum-label" data-node="sum"
                 x="${mx}" y="${my - 10}" text-anchor="middle"
                 font-size="13" font-weight="700" fill="#111827">—</text>`;
      }

      /* --- 8) Nodo ACTIVACIÓN (RECTÁNGULO con la curva dibujada dentro) -
         Cuadrado/rectángulo verde con la gráfica de la función de
         activación en su interior. La curva se dibuja en render(). */
      {
        const cx = D.xAct, cy = centerY;
        const rx = cx - D.actW / 2, ry = cy - D.actH / 2;
        // clipPath rectangular para que la curva no salga del nodo.
        svgInner +=
          `<g class="node node-act" data-node="act">
             <rect x="${rx}" y="${ry}" width="${D.actW}" height="${D.actH}" rx="10"
                   fill="url(#actGrad)" stroke="#166534" stroke-width="3"/>
             <g id="act-mini-curve" clip-path="url(#actClip)"></g>
             <text class="node-value" x="${cx}" y="${cy + D.actH / 2 + 20}"
                   text-anchor="middle" font-size="13" font-weight="700" fill="#166534">—</text>
           </g>`;
      }

      /* --- 9) Nodo SALIDA ---------------------------------------------- */
      {
        const cx = D.xOut, cy = centerY;
        svgInner +=
          `<g class="node node-out" data-node="out">
             <circle cx="${cx}" cy="${cy}" r="${D.rOut}"
                     fill="#fef2f2" stroke="${this.colors.output}" stroke-width="3"/>
             <text x="${cx}" y="${cy + 5}" text-anchor="middle"
                   font-size="14" font-weight="700" fill="${this.colors.output}">y</text>
             <text class="node-value" x="${cx}" y="${cy - D.rOut - 10}"
                   text-anchor="middle" font-size="13" font-weight="700" fill="#991b1b">—</text>
           </g>`;
      }

      svg.innerHTML = svgInner;

      // Guardar geometría para etiquetas y actualizaciones.
      this._geo = { height, inputY, biasY, centerY };
    }

    /**
     * Dibuja/actualiza las etiquetas de peso sobre cada conexión entrada→Σ.
     *
     * Cada etiqueta se coloca sobre SU propia curva de conexión, en el punto
     * medio (t=0.5) de la misma. Como las entradas parten de alturas
     * distintas, las etiquetas quedan naturalmente separadas y legibles.
     *
     * La posición se calcula con la fórmula correcta de una Bézier CÚBICA
     * (la conexión usa dos puntos de control), no una aproximación.
     * @private
     */
    _renderWeightLabels() {
      const svg = this.el.neuronSvg;
      // Eliminar etiquetas previas.
      svg.querySelectorAll(".weight-label").forEach((e) => e.remove());

      const D = DIAGRAM;
      const { inputY, centerY } = this._geo;
      const n = this.neuron.inputs.length;

      /**
       * Evalúa una Bézier cúbica en el parámetro t.
       * @param {number} t  Parámetro en [0,1].
       * @param {number} p0 Punto inicial.
       * @param {number} p1 Primer punto de control.
       * @param {number} p2 Segundo punto de control.
       * @param {number} p3 Punto final.
       * @returns {number}
       */
      const cubic = (t, p0, p1, p2, p3) =>
        Math.pow(1 - t, 3) * p0 +
        3 * Math.pow(1 - t, 2) * t * p1 +
        3 * (1 - t) * t * t * p2 +
        t * t * t * p3;

      // 't' a lo largo de cada curva donde se sitúa la etiqueta.
      // 0.5 = punto medio: buen equilibrio entre legibilidad y no tapar nodos.
      const t = 0.5;

      for (let i = 0; i < n; i++) {
        const x1 = D.xInputs + D.rInput, y1 = inputY[i];
        const x2 = D.xSum - D.rSum, y2 = centerY;
        // Puntos de control de la curva cúbica (ver buildNeuronDiagram):
        //   C cx,y1  cx,y2  x2,y2
        const cpx = (x1 + x2) / 2;

        const px = cubic(t, x1, cpx, cpx, x2);
        const py = cubic(t, y1, y1, y2, y2);

        const w = this.neuron.weights[i];
        const label =
          `<g class="weight-label">
             <rect x="${px - 27}" y="${py - 11}" width="54" height="18" rx="5"
                   fill="#fff7ed" stroke="#fdba74" stroke-width="1"/>
             <text x="${px}" y="${py + 2}" text-anchor="middle"
                   font-size="11" font-weight="700" fill="#c2410c">w${i + 1}=${fmt(w, 2)}</text>
           </g>`;
        svg.insertAdjacentHTML("beforeend", label);
      }
    }

    /**
     * Dibuja la mini-curva de la función de activación dentro del nodo de
     * activación. Se regenera en cada render porque depende de la función.
     * @private
     */
    _renderActivationMiniCurve() {
      const g = this.el.neuronSvg.querySelector("#act-mini-curve");
      if (!g) return;
      g.innerHTML = "";

      const meta = this.neuron.activationMeta;
      const range = meta.range;
      const D = DIAGRAM;
      const cx = D.xAct, cy = this._geo.centerY;
      // Área útil = interior del RECTÁNGULO de activación (con margen).
      const innerW = D.actW - 16, innerH = D.actH - 16;
      const xToPx = (x) => cx - innerW / 2 + ((x - range.xMin) / (range.xMax - range.xMin)) * innerW;
      const yToPx = (y) => cy + innerH / 2 - ((y - range.yMin) / (range.yMax - range.yMin)) * innerH;

      let d = "";
      const steps = 80;
      for (let i = 0; i <= steps; i++) {
        const x = range.xMin + (i / steps) * (range.xMax - range.xMin);
        let y;
        try { y = meta.fn(x); } catch (e) { y = NaN; }
        if (!Number.isFinite(y)) continue;
        d += (d === "" ? "M" : " L") + xToPx(x).toFixed(1) + "," + yToPx(y).toFixed(1);
      }
      // Trazo blanco semitransparente (sombra) + trazo azul oscuro encima,
      // para que la curva destaque sobre el degradado verde del nodo.
      g.innerHTML =
        `<path d="${d}" fill="none" stroke="#ffffff"
                stroke-width="5" stroke-linecap="round" opacity="0.85"/>
         <path d="${d}" fill="none" stroke="#0f172a"
                stroke-width="2.2" stroke-linecap="round"/>`;

      // Punto rojo del valor actual.
      const net = this.neuron.netInput, out = this.neuron.output;
      if (Number.isFinite(out)) {
        g.innerHTML +=
          `<circle cx="${xToPx(net).toFixed(1)}" cy="${yToPx(out).toFixed(1)}"
                   r="3.5" fill="#ef4444" stroke="#fff" stroke-width="1"/>`;
      }
    }

    /* ====================================================================
     * Canvas: gráfica grande de la función de activación (panel inferior).
     * =================================================================== */

    _initCanvas() {
      this.ctx = this.el.graphCanvas.getContext("2d");
      this._resizeCanvas();
      window.addEventListener("resize", () => this._resizeCanvas());
    }

    /** Ajusta el tamaño físico del canvas al mostrado (anti-borroso). */
    _resizeCanvas() {
      const cssW = this.el.graphCanvas.clientWidth || 360;
      const cssH = this.el.graphCanvas.clientHeight || 260;
      const dpr = window.devicePixelRatio || 1;
      this.el.graphCanvas.width = cssW * dpr;
      this.el.graphCanvas.height = cssH * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this._cssW = cssW;
      this._cssH = cssH;
      if (this.neuron) this._drawGraph();
    }

    /** Dibuja ejes, curva, punto rojo y línea vertical en el net input. */
    _drawGraph() {
      const ctx = this.ctx;
      const W = this._cssW, H = this._cssH;
      const meta = this.neuron.activationMeta;
      const range = meta.range;
      const padL = 44, padR = 16, padT = 16, padB = 28;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const xToPx = (x) => padL + ((x - range.xMin) / (range.xMax - range.xMin)) * plotW;
      const yToPx = (y) => padT + plotH - ((y - range.yMin) / (range.yMax - range.yMin)) * plotH;

      ctx.clearRect(0, 0, W, H);

      // Ejes y rejilla.
      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      if (range.yMin <= 0 && range.yMax >= 0) {
        ctx.beginPath(); ctx.moveTo(padL, yToPx(0)); ctx.lineTo(W - padR, yToPx(0)); ctx.stroke();
      }
      if (range.xMin <= 0 && range.xMax >= 0) {
        ctx.beginPath(); ctx.moveTo(xToPx(0), padT); ctx.lineTo(xToPx(0), H - padB); ctx.stroke();
      }
      ctx.strokeStyle = "#9ca3af";
      ctx.strokeRect(padL, padT, plotW, plotH);

      // Etiquetas.
      ctx.fillStyle = "#6b7280";
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(`${meta.name} — ${meta.equation}`, padL + 4, padT + 12);
      ctx.fillText(range.xMin, padL, H - 8);
      ctx.textAlign = "right";
      ctx.fillText(range.xMax, W - padR, H - 8);
      ctx.textAlign = "left";
      ctx.fillText(fmt(range.yMin, 1), 6, H - padB + 4);
      ctx.fillText(fmt(range.yMax, 1), 6, padT + 8);

      // Curva.
      ctx.strokeStyle = this.colors.activation;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      const steps = 240;
      let started = false;
      for (let i = 0; i <= steps; i++) {
        const x = range.xMin + (i / steps) * (range.xMax - range.xMin);
        let y;
        try { y = meta.fn(x); } catch (e) { y = NaN; }
        if (!Number.isFinite(y)) { started = false; continue; }
        if (y < range.yMin - 2 || y > range.yMax + 2) { started = false; continue; }
        const px = xToPx(x), py = yToPx(y);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Línea vertical + punto rojo.
      const net = this.neuron.netInput, out = this.neuron.output;
      const clampedPx = Math.max(padL, Math.min(W - padR, xToPx(net)));
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(clampedPx, padT); ctx.lineTo(clampedPx, H - padB); ctx.stroke();
      ctx.setLineDash([]);

      if (Number.isFinite(out)) {
        const pyOut = yToPx(Math.max(range.yMin, Math.min(range.yMax, out)));
        ctx.fillStyle = "#ef4444";
        ctx.beginPath(); ctx.arc(clampedPx, pyOut, 6, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.stroke();

        ctx.fillStyle = "#111827";
        ctx.font = "bold 12px system-ui, sans-serif";
        ctx.textAlign = clampedPx > W / 2 ? "right" : "left";
        ctx.fillText(`f(${fmt(net)}) = ${fmt(out, 4)}`,
          clampedPx + (clampedPx > W / 2 ? -8 : 8),
          Math.max(padT + 12, pyOut - 10));
      }
    }

    /* --------------------------------------------------------------------
     * Botones de acción (callbacks definidos por el controlador).
     * ------------------------------------------------------------------ */
    _buildButtons() {
      const map = {
        "btn-randomize": "onRandomize",
        "btn-reset": "onReset",
        "btn-example1": "onExample1",
        "btn-example2": "onExample2"
      };
      Object.keys(map).forEach((id) => {
        const btn = document.getElementById(id);
        if (!btn) return;
        // Registramos SIEMPRE el listener (el callback se asigna más tarde
        // desde app.js, por eso se resuelve dinámicamente en el momento del
        // clic, no en el de registro).
        btn.addEventListener("click", () => {
          const cb = this[map[id]];
          if (typeof cb === "function") cb();
        });
      });
    }

    /* ====================================================================
     * Render principal: se ejecuta en cada cambio del modelo.
     * =================================================================== */
    render() {
      const n = this.neuron;

      // Reconstruir el diagrama si cambió el nº de entradas (escalabilidad).
      if (n.inputs.length !== this._lastNodeCount) {
        this.buildNeuronDiagram();
      }

      // Ecuación y sustitución numérica.
      renderLatex(this._renderEquation(), this.el.equation);
      renderLatex(this._renderNumeric(), this.el.numeric);

      // Valores dentro de los nodos.
      this._updateNodes();

      // Etiquetas de peso y mini-curva de activación.
      if (this._geo) {
        this._renderWeightLabels();
        this._renderActivationMiniCurve();
      }

      // Panel de salida.
      this.el.outputValue.textContent = fmt(n.output, 4);
      this.el.outputActivation.textContent = n.activationMeta.name;
      this.el.outputNet.textContent = fmt(n.netInput);

      // Selector: resaltar la activa.
      this.el.activationList.querySelectorAll(".act-card").forEach((c) => {
        c.classList.toggle("active", c.dataset.key === n.activationFunction);
      });

      // Gráfica inferior.
      this._drawGraph();

      // Animación de flujo.
      this._animateFlow();
    }

    /** Construye la expresión LaTeX de la ecuación simbólica. */
    _renderEquation() {
      const n = this.neuron;
      const terms = n.inputs.map((_, i) => `x_{${i + 1}} \\cdot w_{${i + 1}}`);
      return `\\text{Net Input} = ${terms.join(" + ")} + b`;
    }

    /** Construye la expresión LaTeX de la sustitución numérica con el resultado. */
    _renderNumeric() {
      const n = this.neuron;
      const terms = n.inputs.map((x, i) => `(${fmt(x, 2)})(${fmt(n.weights[i], 3)})`);
      const expr = terms.join(" + ") + ` + (${fmt(n.bias, 3)})`;
      return `${expr} = ${fmt(n.netInput)}`;
    }

    /**
     * Actualiza los textos de valor dentro de cada nodo del diagrama.
     *
     * Cada nodo identifica su texto editable con data-node. La mayoría usan
     * un elemento `.node-value`, salvo la sumatoria, cuyo net input se
     * muestra en una etiqueta `.sum-label` independiente (sobre la conexión
     * Σ→activación).
     * @private
     */
    _updateNodes() {
      const n = this.neuron;
      const setText = (selector, text) => {
        const t = this.el.neuronSvg.querySelector(selector);
        if (t) t.textContent = text;
      };
      // Valores de las entradas (uno por nodo de entrada).
      n.inputs.forEach((x, i) => {
        const g = this.el.neuronSvg.querySelector(`[data-node="input-${i}"]`);
        if (g) {
          const t = g.querySelector(".node-value");
          if (t) t.textContent = fmt(x, 2);
        }
      });
      // Valor del bias (debajo de su círculo).
      {
        const g = this.el.neuronSvg.querySelector(`[data-node="bias"]`);
        if (g) {
          const t = g.querySelector(".node-value");
          if (t) t.textContent = fmt(n.bias, 3);
        }
      }
      // Net input: sobre la conexión Σ→activación (etiqueta .sum-label).
      setText(".sum-label", `Net = ${fmt(n.netInput)}`);
      // Salida de la activación (rectángulo) y nodo de salida.
      {
        const g = this.el.neuronSvg.querySelector(`[data-node="act"]`);
        if (g) {
          const t = g.querySelector(".node-value");
          if (t) t.textContent = `y = ${fmt(n.output, 3)}`;
        }
      }
      {
        const g = this.el.neuronSvg.querySelector(`[data-node="out"]`);
        if (g) {
          const t = g.querySelector(".node-value");
          if (t) t.textContent = fmt(n.output, 4);
        }
      }
    }

    /* ====================================================================
     * Animación del flujo de información (~300 ms en cascada).
     * Ilumina conexiones y nodos en el orden:
     *   entradas → conexiones entrada→Σ → Σ → Σ→act → activación →
     *   act→out → salida.
     * =================================================================== */
    _animateFlow() {
      const svg = this.el.neuronSvg;
      const inputConns = svg.querySelectorAll(".conn-input");
      const biasConn = svg.querySelector(".conn-bias");
      const sumConn = svg.querySelector(".conn-sum");
      const actConn = svg.querySelector(".conn-act");
      const nodes = svg.querySelectorAll(".node");

      // Limpiar temporizadores anteriores para evitar fugas y parpadeos
      if (this._flowTimeouts) {
        this._flowTimeouts.forEach(clearTimeout);
      }
      this._flowTimeouts = [];

      // Limpiar estados previos.
      svg.querySelectorAll(".lit").forEach((el) => el.classList.remove("lit"));

      const litMs = 420;   // duración del brillo de cada elemento
      const fire = (el, delay) => {
        if (!el) return;
        const id1 = setTimeout(() => {
          el.classList.add("lit");
          const id2 = setTimeout(() => el.classList.remove("lit"), litMs);
          this._flowTimeouts.push(id2);
        }, delay);
        this._flowTimeouts.push(id1);
      };

      // 1) Nodos de entrada + sus conexiones (casi simultáneos).
      nodes.forEach((node) => {
        if (node.classList.contains("node-input")) fire(node, 0);
      });
      inputConns.forEach((c) => fire(c, 30));
      fire(biasConn, 30);
      fire(svg.querySelector(".node-bias"), 0);

      // 2) Sumatoria.
      fire(svg.querySelector(".node-sum"), 110);

      // 3) Σ → activación + nodo activación.
      fire(sumConn, 170);
      fire(svg.querySelector(".node-act"), 220);

      // 4) Activación → salida + nodo salida.
      fire(actConn, 260);
      fire(svg.querySelector(".node-out"), 300);
    }

    /* ====================================================================
     * Sincronización de controles desde el controlador (presets).
     * =================================================================== */
    syncControls() {
      const n = this.neuron;
      this._inputRows.forEach((row, i) => row._setValue(n.inputs[i]));
      this._weightRows.forEach((row, i) => row._setValue(n.weights[i]));
      this._biasRow._setValue(n.bias);
    }
  }

  window.NeuronUI = NeuronUI;
})();
