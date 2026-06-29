/* ===========================================================================
 * app.js
 * ---------------------------------------------------------------------------
 * Controlador principal. Punto de entrada de la aplicación.
 *
 * Responsabilidades:
 *   - Instanciar el modelo (Neuron) a partir de una configuración inicial.
 *   - Instanciar la vista (NeuronUI) enlazando los contenedores del DOM.
 *   - Conectar los callbacks de la vista (selector de activación, botones).
 *   - Gestionar los presets (Ejemplo 1 / Ejemplo 2), aleatorizar y reiniciar.
 *   - Mantener el historial de las últimas 20 evaluaciones.
 *
 * No contiene lógica matemática ni manipulación de estilo compleja: esa
 * responsabilidad vive en neuron.js y ui.js respectivamente. Aquí se orquesta.
 * =========================================================================== */

(function () {
  "use strict";

  /* ----------------------------------------------------------------------
   * Configuración inicial.
   *
   * El nº de entradas se deriva de este arreglo. Si en el futuro se añaden
   * más elementos, toda la interfaz se reconstruye automáticamente
   * (requisito de escalabilidad: nada "hardcoded" a 4 entradas).
   * -------------------------------------------------------------------- */
  const INITIAL_INPUTS  = [1.0, -0.5];
  const INITIAL_WEIGHTS = [0.5, -0.3];
  const INITIAL_BIAS    = 0.1;
  const INITIAL_ACTIVATION = "sigmoid";

  /** Máximo nº de entradas del historial. */
  const HISTORY_MAX = 20;

  /* ----------------------------------------------------------------------
   * Presets pedagógicos.
   * Cada preset define inputs, weights, bias y función de activación.
   * -------------------------------------------------------------------- */
  const PRESETS = {
    example1: {
      title: "Ejemplo 1 — Sigmoid (probabilidad)",
      inputs:  [2.0, 1.5, -1.0, 0.5],
      weights: [0.45, -0.72, 0.6, 0.3],
      bias:    0.2,
      activation: "sigmoid"
    },
    example2: {
      title: "Ejemplo 2 — ReLU (red profunda)",
      inputs:  [-1.0, 3.0, 0.5, -2.0],
      weights: [0.8, -0.4, 1.2, 0.6],
      bias:    -0.5,
      activation: "relu"
    },
    // Estado "de fábrica" para el botón Reiniciar.
    default: {
      title: "Configuración inicial",
      inputs:  INITIAL_INPUTS,
      weights: INITIAL_WEIGHTS,
      bias:    INITIAL_BIAS,
      activation: INITIAL_ACTIVATION
    }
  };

  /* ----------------------------------------------------------------------
   * Funciones utilitarias.
   * -------------------------------------------------------------------- */

  /**
   * Genera un número aleatorio dentro de [min, max].
   * @param {number} min
   * @param {number} max
   * @returns {number}
   */
  function randomBetween(min, max) {
    return Math.random() * (max - min) + min;
  }

  /* ----------------------------------------------------------------------
   * Punto de arranque: se ejecuta cuando el DOM está listo.
   * -------------------------------------------------------------------- */

  function init() {
    // 1) Crear el modelo.
    const neuron = new window.Neuron(
      INITIAL_INPUTS,
      INITIAL_WEIGHTS,
      INITIAL_BIAS,
      INITIAL_ACTIVATION
    );

    // 2) Recoger las referencias del DOM definidas en index.html.
    const elements = {
      // Paneles de control.
      inputsPanel:     document.getElementById("inputs-panel"),
      weightsPanel:    document.getElementById("weights-panel"),
      biasPanel:       document.getElementById("bias-panel"),
      activationList:  document.getElementById("activation-list"),
      // Zona central: SVG de la neurona (nodos circulares).
      neuronSvg:       document.getElementById("neuron-svg"),
      equation:        document.getElementById("equation"),
      numeric:         document.getElementById("numeric"),
      graphCanvas:     document.getElementById("graph-canvas"),
      // Panel de salida.
      outputValue:     document.getElementById("output-value"),
      outputActivation:document.getElementById("output-activation"),
      outputNet:       document.getElementById("output-net"),
      // Historial.
      historyList:     document.getElementById("history-list")
    };

    // 3) Crear la vista, que a su vez se suscribe al modelo.
    const ui = new window.NeuronUI(neuron, elements);

    /* ------------------------------------------------------------------
     * Conexión de los callbacks de la vista hacia el modelo/controlador.
     * ---------------------------------------------------------------- */

    // Selector de función de activación.
    ui.onActivationChange = (key) => neuron.setActivation(key);

    // Botones: delegan en funciones locales del controlador.
    ui.onRandomize = () => randomizeWeights(neuron, ui);
    ui.onReset     = () => applyPreset(neuron, ui, "default");
    ui.onExample1  = () => applyPreset(neuron, ui, "example1");
    ui.onExample2  = () => applyPreset(neuron, ui, "example2");

    // Selector de número de entradas (2, 3 o 4).
    const inputsCountSelect = document.getElementById("inputs-count");
    if (inputsCountSelect) {
      inputsCountSelect.value = neuron.inputs.length.toString();
      inputsCountSelect.addEventListener("change", () => {
        const count = parseInt(inputsCountSelect.value, 10);
        if (Number.isFinite(count)) changeInputCount(neuron, ui, count);
      });
    }

    /* ------------------------------------------------------------------
     * Historial: cada vez que la neurona cambia, registramos la
     * evaluación (respetando el límite de HISTORY_MAX entradas).
     * ---------------------------------------------------------------- */

    /** @type {Array} buffer en memoria del historial. */
    const history = [];

    neuron.subscribe((snapshot) => {
      // Ignorar cambios transitorios (ej. arrastre continuo del slider) en el historial.
      if (snapshot.transient) return;
      // El historial guarda instantáneas legibles.
      history.unshift(neuron.snapshot());
      if (history.length > HISTORY_MAX) history.length = HISTORY_MAX;
      renderHistory(history, elements.historyList);
    });

    // 4) Repintado inicial explícito (la suscripción ya pintará una vez,
    //    pero garantizamos controles y diagrama sincronizados).
    ui.render();
    ui.syncControls();

    // Exponemos opcionalmente para depuración en consola.
    window._app = { neuron, ui, history };
  }

  /* ----------------------------------------------------------------------
   * Acciones del controlador.
   * -------------------------------------------------------------------- */

  /**
   * Aleatoriza pesos y bias dentro de sus rangos válidos.
   * @param {Neuron} neuron
   * @param {NeuronUI} ui
   */
  function randomizeWeights(neuron, ui) {
    // Pesos: rango [-5, 5].
    neuron.weights.forEach((_, i) => {
      neuron.setWeight(i, randomBetween(-5, 5));
    });
    // Bias: rango [-5, 5].
    neuron.setBias(randomBetween(-5, 5));
    ui.syncControls();
  }

  /**
   * Aplica un preset completo al modelo.
   * Reconstruye los paneles si el nº de entradas cambia (escalabilidad).
   * @param {Neuron} neuron
   * @param {NeuronUI} ui
   * @param {string} presetKey Clave en PRESETS.
   */
  function applyPreset(neuron, ui, presetKey) {
    const p = PRESETS[presetKey];
    if (!p) return;

    // Si el nº de entradas difiere, lo ajustamos primero.
    if (p.inputs.length !== neuron.inputs.length) {
      changeInputCount(neuron, ui, p.inputs.length);
    }

    // Actualizamos los valores agrupados en un lote para renderizar solo una vez.
    neuron.batchUpdate(() => {
      p.inputs.forEach((v, i) => neuron.setInput(i, v));
      p.weights.forEach((v, i) => neuron.setWeight(i, v));
      neuron.setBias(p.bias);
      neuron.setActivation(p.activation);
    });
    ui.syncControls();
  }

  /**
   * Cambia el número de entradas de la neurona, conservando los valores
   * existentes cuando es posible y rellenando con valores por defecto los
   * nuevos. Reconstruye los paneles de control y el diagrama.
   *
   * @param {Neuron} neuron
   * @param {NeuronUI} ui
   * @param {number} newCount Nuevo número de entradas.
   */
  function changeInputCount(neuron, ui, newCount) {
    const old = neuron.inputs.length;
    if (newCount === old) return;

    // Valores por defecto razonables para entradas/pesos nuevos.
    const DEFAULT_INPUT  = 1.0;
    const DEFAULT_WEIGHT = 0.5;

    if (newCount > old) {
      // Añadir entradas/pesos nuevos.
      for (let i = old; i < newCount; i++) {
        neuron.inputs.push(DEFAULT_INPUT);
        neuron.weights.push(DEFAULT_WEIGHT);
      }
    } else {
      // Recortar a newCount.
      neuron.inputs.length  = newCount;
      neuron.weights.length = newCount;
    }

    // Reconstruir paneles de control con el nuevo tamaño.
    ui._buildInputsPanel();
    ui._buildWeightsPanel();
    ui._buildBiasPanel();

    // Forzar recálculo + repintado (incluido el diagrama, que se reconstruye
    // solo porque cambia el nº de entradas).
    neuron.compute();
    neuron._emit();
    ui.syncControls();

    // Sincronizar el valor del selector dropdown si existe.
    const inputsCountSelect = document.getElementById("inputs-count");
    if (inputsCountSelect) {
      inputsCountSelect.value = newCount.toString();
    }
  }

  /* ----------------------------------------------------------------------
   * Render del historial.
   * -------------------------------------------------------------------- */

  /**
   * Dibuja la lista de las últimas evaluaciones.
   * @param {Array} history
   * @param {HTMLElement} container
   */
  function renderHistory(history, container) {
    if (history.length === 0) {
      container.innerHTML = `<li class="history-empty">Aún no hay evaluaciones.</li>`;
      return;
    }
    container.innerHTML = history.map((h, idx) => {
      const inputsTxt = h.inputs.map((x) => x.toFixed(1)).join(", ");
      return (
        `<li class="history-item">
           <span class="h-idx">#${history.length - idx}</span>
           <span class="h-block"><em>Entradas</em> [${inputsTxt}]</span>
           <span class="h-arrow">↓</span>
           <span class="h-block"><em>Net</em> ${h.netInput.toFixed(3)}</span>
           <span class="h-arrow">↓</span>
           <span class="h-block"><em>f</em> ${h.activationName}</span>
           <span class="h-arrow">↓</span>
           <span class="h-block h-out"><em>Salida</em> ${h.output.toFixed(4)}</span>
         </li>`
      );
    }).join("");
  }

  /* ----------------------------------------------------------------------
   * Arranque.
   * -------------------------------------------------------------------- */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
