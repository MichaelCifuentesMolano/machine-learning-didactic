/* ===========================================================================
 * neuron.js
 * ---------------------------------------------------------------------------
 * Clase `Neuron` (orientada a objetos).
 *
 * Es el "motor matemático" de la simulación. No sabe nada del DOM: solo
 * contiene estado (entradas, pesos, bias, función de activación) y expone
 * métodos para consultarlo y modificarlo. La vista (ui.js / app.js) escucha
 * los cambios y se redibuja.
 *
 * Escalabilidad
 * -------------
 * No hay nada "hardcoded" a 4 entradas. El número de entradas se deriva del
 * tamaño del arreglo `inputs` que se pasa al constructor. Si en el futuro
 * se pasan 20 elementos, todo sigue funcionando.
 *
 * Para notificar cambios a la vista se utiliza el patrón observador: la vista
 * se suscribe con `subscribe()` y la neurona emite el evento "change".
 * =========================================================================== */

class Neuron {
  /**
   * @param {number[]} inputs             Valores iniciales de las entradas.
   * @param {number[]} weights            Pesos iniciales (misma longitud que inputs).
   * @param {number}   bias               Bias inicial.
   * @param {string}   activationFunction Clave de ActivationFunctions (p.ej. "sigmoid").
   */
  constructor(inputs, weights, bias, activationFunction) {
    // --- Validaciones defensivas ------------------------------------------
    if (!Array.isArray(inputs) || !Array.isArray(weights)) {
      throw new Error("inputs y weights deben ser arreglos.");
    }
    if (inputs.length !== weights.length) {
      throw new Error("inputs y weights deben tener la misma longitud.");
    }
    if (!window.ActivationFunctions ||
        !Object.prototype.hasOwnProperty.call(window.ActivationFunctions, activationFunction)) {
      throw new Error(`Función de activación desconocida: "${activationFunction}".`);
    }

    /** @type {number[]} Valores de las entradas x1..xn */
    this.inputs = [...inputs];

    /** @type {number[]} Pesos w1..wn */
    this.weights = [...weights];

    /** @type {number} Término independiente (bias). */
    this.bias = bias;

    /** @type {string} Clave de la función de activación activa. */
    this.activationFunction = activationFunction;

    // --- Cache de la última evaluación -----------------------------------
    // Evita recalcular y permite a la vista leer valores sin recalcular.
    this._netInput = 0;   // Σ xᵢ·wᵢ + bias
    this._output = 0;     // f(netInput)
    this._products = [];  // [x₁·w₁, x₂·w₂, ...]  para mostrar el desglose

    // --- Sistema de suscriptores (patrón observador) ----------------------
    /** @type {Function[]} */
    this._listeners = [];

    /** @type {boolean} Flag para evitar emitir eventos durante actualizaciones en bloque. */
    this._silent = false;

    // Cálculo inicial.
    this._recompute();
  }

  /* =======================================================================
   * Núcleo matemático
   * ===================================================================== */

  /**
   * Calcula el "net input": Σ xᵢ·wᵢ + bias.
   * También guarda el desglose producto a producto.
   * @returns {number}
   */
  computeNetInput() {
    this._products = this.inputs.map((x, i) => x * this.weights[i]);
    // `reduce` recorre los productos; no asumimos cuántos hay.
    const sum = this._products.reduce((acc, p) => acc + p, 0);
    return sum + this.bias;
  }

  /**
   * Aplica la función de activación al net input.
   * @param {number} netInput
   * @returns {number}
   */
  activate(netInput) {
    const fnObj = window.ActivationFunctions[this.activationFunction];
    return fnObj.fn(netInput);
  }

  /**
   * Orquesta el cálculo completo: net input + activación.
   * @returns {{ netInput: number, output: number, products: number[] }}
   */
  compute() {
    const netInput = this.computeNetInput();
    const output = this.activate(netInput);
    this._netInput = netInput;
    this._output = output;
    return { netInput, output, products: this._products };
  }

  /**
   * Agrupa múltiples modificaciones al estado de la neurona y notifica una sola vez al final.
   * @param {Function} callback
   */
  batchUpdate(callback) {
    const wasSilent = this._silent;
    this._silent = true;
    try {
      callback();
    } finally {
      this._silent = wasSilent;
    }
    this._recompute(false);
  }

  /**
   * Recalcula y avisa a los observadores. Es el único punto que emite el
   * evento "change", de modo que cualquier mutación centraliza aquí.
   * @param {boolean} transient
   * @private
   */
  _recompute(transient = false) {
    if (this._silent) return;
    this.compute();
    this._emit(transient);
  }

  /* =======================================================================
   * Setters (mutadores) — cada uno recalcula y notifica.
   * ===================================================================== */

  /**
   * Establece el valor de la entrada i-ésima.
   * @param {number} index
   * @param {number} value
   * @param {boolean} transient Indica si es un cambio transitorio (ej. arrastre de slider)
   */
  setInput(index, value, transient = false) {
    this._guardIndex(index);
    this.inputs[index] = value;
    this._recompute(transient);
  }

  /**
   * Establece el valor del peso i-ésimo.
   * @param {number} index
   * @param {number} value
   * @param {boolean} transient Indica si es un cambio transitorio (ej. arrastre de slider)
   */
  setWeight(index, value, transient = false) {
    this._guardIndex(index);
    this.weights[index] = value;
    this._recompute(transient);
  }

  /**
   * Establece el bias.
   * @param {number} value
   * @param {boolean} transient Indica si es un cambio transitorio (ej. arrastre de slider)
   */
  setBias(value, transient = false) {
    this.bias = value;
    this._recompute(transient);
  }

  /**
   * Cambia la función de activación por su clave.
   * @param {string} key Clave dentro de ActivationFunctions.
   * @param {boolean} transient Indica si es un cambio transitorio
   */
  setActivation(key, transient = false) {
    if (!window.ActivationFunctions ||
        !Object.prototype.hasOwnProperty.call(window.ActivationFunctions, key)) {
      throw new Error(`Función de activación desconocida: "${key}".`);
    }
    this.activationFunction = key;
    this._recompute(transient);
  }

  /* =======================================================================
   * Getters de conveniencia para la vista.
   * ===================================================================== */

  /** Devuelve el último net input calculado. */
  get netInput() { return this._netInput; }

  /** Devuelve la última salida calculada. */
  get output() { return this._output; }

  /** Devuelve el desglose de productos [x₁·w₁, ...]. */
  get products() { return [...this._products]; }

  /** Devuelve el objeto descriptivo de la función de activación activa. */
  get activationMeta() {
    return window.ActivationFunctions[this.activationFunction];
  }

  /* =======================================================================
   * Utilidades
   * ===================================================================== */

  /**
   * Valida un índice de entrada/peso.
   * @param {number} index
   * @private
   */
  _guardIndex(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.inputs.length) {
      throw new RangeError(`Índice fuera de rango: ${index}`);
    }
  }

  /* =======================================================================
   * Sistema de eventos (observador)
   * ===================================================================== */

  /**
   * Suscribe una función callback al evento "change".
   * @param {Function} listener
   */
  subscribe(listener) {
    if (typeof listener === "function") this._listeners.push(listener);
  }

  /**
   * Notifica a todos los suscriptores.
   * @private
   */
  _emit(transient = false) {
    if (this._silent) return;
    const snapshot = {
      inputs: [...this.inputs],
      weights: [...this.weights],
      bias: this.bias,
      netInput: this._netInput,
      output: this._output,
      products: [...this._products],
      activation: this.activationFunction,
      transient: transient
    };
    this._listeners.forEach((fn) => {
      try { fn(snapshot); } catch (e) { console.error("Listener de Neuron falló:", e); }
    });
  }

  /* =======================================================================
   * Serialización — útil para el historial de evaluaciones.
   * ===================================================================== */

  /** Devuelve una instantánea legible del estado actual. */
  snapshot() {
    return {
      inputs: [...this.inputs],
      weights: [...this.weights],
      bias: this.bias,
      netInput: this._netInput,
      output: this._output,
      activation: this.activationFunction,
      activationName: this.activationMeta.name
    };
  }
}

// Exponer globalmente para los demás scripts.
window.Neuron = Neuron;
