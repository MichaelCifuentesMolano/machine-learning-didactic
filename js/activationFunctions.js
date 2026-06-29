/* ===========================================================================
 * activationFunctions.js
 * ---------------------------------------------------------------------------
 * Catálogo de funciones de activación para la neurona.
 *
 * Cada función es un objeto independiente con la misma "forma" (interfaz):
 *
 *   {
 *     name:        Nombre legible para el selector.
 *     equation:    Ecuación matemática (texto plano / Unicode).
 *     description: Breve explicación pedagógica.
 *     fn:          Implementación numérica f(x).
 *     range:       Rango sugerido { xMin, xMax, yMin, yMax } para dibujar
 *                  la gráfica de forma legible.
 *   }
 *
 * Diseño "abierto / cerrado": para añadir una nueva función en el futuro
 * basta con agregar una entrada a este diccionario. No es necesario tocar
 * neuron.js, ui.js ni app.js. Esto satisface el requisito de escalabilidad.
 * =========================================================================== */

const ActivationFunctions = {

  /* --- Lineal ----------------------------------------------------------- */
  linear: {
    name: "Linear",
    equation: "f(x) = x",
    description:
      "La salida es idéntica a la entrada. No introduce no linealidad; " +
      "se usa en la capa de salida de problemas de regresión.",
    fn: (x) => x,
    range: { xMin: -6, xMax: 6, yMin: -6, yMax: 6 }
  },

  /* --- Escalón binario -------------------------------------------------- */
  binaryStep: {
    name: "Binary Step",
    equation: "f(x) = 1 si x ≥ 0,  0 en otro caso",
    description:
      "Devuelve 0 ó 1. Es la función del perceptrón clásico, pero no es " +
      "diferenciable, por lo que no se usa con backpropagation.",
    fn: (x) => (x >= 0 ? 1 : 0),
    range: { xMin: -6, xMax: 6, yMin: -0.5, yMax: 1.5 }
  },

  /* --- Sigmoide --------------------------------------------------------- */
  sigmoid: {
    name: "Sigmoid",
    equation: "f(x) = 1 / (1 + e⁻ˣ)",
    description:
      "Comprime cualquier valor al intervalo (0, 1). Útil para modelar " +
      "probabilidades. Suaviza la salida y es diferenciable.",
    fn: (x) => 1 / (1 + Math.exp(-x)),
    range: { xMin: -6, xMax: 6, yMin: 0, yMax: 1 }
  },

  /* --- Tangente hiperbólica --------------------------------------------- */
  tanh: {
    name: "Tanh",
    equation: "f(x) = tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)",
    description:
      "Aplana los valores al intervalo (−1, 1). Está centrada en cero, " +
      "lo que suele acelerar el aprendizaje frente a la sigmoide.",
    fn: (x) => Math.tanh(x),
    range: { xMin: -6, xMax: 6, yMin: -1, yMax: 1 }
  },

  /* --- ReLU ------------------------------------------------------------- */
  relu: {
    name: "ReLU",
    equation: "f(x) = máx(0, x)",
    description:
      "Si x es positivo devuelve x; si no, 0. Muy usada en capas ocultas " +
      "por su bajo coste computacional y porque mitiga el desvanecimiento " +
      "del gradiente.",
    fn: (x) => (x > 0 ? x : 0),
    range: { xMin: -6, xMax: 6, yMin: -1, yMax: 6 }
  },

  /* --- Leaky ReLU ------------------------------------------------------- */
  leakyRelu: {
    name: "Leaky ReLU",
    equation: "f(x) = x si x > 0,  α·x en otro caso   (α = 0.01)",
    description:
      "Variante de ReLU que permite un pequeño gradiente negativo (α·x) " +
      "evitando que las neuronas " +
      "\u201cmueran\u201d cuando la entrada es negativa.",
    fn: (x) => (x > 0 ? x : 0.01 * x),
    range: { xMin: -6, xMax: 6, yMin: -1, yMax: 6 }
  },

  /* --- ELU -------------------------------------------------------------- */
  elu: {
    name: "ELU",
    equation: "f(x) = x si x > 0,  α·(eˣ − 1) en otro caso   (α = 1)",
    description:
      "Exponential Linear Unit. Para valores negativos tiende suavemente " +
      "a −α, produciendo activaciones centradas en cero.",
    fn: (x) => (x > 0 ? x : 1 * (Math.exp(x) - 1)),
    range: { xMin: -6, xMax: 6, yMin: -1.5, yMax: 6 }
  },

  /* --- Softplus --------------------------------------------------------- */
  softplus: {
    name: "Softplus",
    equation: "f(x) = ln(1 + eˣ)",
    description:
      "Aproximación suave y diferenciable de ReLU. Siempre es positiva y " +
      "crece de forma logarítmica.",
    fn: (x) => Math.log(1 + Math.exp(x)),
    range: { xMin: -6, xMax: 6, yMin: 0, yMax: 6 }
  },

  /* --- Swish ------------------------------------------------------------ */
  swish: {
    name: "Swish",
    equation: "f(x) = x · σ(x)   donde σ es la sigmoide",
    description:
      "f(x) = x · sigmoid(x). No es monótona y suele superar a ReLU en " +
      "redes profundas. Descubierta mediante búsqueda automática.",
    fn: (x) => x / (1 + Math.exp(-x)),
    range: { xMin: -6, xMax: 6, yMin: -1, yMax: 6 }
  },

  /* --- GELU ------------------------------------------------------------- */
  gelu: {
    name: "GELU",
    equation: "f(x) = x · Φ(x)  ≈  x · ½·(1 + tanh(√(2/π)·(x + 0.044715·x³)))",
    description:
      "Gaussian Error Linear Unit. Multiplica la entrada por su propia " +
      "probabilidad gaussiana. Muy usada en modelos tipo Transformer (BERT, GPT).",
    fn: (x) => {
      const c = Math.sqrt(2 / Math.PI);
      const inner = c * (x + 0.044715 * Math.pow(x, 3));
      return 0.5 * x * (1 + Math.tanh(inner));
    },
    range: { xMin: -6, xMax: 6, yMin: -1, yMax: 6 }
  },

  /* --- Mish ------------------------------------------------------------- */
  mish: {
    name: "Mish",
    equation: "f(x) = x · tanh(softplus(x))",
    description:
      "f(x) = x · tanh(ln(1 + eˣ)). Suave, no monótona y acotada hacia " +
      "abajo. Suele mejorar los resultados de Swish y ReLU.",
    fn: (x) => x * Math.tanh(Math.log(1 + Math.exp(x))),
    range: { xMin: -6, xMax: 6, yMin: -1, yMax: 6 }
  }
};

// Exponer en window para que los demás módulos (cargados como <script>)
// puedan acceder sin necesidad de módulos ES6, manteniendo el orden simple.
window.ActivationFunctions = ActivationFunctions;
