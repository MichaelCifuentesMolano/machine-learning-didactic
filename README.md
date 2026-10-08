# 🧠 Curso Interactivo de Redes Neuronales

Un simulador educativo interactivo, visual y modular diseñado en **HTML5, CSS3 y Vanilla JavaScript** (sin frameworks ni librerías pesadas) para aprender y enseñar los fundamentos de las redes neuronales y el aprendizaje automático de manera intuitiva y práctica.

El proyecto es **100% estático**: basta con hacer doble clic en `index.html` en cualquier computadora. Las ecuaciones se renderizan con KaTeX desde un CDN; sin conexión a internet se muestran en texto plano.

---

## 📚 Bloques Didácticos

El curso se divide en tres bloques progresivos:

### 1. [Bloque 1: La Neurona / Perceptrón](bloque1.html)
* **Objetivo:** Comprender cómo una neurona combina entradas con pesos y bias para calcular un valor de entrada neto (*Net Input*) y aplicar una función de activación.
* **Características:**
  * Número de entradas dinámico y escalable (2, 3 o 4 entradas).
  * Diagrama animado de la neurona en SVG que ilustra el flujo de información.
  * Ecuación y sustitución numérica renderizada en tiempo real mediante **KaTeX**.
  * Gráfico interactivo del comportamiento de la función de activación seleccionada.
  * Historial de las últimas 20 evaluaciones realizadas.

### 2. [Bloque 2: La Frontera de Decisión](bloque2.html)
* **Objetivo:** Descubrir que los pesos y el bias controlan la orientación y posición de la recta que separa dos clases en un plano cartesiano.
* **Características:**
  * Plano cartesiano interactivo 2D dibujado en Canvas con puntos aleatorios de dos clases (azul y rojo).
  * Interacción directa: el estudiante puede mover y arrastrar el punto de prueba en el plano usando el ratón o pantalla táctil para ver cómo cambia la predicción de la neurona.
  * Sliders reactivos para modificar los pesos y el bias y ver cómo se desplaza la frontera de decisión ($x_1 \cdot w_1 + x_2 \cdot w_2 + b = 0$) instantáneamente.

### 3. [Bloque 3: La Regla de Aprendizaje del Perceptrón](bloque3.html)
* **Objetivo:** Aprender cómo la neurona puede aprender de forma autónoma y ajustar sus propios pesos y bias a partir de sus errores.
* **Características:**
  * Usa por defecto la función escalón (*Binary Step*), como el perceptrón clásico.
  * Tasa de aprendizaje ($\eta$) configurable mediante controles deslizantes.
  * **Aprender una vez:** Evalúa secuencialmente los puntos del plano, calcula el error ($e = d - y$) y muestra paso a paso las fórmulas de actualización de la regla del perceptrón ($\Delta w_i = \eta \cdot e \cdot x_i$, $\Delta b = \eta \cdot e$) en LaTeX profesional.
  * **Entrenar época:** Ejecuta el entrenamiento de forma animada en todo el conjunto de datos, permitiendo visualizar literalmente cómo la frontera de decisión se ajusta sola hasta lograr separar las clases.

---

## 🛠️ Tecnologías Utilizadas

* **HTML5:** Estructura semántica de los bloques.
* **CSS3 (Vanilla):** Diseño moderno estilo "editor de nodos" con modo claro, variables CSS y animaciones fluidas.
* **JavaScript (ES6):** Programación orientada a objetos (modelo matemático modular `Neuron` en [neuron.js](js/neuron.js)).
* **KaTeX:** Biblioteca matemática ultra rápida cargada mediante CDN para representar las ecuaciones y desgloses numéricos con calidad de publicación científica (LaTeX).
* **HTML5 Canvas & SVG:** Renderizado del plano cartesiano 2D e ilustración interactiva de la neurona.

---

## 🚀 Cómo Empezar

1. Descarga o clona este repositorio:
   ```bash
   git clone https://github.com/MichaelCifuentesMolano/machine-learning-didactic.git
   ```
2. Ve al directorio del proyecto y abre el archivo `index.html` haciendo doble clic en él o arrastrándolo a cualquier navegador web moderno (Chrome, Firefox, Safari, Edge).

*Nota: No se requiere instalar Node.js, Python, ni ningún servidor web local.*

---

## 📂 Estructura del Proyecto

```text
Machine Learning Didactic/
├── index.html          # Índice del curso y menú principal
├── bloque1.html        # Simulación de la neurona básica
├── bloque2.html        # Visualización de la frontera de decisión
├── bloque3.html        # Entrenamiento con la regla del perceptrón
├── css/
│   ├── index.css       # Estilos específicos del índice
│   └── styles.css      # Estilos comunes de los bloques didácticos
└── js/
    ├── activationFunctions.js  # Definición y fórmulas de funciones de activación
    ├── neuron.js               # Clase Neuron (Modelo matemático modular)
    ├── ui.js                   # Vista interactiva del Bloque 1 (NeuronUI)
    ├── app.js                  # Controlador principal del Bloque 1
    ├── bloque2.js              # Controlador e interacción del Bloque 2
    └── bloque3.js              # Controlador del aprendizaje del Bloque 3
```

---

## ⚖️ Licencia

Este proyecto es de uso educativo y libre. Siéntete libre de clonarlo, modificarlo y usarlo en tus clases o proyectos personales de aprendizaje automático.
