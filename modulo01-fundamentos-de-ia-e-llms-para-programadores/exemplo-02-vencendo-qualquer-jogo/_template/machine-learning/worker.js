importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENTIONS = 640; // Tamanho de entrada esperado pelo modelo (ex: 640x640)
const CLASS_THRESHOLD = 0.6; // Limite de confiança para filtrar previsões

let _labels = [];
let _model = null;
async function loadModelAndLabels() {
    await tf.ready()

    _labels = await (await fetch(LABELS_PATH)).json()
    _model = await tf.loadGraphModel(MODEL_PATH);
    
    // warmup - aquecendo o modelo para reduzir a latência na primeira previsão
    const dummyInput = tf.ones(_model.inputs[0].shape);
    _model.executeAsync(dummyInput);
    tf.dispose(dummyInput);

    postMessage({ type: "model-loaded" });
}

/**
 * Pré-processamento da imagem para o modelo YOLOv5n
 * @param {*} input 
 * @description O modelo YOLOv5n espera uma imagem de entrada com dimensões específicas e normalizada.
 * O pré-processamento inclui:
 * 1. Redimensionar a imagem para as dimensões esperadas pelo modelo (por exemplo, 640x640).
 * 2. Normalizar os valores dos pixels para o intervalo [0, 1].
 * 3. Adicionar uma dimensão de lote (batch dimension) para que a entrada seja compatível com o modelo.
 * 
 * - tf.tidy: Garante que os tensores intermediários sejam descartados para evitar vazamentos de memória.
 * - tf.browser.fromPixels: Converte a imagem de entrada em um tensor.
 * - tf.image.resizeBilinear: Redimensiona a imagem para as dimensões esperadas pelo modelo.
 * - tf.div: Normaliza os valores dos pixels dividindo por 255.
 * - tf.expandDims: Adiciona uma dimensão de lote para que a entrada seja compatível com o modelo.
 * 
 * O resultado final é um tensor de forma [1, 640, 640, 3] (ou as dimensões esperadas pelo modelo) que pode ser passado para o modelo para fazer previsões.
 * 
 * @returns {tf.Tensor} O tensor pré-processado pronto para ser usado como entrada para o modelo.
 */
function preprocessImage(input) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input);

        return tf.image
            .resizeBilinear(image, [INPUT_MODEL_DIMENTIONS, INPUT_MODEL_DIMENTIONS])
            .div(255)
            .expandDims(0);
    });
}

async function runInference(tensor) {
    const output = await _model.executeAsync(tensor);
    tf.dispose(tensor);

    // Assume que as 3 primeiras saídas do modelo correspondem a caixas delimitadoras, pontuações e classes
    const [boxes, scores, classes] = output.slice(0, 3);

    const [boxesData, scoresData, classesData] = await Promise.all([
        boxes.data(),
        scores.data(),
        classes.data()
    ]);

    output.forEach(t => t.dispose());

    return {
        boxes: boxesData,
        scores: scoresData,
        classes: classesData,
    };
}

/**
 * Filtra e processa as previsões do modelo, retornando apenas aquelas que correspondem à classe "kite" e têm uma pontuação acima do limite definido.
 * @param {*} predictions - Objeto contendo as caixas delimitadoras, pontuações e classes retornadas pelo modelo.
 * @param {*} width - Largura da imagem original.
 * @param {*} height - Altura da imagem original.
 * @description O processo de filtragem inclui:
 * 1. Iterar sobre as previsões retornadas pelo modelo.
 * 2. Verificar se a pontuação da previsão é maior que o limite definido (CLASS_THRESHOLD). Se não for, a previsão é ignorada.
 * 3. Verificar se a classe da previsão é "kite". Se não for, a previsão é ignorada.
    * 4. Para as previsões que passam pelos filtros, os dados relevantes (como coordenadas da caixa delimitadora, pontuação e classe) podem ser processados e retornados para uso posterior.
 */

function* processPrediction({ boxes, scores, classes }, width, height) {
    for (let index = 0; index < scores.length; index++) {
        if (scores[index] < CLASS_THRESHOLD) continue; // Filtra previsões com baixa confiança
        
        const label = _labels[classes[index]];
        if (label !== 'kite') continue; // Filtra apenas a classe "kite"

        let [x1, y1, x2, y2] = boxes.slice(index * 4, (index + 1) * 4);
        // Converte as coordenadas normalizadas para as coordenadas da imagem original
        x1 *= width;
        x2 *= width;
        y1 *= height;
        y2 *= height;

        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;
        const centerX = x1 + boxWidth / 2;
        const centerY = y1 + boxHeight / 2;

        yield {
            x: centerX,
            y: centerY,
            score: (scores[index] * 100).toFixed(2),
        };
    }
}

loadModelAndLabels()

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return;
    if (!_model) return;

    const image = preprocessImage(data.image);
    const { width, height } = data.image;

    const inferenceResults = await runInference(image);

    for (const prediction of processPrediction(inferenceResults, width, height)) {
      postMessage({
          type: 'prediction',
          ...prediction,
      });
    }

};

console.log('🧠 YOLOv5n Web Worker initialized');
