import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  // Criamos um modelo sequencial simples
  const model = tf.sequential();

  // Primeira camada densa com 80 neurônios e função de ativação ReLU, recebendo 7 entradas (idade + 3 cores + 3 localizações)

  // 80 neurônios porque temos um dataset pequeno, mas queremos que o modelo tenha capacidade de aprender padrões complexos. 
  // A função de ativação ReLU é uma escolha comum para camadas ocultas, pois ajuda a introduzir não linearidade no modelo, 
  // permitindo que ele aprenda relações mais complexas entre as características de entrada e as categorias de saída.

  // Adicionamos uma camada densa com 80 neurônios e função de ativação ReLU
  model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }));

  // Adicionamos a camada de saída com 3 neurônios (para as 3 categorias) e função de ativação softmax
  // A função de ativação softmax é usada na camada de saída para problemas de classificação multiclasse, 
  // pois ela converte as saídas em probabilidades que somam 1, facilitando a interpretação dos resultados como categorias previstas.
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  // Compilamos o modelo com otimizador Adam e função de perda categoricalCrossentropy

  // O otimizador Adam é uma escolha popular para treinamento de redes neurais, 
  // pois combina as vantagens de outros otimizadores e geralmente converge mais rápido.

  // A função de perda categoricalCrossentropy é adequada para problemas de classificação multiclasse, 
  // onde as saídas são representadas como vetores one-hot encoded, como é o caso do nosso dataset.
  
  // A métrica de acurácia é usada para avaliar o desempenho do modelo durante o treinamento, 
  // indicando a proporção de previsões corretas em relação ao total de amostras.

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Treinamos o modelo por um número definido de épocas
  await model.fit(inputXs, outputYs, {
    verbose: 0, // Desativa a saída detalhada do treinamento
    epochs: 100, // Número de épocas para treinar o modelo
    shuffle: true, // Embaralha os dados a cada época para melhorar a generalização do modelo
    // callbacks: {
    //   onEpochEnd: (epoch, logs) => {
    //     console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
    //   },
    // },
  });

  return model;
}

async function predict(model, pessoaNovaNormalizada) {
  // Temos que converter os dados de entrada para um tensor 2D, mesmo que seja apenas uma amostra
  const inputTensor = tf.tensor2d(pessoaNovaNormalizada);

  const prediction = model.predict(inputTensor);
  
  // A previsão é um tensor, então precisamos extrair os valores para interpretar a categoria prevista
  const predictedCategory = await prediction.array()
  return predictedCategory[0].map((prob, index) => ({
    prob, index
  }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

const models = await trainModel(inputXs, outputYs);

const pessoaNova = { nome: "Maria", idade: 28, cor: "azul", localizacao: "Rio" };

// Normalizamos os dados da nova pessoa
// Exemplo: idade minima = 25, idade máxima = 40, então idade_normalizada = (28 - 25) / (40 - 25) = 0.2
const idadeMin = 25;
const idadeMax = 40;
const idadeNormalizada = (pessoaNova.idade - idadeMin) / (idadeMax - idadeMin);

const pessoaNovaNormalizada = [
  [
    idadeNormalizada, // idade normalizada
    1, // azul
    0, // vermelho
    0, // verde
    0, // São Paulo
    1, // Rio
    0, // Curitiba
  ],
];

const predictions = await predict(models, pessoaNovaNormalizada);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map(pred => `${labelsNomes[pred.index]}: ${(pred.prob * 100).toFixed(2)}%`)
  .join("\n");

console.log(results);