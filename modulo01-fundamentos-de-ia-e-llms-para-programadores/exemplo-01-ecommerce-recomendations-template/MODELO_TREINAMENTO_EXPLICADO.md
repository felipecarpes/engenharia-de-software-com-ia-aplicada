# Sistema de Recomendação com Machine Learning - Análise Completa

## 📌 Visão Geral

A arquitetura separa bem **UI**, **regras de negócio** e **treino de modelo**. O fluxo é:

```
Interface → Browser Events → Controller → Web Worker → TensorFlow.js → Recomendações
```

### Pontos de Entrada Principais

| Arquivo                                                                  | Responsabilidade                                            |
| ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| [src/index.js](src/index.js)                                             | Bootstrap da aplicação e inicialização automática do treino |
| [src/controller/WorkerController.js](src/controller/WorkerController.js) | Orquestração da comunicação com o Web Worker                |
| [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js) | Treino e inferência do modelo TensorFlow.js                 |

---

## 🔄 Passo a Passo Completo do Treinamento

### 1. Inicialização e Treino Automático no Carregamento

**Arquivo**: [src/index.js](src/index.js#L31)

```javascript
const users = await userService.getDefaultUsers();
w.triggerTrain(users);
```

Quando a aplicação carrega:

- Usuários padrão são carregados de `data/users.json`
- O `WorkerController` dispara automaticamente o treino com `triggerTrain(users)`

### 2. Evento de Treino vai para o Worker

**Arquivo**: [src/controller/WorkerController.js](src/controller/WorkerController.js#L23)

O controller escuta o evento de treino:

```javascript
this.#events.onTrainModel((data) => {
  this.#alreadyTrained = false;
  this.triggerTrain(data);
});
```

A mensagem é enviada via `postMessage`:

```javascript
triggerTrain(users) {
    this.#worker.postMessage({ action: workerEvents.trainModel, users });
}
```

### 3. Construção do Contexto de Features

**Arquivo**: [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L18)

A função `makeContext(products, users)` cria a estrutura de metadados necessária:

#### 3.1 Normalização de Valores Numéricos

**Linha 16**: Função de normalização para escalar entre 0 e 1:

```javascript
const normalize = (value, min, max) => (value - min) / (max - min || 1);
```

Minagem de min/max:

- Idades: `Math.min(...ages)` e `Math.max(...ages)`
- Preços: `Math.min(...prices)` e `Math.max(...prices)`

#### 3.2 Criação de Índices One-Hot

**Linhas 28-36**: Extração e indexação de cores e categorias

```javascript
const colors = [...new Set(products.map((p) => p.color))];
const categories = [...new Set(products.map((p) => p.category))];

const colorsIndex = Object.fromEntries(
  colors.map((color, index) => [color, index]),
);
const categoriesIndex = Object.fromEntries(
  categories.map((category, index) => [category, index]),
);
```

**O que isso faz**: Transforma valores categóricos (ex: "vermelho", "azul") em índices numéricos (ex: 0, 1).

#### 3.3 Feature Comportamental: Média de Idade por Produto

**Linhas 45-60**: Computar a idade média dos compradores por produto

```javascript
const ageSums = {};
const ageCounts = {};

users.forEach((user) => {
  user.purchases.forEach((product) => {
    ageSums[product.name] = (ageSums[product.name] || 0) + user.age;
    ageCounts[product.name] = (ageCounts[product.name] || 0) + 1;
  });
});

const prodAgeAvgNorm = Object.fromEntries(
  products.map((product) => {
    const avg = ageCounts[product.name]
      ? ageSums[product.name] / ageCounts[product.name]
      : midAge;
    return [product.name, normalize(avg, minAge, maxAge)];
  }),
);
```

**Insight**: Se "Boné Estiloso" foi comprado por usuários com média de idade 35 anos, isso fica codificado no vetor. O modelo aprende que pessoas de idade similar têm mais chance de comprar boné.

#### 3.4 Dimensionalidade Final

**Linhas 71-75**: Cálculo da dimensão do vetor

```javascript
colorsQty: colors.length,
colorsQty: colors.length,
prodAgeAvgNorm,
// price + age + colors + categories
dimentions: 2 + colors.length + categories.length,
```

$$
d = 2 + |\text{cores}| + |\text{categorias}|
$$

Retorno do contexto com todos os metadados armazenados em `_globalCtx` para uso posterior.

---

### 4. Engenharia de Features com Pesos Manuais

**Arquivo**: [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L8)

Pesos que definem a importância relativa de cada feature:

```javascript
const WEIGHTS = {
  category: 0.4, // 40% da relevância
  color: 0.3, // 30% da relevância
  price: 0.2, // 20% da relevância
  age: 0.1, // 10% da relevância
};
```

**Interpretação**:

- Categoria é a **feature mais discriminativa** (40%)
- Cor tem relevância média (30%)
- Preço e idade têm importância menor mas não nula

Esses pesos são multiplicados nos tensores durante a codificação.

---

### 5. Codificação de Produto

**Arquivo**: [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L82)

```javascript
function encodeProduct(product, context) {
  // Preço normalizado × peso
  const price = tf.tensor1d([
    normalize(product.price, context.minPrice, context.maxPrice) *
      WEIGHTS.price,
  ]);

  // Idade média de compradores × peso
  const age = tf.tensor1d([
    (context.prodAgeAvgNorm[product.name] ?? 0.5) * WEIGHTS.age,
  ]);

  // One-hot de cor ponderado
  const color = oneHotWeighted(
    context.colorsIndex[product.color],
    context.colorsQty,
    WEIGHTS.color,
  );

  // One-hot de categoria ponderado
  const category = oneHotWeighted(
    context.categoriesIndex[product.category],
    context.categoriesQty,
    WEIGHTS.category,
  );

  // Concatena: [preço, idade, one_hot_cores, one_hot_categorias]
  return tf.concat1d([price, age, color, category]);
}
```

**Exemplo Concreto**:

Suponha:

- Cores = ["preto", "cinza", "azul"]
- Categorias = ["acessórios", "eletrônicos", "vestuário"]

Para "Boné Estiloso" (categoria: acessórios, cor: preto, preço: 39.99):

```
Preço normalizado:    [0.3 × 0.2]     = [0.06]
Idade média:          [0.6 × 0.1]     = [0.06]
One-hot cor (preto):  [1, 0, 0] × 0.3 = [0.3, 0, 0]
One-hot cat (acessórios): [1, 0, 0] × 0.4 = [0.4, 0, 0]

VETOR FINAL: [0.06, 0.06, 0.3, 0, 0, 0.4, 0, 0]
(8 dimensões)
```

---

### 6. Codificação de Usuário

**Arquivo**: [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L108)

#### 6.1 Usuário com Histórico de Compras

```javascript
if (user.purchases.length) {
  return tf
    .stack(user.purchases.map((product) => encodeProduct(product, context)))
    .mean(0) // Média dos vetores de produtos
    .reshape([1, context.dimentions]);
}
```

**O que faz**: Tira o vetor de cada produto que o usuário comprou, calcula a média vetorial.

**Intuição**: O "perfil" do usuário é representado como a média dos produtos que comprou.

#### 6.2 Usuário Sem Histórico

```javascript
return tf
  .concat1d([
    tf.zeros([1]), // preço = 0 (ignorado)
    tf.tensor1d([
      normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age,
    ]),
    tf.zeros([context.categoriesQty]), // categorias = 0
    tf.zeros([context.colorsQty]), // cores = 0
  ])
  .reshape([1, context.dimentions]);
```

**Fallback**: Se o usuário nunca comprou, usa apenas sua idade. Categorias e cores ficam zeradas.

---

### 7. Montagem do Dataset Supervisionado

**Arquivo**: [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L128)

```javascript
function createTrainData(context) {
  const inputs = [];
  const labels = [];

  context.users
    .filter((user) => user.purchases.length) // Apenas usuários com compras
    .forEach((user) => {
      const userVector = encodeUser(user, context).dataSync();

      context.products.forEach((product) => {
        const productVector = encodeProduct(product, context).dataSync();

        // Label: 1 se comprou, 0 se não comprou
        const label = user.purchases.some(
          (purchase) => purchase.name === product.name,
        )
          ? 1
          : 0;

        // Entrada = concatenação [userVector, productVector]
        inputs.push([...userVector, ...productVector]);
        labels.push(label);
      });
    });

  return {
    xs: tf.tensor2d(inputs),
    ys: tf.tensor2d(labels, [labels.length, 1]),
    inputDimention: context.dimentions * 2, // user + product
  };
}
```

#### 7.1 Estrutura do Dataset

Para cada usuário com compras e cada produto:

| Entrada                               | Rótulo |
| ------------------------------------- | ------ |
| [user_vector... \| product_vector...] | 1 ou 0 |

**Dimensionalidade**:

- Input: `2 × (2 + num_colors + num_categories)` = $2d$
- Output: 1 (probabilidade de compra)

**Exemplo**:

- Usuário Rafael comprou [Boné, Mochila]
- Há 10 produtos no catálogo
- Gera 10 linhas no dataset:
  - [rafael_vec \| boné_vec] → 1
  - [rafael_vec \| mochila_vec] → 1
  - [rafael_vec \| produto_3_vec] → 0
  - [rafael_vec \| produto_4_vec] → 0
  - ...

---

### 8. Construção da Rede Neural

**Arquivo**: [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L191)

```javascript
async function configureNeuralNetworkAndTrain(trainData) {
  const model = tf.sequential();

  // Camada de Entrada: 128 neurônios, ReLU
  model.add(
    tf.layers.dense({
      inputShape: [trainData.inputDimention], // 2d features
      units: 128,
      activation: "relu",
    }),
  );

  // Camada Oculta 1: 64 neurônios, ReLU
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));

  // Camada Oculta 2: 32 neurônios, ReLU
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));

  // Camada de Saída: 1 neurônio, Sigmoid (probabilidade)
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  // Compilação
  model.compile({
    optimizer: tf.train.adam(0.01), // Taxa de aprendizado: 0.01
    loss: "binaryCrossentropy", // Para classificação binária
    metrics: ["accuracy"],
  });

  // Treinamento
  await model.fit(trainData.xs, trainData.ys, {
    epochs: 100, // 100 ciclos completos do dataset
    batchSize: 32, // 32 exemplos por atualização
    shuffle: true, // Embaralhar ordem
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        postMessage({
          type: workerEvents.trainingLog,
          epoch,
          loss: logs.loss,
          accuracy: logs.acc,
        });
      },
    },
  });

  return model;
}
```

#### 8.1 Arquitetura Explicada

```
Input Layer (2d features)
        ↓
    ReLU (128) ← Expansão para capturar padrões não-lineares
        ↓
    ReLU (64)  ← Compressão gradual de informação
        ↓
    ReLU (32)  ← Extração de features de alto nível
        ↓
Sigmoid (1)   ← Probabilidade final [0, 1]
```

**Justificativa dos parâmetros**:

- **128 → 64 → 32 → 1**: Funil compressivo que destila informação
- **ReLU**: Mantém sinais positivos, permite aprender não-linearidades
- **Sigmoid**: Comprime saída para intervalo [0, 1] (interpretável como probabilidade)
- **Adam lr=0.01**: Otimizador adaptativo com taxa moderada
- **Binary Crossentropy**: Loss adequado para classificação 0/1
- **100 épocas, batch 32**: Convergência sem overfitting

#### 8.2 Telemetria de Cada Época

A cada época, métricas são enviadas via `postMessage`:

```javascript
callbacks: {
    onEpochEnd: (epoch, logs) => {
        postMessage({
            type: workerEvents.trainingLog,
            epoch,
            loss: logs.loss,
            accuracy: logs.acc,
        });
    },
}
```

---

### 9. Telemetria de Treino para UI

**Worker**: [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L233)

Os logs de treino são transmitidos a cada época e recebidos pelo controller:

**Controller**: [src/controller/WorkerController.js](src/controller/WorkerController.js#L63)

```javascript
if (event.data.type === workerEvents.trainingLog) {
  this.#events.dispatchTFVisLogs(event.data);
}
```

**View**: [src/view/TFVisorView.js](src/view/TFVisorView.js#L31)

Os logs são plotados em tempo real:

```javascript
handleTrainingLog(log) {
    const { epoch, loss, accuracy } = log;
    this.#lossPoints.push({ x: epoch, y: loss });
    this.#accPoints.push({ x: epoch, y: accuracy });

    // Gráfico de Precisão
    tfvis.render.linechart(
        { name: 'Precisão do Modelo', tab: 'Treinamento' },
        { values: this.#accPoints, series: ['precisão'] },
        { xLabel: 'Época', yLabel: 'Precisão (%)' }
    );

    // Gráfico de Erro
    tfvis.render.linechart(
        { name: 'Erro de Treinamento', tab: 'Treinamento' },
        { values: this.#lossPoints, series: ['erros'] },
        { xLabel: 'Época', yLabel: 'Valor do Erro' }
    );
}
```

**Resultado**: Dois gráficos lado a lado mostram Loss diminuindo e Accuracy aumentando.

---

### 10. Conclusão de Treino e Habilitação de Recomendação

**Worker**: [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L278)

Após 100 épocas, o worker envia sinais de conclusão:

```javascript
postMessage({
  type: workerEvents.progressUpdate,
  progress: { progress: 100 },
});
postMessage({ type: workerEvents.trainingComplete });
```

**Controller**: [src/controller/ModelTrainingController.js](src/controller/ModelTrainingController.js#L37)

```javascript
this.#events.onTrainingComplete(() => {
  this.#alreadyTrained = true;
  if (!this.#currentUser) return;
  this.#modelView.enableRecommendButton(); // Habilita botão
});
```

**Lógica**:

- Se modelo está treinado E usuário selecionado → Botão ativo
- Se modelo NÃO está treinado E usuário selecionado → Botão inativo

---

### 11. Inferência - Gerando Recomendações

**Arquivo**: [src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L285)

```javascript
function recommend(user) {
  if (!_model) return;
  const context = _globalCtx;

  // 1️⃣ Codificar usuário no mesmo formato de treino
  const userVector = encodeUser(user, context).dataSync();

  // 2️⃣ Criar pares: usuario + cada produto
  const inputs = context.productVectors.map(({ vector }) => {
    return [...userVector, ...vector];
  });

  // 3️⃣ Converter para Tensor
  const inputTensor = tf.tensor2d(inputs);

  // 4️⃣ Rodar modelo em lote
  const predictions = _model.predict(inputTensor);

  // 5️⃣ Extrair scores e ordenar
  const scores = predictions.dataSync();
  const recommendations = context.productVectors.map((item, index) => {
    return {
      ...item.meta,
      name: item.name,
      score: scores[index], // Score do modelo [0, 1]
    };
  });
  const sortedItems = recommendations.sort((a, b) => b.score - a.score);

  // 6️⃣ Retornar para UI
  postMessage({
    type: workerEvents.recommend,
    user,
    recommendations: sortedItems,
  });
}
```

#### 11.1 Fluxo de Recomendação

```
User Select → ProductController (detecta seleção)
              ↓
        dispatchRecommend(user)
              ↓
  WorkerController recebe evento
              ↓
  postMessage({ action: 'recommend', user })
              ↓
  Worker: recommend(user)
        • encodeUser
        • cria pares user+produto
        • model.predict(pares)
        • ordena por score
              ↓
  postMessage({ recommendations: [...] })
              ↓
  ProductView renderiza produtos ordenados
```

**Resultado**: Produtos com score alto aparecem primeiro.

---

## 📥 Como os Dados Entram Nesse Ciclo

### Entrada 1: Usuários

**Serviço**: [src/service/UserService.js](src/service/UserService.js#L4)

```javascript
async getDefaultUsers() {
    const response = await fetch('./data/users.json');
    const users = await response.json();
    this.#setStorage(users);  // Persiste em sessionStorage
    return users;
}
```

**Persistência**: [src/service/UserService.js](src/service/UserService.js#L38)

```javascript
#getStorage() {
    const data = sessionStorage.getItem('ew-academy-users');
    return data ? JSON.parse(data) : [];
}

#setStorage(data) {
    sessionStorage.setItem('ew-academy-users', JSON.stringify(data));
}
```

### Entrada 2: Produtos

**Carregamento uma vez no worker**:

[src/workers/modelTrainingWorker.js](src/workers/modelTrainingWorker.js#L255)

```javascript
const products = await (await fetch("/data/products.json")).json();
```

### Entrada 3: Compra (Adição de Produto ao Usuário)

**Controller**: [src/controller/UserController.js](src/controller/UserController.js#L54)

```javascript
async handlePurchaseAdded({ user, product }) {
    const updatedUser = await this.#userService.getUserById(user.id);
    updatedUser.purchases.push({ ...product })
    await this.#userService.updateUser(updatedUser);

    // Dispara event para atualizar UI e listas
    this.#events.dispatchUsersUpdated({
        users: await this.#userService.getUsers()
    });
}
```

**View**: [src/view/ModelTrainingView.js](src/view/ModelTrainingView.js#L62)

```javascript
renderAllUsersPurchases(users) {
    const html = users.map(user => {
        const purchasesHtml = user.purchases.map(purchase => {
            return `<span class="badge">${purchase.name}</span>`;
        }).join('');
        return `<div class="user-purchase-summary">
            <h6>${user.name}</h6>
            ${purchasesHtml}
        </div>`;
    }).join('');
    this.#allUsersPurchasesList.innerHTML = html;
}
```

---

## 🎓 Leitura Didática - O Que Foi Implementado

Você implementou um **classificador supervisionado de compatibilidade usuário-produto**.

### Tipo de Modelo

| Aspecto                 | Implementação                                |
| ----------------------- | -------------------------------------------- |
| **Tipo de Aprendizado** | Supervisionado (rótulos 0/1)                 |
| **Tarefa**              | Classificação Binária (vai comprar? sim/não) |
| **Features**            | Engenharia manual + one-hot encoding         |
| **Arquitetura**         | MLP Denso (feed-forward)                     |
| **Saída**               | Probabilidade [0, 1]                         |

### Por Que Essa Abordagem?

1. **Feature Engineering Manual**: Permite controle total sobre o que o modelo vê
2. **Dados Supervisionados**: Rótulo claro (comprou / não comprou)
3. **Web Worker**: Treino não bloqueia UI
4. **TensorFlow.js**: Tudo no navegador, sem servidor
5. **Visualização em Tempo Real**: TFVis mostra convergência live

### Insights que Seu Modelo Captura

- **Similaridade de Preferências**: Usuários que compram itens parecidos
- **Demografia de Produtos**: Idade média dos compradores de cada item
- **Feature Interação**: ReLU permite combinações não-lineares
- **Padrão de Compra**: Se comprou categoria X, provavelmente vai comprar Y

### Limitações (Para Discussão em Aula)

1. **Sem Contexto Temporal**: Não diferencia compra recente vs. antiga
2. **Sem Contexto Externo**: Preço estático, sem promoções
3. **Sem Colaborativo Puro**: Não usa "users similar to you"
4. **Sem Embedding Aprendido**: One-hot é fixo, não se ajusta
5. **Retreino Manual**: Não há retreino automático com novas compras

---

## ⚙️ Observações Técnicas Importantes

### 1. Inconsistência no Barramento de Eventos

**Problema**: Em [src/events/constants.js](src/events/constants.js#L1), os eventos de tfvis NÃO estão declarados:

```javascript
export const events = {
  // ... sim
  // `tfvisLogs` e `tfvisData` não aparecem aqui!
};
```

Mas são usados em [src/events/events.js](src/events/events.js#L54):

```javascript
static onTFVisLogs(callback) {
    document.addEventListener(events.tfvisLogs, (event) => {
        // ☝️ Isso usa constante que não existe em constants.js
    });
}
```

**Solução**: Deveria estar em constants.js:

```javascript
export const events = {
  // ... existing
  tfvisLogs: "tfvis:logs",
  tfvisData: "tfvis:data",
};
```

### 2. Método Duplicado em Events

[src/events/events.js](src/events/events.js#L79) e [src/events/events.js](src/events/events.js#L143):

Ambas definem `onProgressUpdate`. A segunda sobrescreve a primeira.

**Fix**: Remover uma delas.

### 3. Retreino Não Automático

Novas compras atualizam o usuário em sessionStorage, mas **não disparam novo treino automático**.

**Fluxo Atual**:

- User compra produto → atualizado em storage
- Recomendação usa modelo velho
- Treino só acontece no botão ou reload

**Consideração**: Para produção, poderia haver:

- Incremental learning
- Scheduled retraining
- Hard trigger no botão (atual)

### 4. One-Hot Encoding Fixo

Cores e categorias são extraídas uma vez no contexto e **nunca mudam**.

Se novos produtos forem adicionados com cores/categorias nunca vistas, quebram a codificação.

**Solução**: Extrair one-hot dinâmico ou usar embedding aprendido (autoencoder, entity embeddings).

### 5. Performance em Escala

A recomendação roda `model.predict()` **para cada produto**.

Em produção (milhões de produtos), usar:

- Vector DB (Pinecone, Weaviate)
- Approximate Nearest Neighbors
- Batch inference em servidor

---

## 📊 Diagrama do Fluxo Completo

```
┌──────────────────────────────────────┐
│   Carregamento da Aplicação          │
│   (src/index.js)                     │
└─────────────────┬──────────────────┘
                  │
                  ↓
        ┌─────────────────┐
        │ UserController  │ ← Renderiza dropdown de usuários
        │ ProductCtroller │ ← Renderiza catálogo inicial
        └────────┬────────┘
                 │
    ┌────────────┴─────────────┐
    │ Treino Automático Inicial │
    │ (w.triggerTrain(users))  │
    └────────────┬─────────────┘
                 │
                 ↓
    ╔════════════════════════════╗
    ║  Web Worker (Background)   ║
    ║ makeContext()              ║
    ║ encodeProduct/User()       ║
    ║ createTrainData()          ║
    ║ configureNeuralNetwork()   ║
    ║ model.fit() - 100 épocas   ║
    ╚════════════┬═══════════════╝
                 │
  ┌──────────────┼──────────────┐
  │ A cada época │              │
  ↓              │              ↓
┌──────────┐    │         ┌──────────────┐
│  loss %  │    │         │  TFVisView   │
│ accuracy │    │         │  .linechart()│
└──────────┘    │         └──────────────┘
                ↓
    postMessage(trainingLog)

    ┌─────────────────────┐
    │ Treino Concluído    │
    │ model ← treinado    │
    │ _globalCtx ← saved  │
    └──────────┬──────────┘
               │
               ↓
    ┌─────────────────────┐
    │ Botão Recommend     │
    │ ativado             │
    └────────┬────────────┘
             │
    ┌────────┴─────────────────────────────────┐
    │  FLUXO DE COMPRA + RECOMENDAÇÃO          │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │ User Selected               │
    │ ProductController.onUserSel│
    │ dispatchRecommend(user)    │
    └──────────────┬──────────────┘
                   │
                   ↓
    ╔═══════════════════════════════╗
    ║  Web Worker: recommend(user)  ║
    ║ 1. encodeUser()               ║
    ║ 2. criar pares user+product   ║
    ║ 3. model.predict(pares)       ║
    ║ 4. sort() por score           ║
    ║ 5. postMessage(recommendations║
    ╚═════════════╤═════════════════╝
                  │
                  ↓
    ┌───────────────────────────┐
    │ ProductView.render()      │
    │ Renderiza produtos        │
    │ ordenados por score       │
    └───────────────────────────┘


    ┌──────────────────────────────┐
    │ User Clica "Buy Now"         │
    │ ProductController.handleBuy()│
    └──────────────┬───────────────┘
                   │
                   ↓
    ┌──────────────────────────────────┐
    │ UserController.handlePurchaseAdded│
    │ 1. add product to user.purchases │
    │ 2. update in sessionStorage      │
    │ 3. dispatchUsersUpdated()        │
    └──────────────┬───────────────────┘
                   │
         ┌─────────┴──────────┐
         ↓                    ↓
    ┌─────────────┐    ┌──────────────┐
    │ ModelView   │    │ TFVisorView  │
    │ renderAll   │    │ (lista        │
    │Purchases()  │    │ atualizada)  │
    │             │    │              │
    └─────────────┘    └──────────────┘

    [Volta ao fluxo de recomendação se modelo pronto]
```

---

## 📝 Resumo Executivo

| Elemento                | Especificação                                                                     |
| ----------------------- | --------------------------------------------------------------------------------- |
| **Tipo de Aprendizado** | Supervisionado                                                                    |
| **Tarefa Principal**    | Classificação: usuário vai comprar este produto?                                  |
| **Input**               | Vetor usuário (média de compras prévias) + vetor produto (features engenhariadas) |
| **Features**            | Preço normalizado, idade média, one-hot cores, one-hot categorias                 |
| **Pesos Relativos**     | Categoria 40%, Cor 30%, Preço 20%, Idade 10%                                      |
| **Arquitetura NN**      | 128 → 64 → 32 → 1 (ReLU + Sigmoid)                                                |
| **Otimizador**          | Adam (lr=0.01)                                                                    |
| **Loss**                | Binary Crossentropy                                                               |
| **Épocas**              | 100                                                                               |
| **Batch Size**          | 32                                                                                |
| **Persistência**        | sessionStorage (dados) + global \_model (modelo)                                  |
| **Threaded**            | Web Worker (não bloqueia UI)                                                      |
| **Visualização**        | TFVis em tempo real (loss + accuracy)                                             |
| **Saída**               | Ranking de produtos ordenados por probabilidade de compra                         |

---

## 🚀 Próximos Passos para Aprendizado

1. **Alterar pesos**: Mude WEIGHTS e veja como afeta recomendações
2. **Testar arquitetura**: Remova uma camada, adicione Dropout
3. **Feature engineering**: Adicione price bins, age ranges
4. **Métricas**: Implemente confusion matrix, ROC-AUC
5. **Retreino**: Adicione botão de retreino incremental
6. **Embedding**: Substitua one-hot por embeddings aprendidos
7. **Avaliação**: Implemente validação cruzada

---

**Criado**: Abril 2026  
**Disciplina**: Engenharia de Software com IA Aplicada  
**Módulo**: Fundamentos de IA e LLMs para Programadores
