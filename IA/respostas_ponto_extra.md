# Ponto Extra — Raciocínio Neuro-Simbólico Avançado com LTNtorch

**Disciplina:** Inteligência Artificial | **Professor:** Edjard Mota | **Julho de 2026**

Este documento responde às 15 perguntas de raciocínio avançado propostas para o ponto extra, explicando a lógica por trás de cada fórmula LTN, os desafios de implementação e como interpretar o resultado.

---

## Nível 1 — Raciocínio Numérico e de Cardinalidade

### Pergunta 1 — Contagem Exata

> **"Quantos cilindros existem na cena?"**

#### Fórmula LTN

$$Count_{Cylinder} = \sum_{x \in D} IsCylinder(x)$$

#### Raciocínio

No LTN padrão, os predicados retornam valores *fuzzy* em $[0, 1]$. Para uma cena com one-hot perfeito, `IsCylinder(x)` retorna exatamente **1.0** se o objeto for um cilindro e **0.0** caso contrário.

A **contagem suave** é portanto a soma direta desses valores:

```python
dataset_tensor = torch.tensor(dataset_np, dtype=torch.float32)
# Índice 7 = isCylinder no vetor de 11 atributos
count_cylinder = dataset_tensor[:, 7].sum().item()
print(f"Cilindros na cena: {count_cylinder:.0f}")
```

**Interpretação:** Como os dados são one-hot, a soma retorna um inteiro exato. Em cenários com ruído (atributos contínuos), a soma fornece uma estimativa probabilística ("2.7 cilindros" ≈ provavelmente 3).

**Desafio:** O LTN não tem nativo um operador `Count` — é preciso sair do framework lógico e usar Python/PyTorch para somar os valores de verdade retornados pelos predicados. A diferenciabilidade é preservada, então esta operação pode compor com a KB.

---

### Pergunta 2 — Comparação de Quantidades

> **"Existem mais quadrados vermelhos do que círculos azuis?"**

#### Fórmula LTN

$$\left(\sum_x IsSquare(x) \wedge IsRed(x)\right) > \left(\sum_y IsCircle(y) \wedge IsBlue(y)\right)$$

#### Raciocínio

Calculamos a **cardinalidade fuzzy** de cada conjunto e comparamos:

```python
# Cardinalidade suave de cada grupo
sq_red  = (dataset_tensor[:, 6] * dataset_tensor[:, 2]).sum()  # Quadrado ∧ Vermelho
ci_blue = (dataset_tensor[:, 5] * dataset_tensor[:, 4]).sum()  # Círculo ∧ Azul

resposta = (sq_red > ci_blue).item()
print(f"Quadrados vermelhos: {sq_red:.2f} | Círculos azuis: {ci_blue:.2f}")
print(f"Mais quadrados vermelhos? {resposta}")
```

**Interpretação:** A comparação `>` deve ser diferenciável para uso dentro da KB. Uma alternativa é usar `torch.sigmoid(sq_red - ci_blue)` como valor de verdade suave.

**Desafio:** A comparação binária (`>`) não é diferenciável. A solução NeSy é transformá-la em uma **função sigmoide da diferença**, que se aproxima de 1.0 quando o primeiro grupo é maior e de 0.0 quando é menor.

---

### Pergunta 3 — Unicidade

> **"Existe exatamente um objeto que é grande e verde?"**

#### Fórmula LTN

$$\exists x \left(IsLarge(x) \wedge IsGreen(x) \wedge \forall y \left((IsLarge(y) \wedge IsGreen(y)) \Rightarrow (x = y)\right)\right)$$

#### Raciocínio

A fórmula tem dois componentes:
1. **Existência:** Há pelo menos um objeto grande e verde.
2. **Unicidade:** Todo outro objeto que satisfaz as mesmas condições é *o mesmo* objeto.

**Igualdade no LTN:** Objetos são representados por vetores. Dois objetos são "iguais" se seus vetores são idênticos. Podemos definir:

$$Equal(x, y) = e^{-\gamma \cdot ||x - y||^2}$$

```python
Equal = ltn.Predicate(
    func=lambda x, y: torch.exp(-10.0 * torch.sum((x - y)**2, dim=-1))
)

x = ltn.Variable("x", dataset_tensor)
y = ltn.Variable("y", dataset_tensor)

unicidade = Exists(
    x,
    And(
        IsLarge(x), And(IsGreen(x),
        Forall(y, Implies(And(IsLarge(y), IsGreen(y)), Equal(x, y)))
    ))
)
print(f"Exatamente um grande e verde? satAgg = {unicidade.value.item():.4f}")
```

**Interpretação:** `satAgg` próximo de 1.0 confirma unicidade. Se houver dois objetos grandes e verdes, `Forall(y, ...)` terá satisfatibilidade baixa, pois o segundo objeto satisfaz as premissas mas não é igual ao primeiro.

---

### Pergunta 4 — Quantificação Numérica Específica

> **"Existem pelo menos 3 triângulos na cena?"**

#### Fórmula LTN

$$AtLeast3_{Triangle} = \sigma\!\left(\sum_{x} IsTriangle(x) - 3\right) \geq 0.5$$

onde $\sigma$ é a função sigmoide.

#### Raciocínio

Não existe um operador $\geq k$ nativo no LTN. A solução é construir um **wrapper diferenciável**:

```python
def at_least_k(predicate_values: torch.Tensor, k: int, steepness: float = 5.0) -> torch.Tensor:
    """
    Retorna valor próximo de 1.0 se a soma fuzzy >= k, e próximo de 0.0 caso contrário.
    'steepness' controla a suavidade da transição.
    """
    count = predicate_values.sum()
    return torch.sigmoid(steepness * (count - k))

triangle_values = dataset_tensor[:, 9]  # Índice 9 = IsTriangle
sat_at_least_3 = at_least_k(triangle_values, k=3)
print(f"Pelo menos 3 triângulos? {sat_at_least_3.item():.4f}")
```

**Interpretação:** Com 5 triângulos no dataset fixo, `sum = 5.0` e `sigmoid(5*(5-3)) = sigmoid(10) ≈ 1.0`, confirmando a pergunta.

---

## Nível 2 — Raciocínio Espacial Multi-Hop

### Pergunta 5 — Raciocínio de 2 Passos (2-Hop)

> **"Qual é a forma do objeto que está à esquerda do objeto que está atrás do círculo vermelho?"**

#### Fórmula LTN

$$\exists y, z \left(IsCircle(z) \wedge IsRed(z) \wedge Above(z, y) \wedge LeftOf(x, y) \wedge IsShape(x, ?)\right)$$

*(Usamos `Above` para representar "à frente" no eixo Y — o objeto com maior Y está "à frente" visualmente.)*

#### Raciocínio

Esta é uma cadeia de inferência com **dois saltos relacionais**:

```
Passo 1: Encontrar z = círculo vermelho
Passo 2: Encontrar y = objeto atrás de z (Below(y, z) ou Above(z, y))
Passo 3: Encontrar x = objeto à esquerda de y
Passo 4: Consultar a forma de x
```

```python
x = ltn.Variable("x", dataset_tensor)
y = ltn.Variable("y", dataset_tensor)
z = ltn.Variable("z", dataset_tensor)

# Para cada forma candidata, avalia a satisfatibilidade
for forma_nome, forma_pred in zip(
    ["Círculo","Quadrado","Cilindro","Cone","Triângulo"],
    [IsCircle, IsSquare, IsCylinder, IsCone, IsTriangle]
):
    sat = Exists(
        [x, y, z],
        And(IsCircle(z), And(IsRed(z),
        And(pred["Above"](z, y),
        And(pred["LeftOf"](x, y), forma_pred(x)))))
    )
    print(f"  x é {forma_nome}? satAgg = {sat.value.item():.4f}")
```

**Interpretação:** A forma com maior `satAgg` é a resposta mais provável. O gradiente propaga da consulta de forma, passando por `LeftOf` e `Above`, até o ancoragem no círculo vermelho.

**Desafio central:** O modelo precisa aprender que a relação `Above(z, y)` identifica corretamente o objeto intermediário `y`. Se a transitividade não estiver na KB, o modelo pode "pular" o elo intermediário incorretamente.

---

### Pergunta 6 — Filtragem Relacional Composta

> **"Existe algum objeto pequeno que esteja abaixo de um cilindro E à esquerda de um quadrado?"**

#### Fórmula LTN

$$\exists x \left(IsSmall(x) \wedge \exists y \left(IsCylinder(y) \wedge Below(x, y)\right) \wedge \exists z \left(IsSquare(z) \wedge LeftOf(x, z)\right)\right)$$

#### Raciocínio

Esta fórmula é implementada diretamente na Tarefa 4.1 do trabalho. O raciocínio segue três restrições simultâneas sobre o mesmo objeto `x`:

| Restrição | O que exige |
|---|---|
| `IsSmall(x)` | x deve ser de tamanho pequeno |
| `∃y (IsCylinder(y) ∧ Below(x,y))` | Deve existir um cilindro acima de x |
| `∃z (IsSquare(z) ∧ LeftOf(x,z))` | Deve existir um quadrado à direita de x |

```python
Formula_composta = Exists(
    x,
    And(IsSmall(x), And(
        Exists(y, And(IsCylinder(y), pred["Below"](x, y))),
        Exists(z, And(IsSquare(z),   pred["LeftOf"](x, z)))
    ))
)
```

**Interpretação:** `satAgg` alto (> 0.5) significa que existe tal objeto. `satAgg` baixo significa que nenhum objeto satisfaz todas as três condições simultaneamente. Esta é a fórmula mais representativa de raciocínio composicional no estilo CLEVR.

---

### Pergunta 7 — Relações Espaciais Relativas

> **"O objeto à esquerda do quadrado verde tem a mesma cor que o objeto à direita do círculo azul?"**

#### Fórmula LTN

$$\exists t_1, t_2, sq, ci \left(IsSquare(sq) \wedge IsGreen(sq) \wedge LeftOf(t_1, sq) \wedge IsCircle(ci) \wedge IsBlue(ci) \wedge RightOf(t_2, ci) \wedge SameColor(t_1, t_2)\right)$$

#### Raciocínio

O predicado `SameColor` compara os vetores de cor de dois objetos:

$$SameColor(x, y) = 1 - \frac{1}{3}\sum_{i \in \{R,G,B\}} |x_i - y_i|$$

```python
SameColor = ltn.Predicate(
    func=lambda x, y: 1.0 - (
        torch.abs(x[..., 2] - y[..., 2]) +
        torch.abs(x[..., 3] - y[..., 3]) +
        torch.abs(x[..., 4] - y[..., 4])
    ) / 3.0
)

t1 = ltn.Variable("t1", dataset_tensor)
t2 = ltn.Variable("t2", dataset_tensor)
sq = ltn.Variable("sq", dataset_tensor)
ci = ltn.Variable("ci", dataset_tensor)

sat_mesma_cor = Exists(
    [t1, t2, sq, ci],
    And(IsSquare(sq), And(IsGreen(sq), And(pred["LeftOf"](t1, sq),
    And(IsCircle(ci), And(IsBlue(ci),  And(pred["RightOf"](t2, ci),
    SameColor(t1, t2)))))))
)
print(f"Mesma cor? satAgg = {sat_mesma_cor.value.item():.4f}")
```

**Interpretação:** O modelo precisa identificar dois objetos via relações indiretas e depois comparar atributos. Isso testa a capacidade de **composição de conhecimento** — unir raciocínio espacial com raciocínio de atributos.

---

### Pergunta 8 — Dedução de Posição Absoluta

> **"O cone vermelho está entre dois outros objetos quaisquer?"**

#### Fórmula LTN

$$\exists x, y, z \left(IsCone(x) \wedge IsRed(x) \wedge LeftOf(y, x) \wedge RightOf(z, x)\right)$$

#### Raciocínio

"Estar entre" horizontalmente significa que há ao menos um objeto à esquerda e um à direita. Usamos os predicados `LeftOf` e `RightOf` já treinados:

```python
x = ltn.Variable("x", dataset_tensor)
y = ltn.Variable("y", dataset_tensor)
z = ltn.Variable("z", dataset_tensor)

cone_vermelho_no_meio = Exists(
    [x, y, z],
    And(IsCone(x), And(IsRed(x),
    And(pred["LeftOf"](y, x), pred["RightOf"](z, x))))
)
print(f"Cone vermelho no meio? satAgg = {cone_vermelho_no_meio.value.item():.4f}")
```

**Interpretação:** Esta é a consulta `InBetween` aplicada especificamente a cones vermelhos. Se o cone vermelho estiver em uma posição X central (não na extremidade), o resultado deve ser próximo de 1.0.

---

## Nível 3 — Lógica de Conjuntos e Exclusão

### Pergunta 9 — Diferença de Conjuntos

> **"Todo cilindro não vermelho está acima de algum cone?"**

#### Fórmula LTN

$$\forall x \left(\left(IsCylinder(x) \wedge \neg IsRed(x)\right) \Rightarrow \exists y \left(IsCone(y) \wedge Above(x, y)\right)\right)$$

#### Raciocínio

A negação `¬IsRed(x)` é implementada como `Not(IsRed(x))`, que no LTN com `NotStandard` equivale a `1 - IsRed(x)`. A implicação usa o operador de Reichenbach:

$$A \Rightarrow B = 1 - A + A \cdot B$$

```python
x = ltn.Variable("x", dataset_tensor)
y = ltn.Variable("y", dataset_tensor)

cilindro_nao_vermelho_acima_cone = Forall(
    x,
    Implies(
        And(IsCylinder(x), Not(IsRed(x))),
        Exists(y, And(IsCone(y), pred["Above"](x, y)))
    )
)
print(f"satAgg = {cilindro_nao_vermelho_acima_cone.value.item():.4f}")
```

**Interpretação:** Se todos os cilindros verdes e azuis estiverem acima de pelo menos um cone, o resultado será próximo de 1.0. O operador de Reichenbach garante que, quando a premissa (`IsCylinder ∧ ¬IsRed`) é falsa (objeto não é cilindro não-vermelho), a implicação é **vacuamente verdadeira** (valor 1.0), evitando penalização incorreta.

---

### Pergunta 10 — Verificação de Conjuntos Disjuntos

> **"Existe algum objeto que tenha a forma do Objeto A e a cor do Objeto B?"**

#### Fórmula LTN

$$\exists x \left(SameShape(x, A) \wedge SameColor(x, B)\right)$$

onde $A$ e $B$ são constantes (objetos específicos da cena).

#### Raciocínio

Objetos específicos são representados como **constantes LTN** (tensores fixos):

```python
# Suponha que A = objeto 3 (cone vermelho) e B = objeto 10 (cilindro verde)
objeto_A = ltn.Constant(dataset_tensor[3])  # cone vermelho
objeto_B = ltn.Constant(dataset_tensor[10]) # cilindro verde

SameShape = ltn.Predicate(
    func=lambda x, ref: 1.0 - torch.sum(
        torch.abs(x[..., 5:10] - ref[5:10]), dim=-1
    ) / 2.0
)

x = ltn.Variable("x", dataset_tensor)

intersecao = Exists(
    x,
    And(SameShape(x, objeto_A), SameColor(x, objeto_B))
)
print(f"Existe tal objeto? satAgg = {intersecao.value.item():.4f}")
```

**Interpretação:** `satAgg` alto significa que existe um objeto com a forma de A (cone) e a cor de B (verde) — um cone verde. O modelo infere isso sem treino específico, apenas usando os predicados já definidos.

---

### Pergunta 11 — Identificação de Atributo Comum

> **"Qual característica todos os objetos à esquerda de algum triângulo têm em comum?"**

#### Fórmula LTN

Para cada atributo candidato $A \in \{IsRed, IsGreen, IsBlue, IsCircle, \ldots\}$:

$$sat_A = \text{SatAgg}\!\left[\forall x \left(\exists y \left(IsTriangle(y) \wedge LeftOf(x, y)\right) \Rightarrow Attribute_A(x)\right)\right]$$

O atributo com maior $sat_A$ é a resposta.

#### Raciocínio

```python
candidatos = {
    "Vermelho": IsRed,  "Verde": IsGreen,  "Azul": IsBlue,
    "Círculo": IsCircle, "Quadrado": IsSquare, "Cilindro": IsCylinder,
    "Cone": IsCone,      "Triângulo": IsTriangle,
    "Pequeno": IsSmall,  "Grande": IsLarge,
}

x = ltn.Variable("x", dataset_tensor)
y = ltn.Variable("y", dataset_tensor)

resultados = {}
for nome, pred_atrib in candidatos.items():
    sat = Forall(
        x,
        Implies(
            Exists(y, And(IsTriangle(y), pred["LeftOf"](x, y))),
            pred_atrib(x)
        )
    )
    resultados[nome] = sat.value.item()

melhor = max(resultados, key=resultados.get)
print(f"Atributo mais comum: {melhor} (satAgg = {resultados[melhor]:.4f})")
for nome, val in sorted(resultados.items(), key=lambda t: -t[1]):
    print(f"  {nome:<12}: {val:.4f}")
```

**Interpretação:** Este é um algoritmo de **busca por abduação** — em vez de responder uma pergunta sim/não, buscamos a hipótese mais bem suportada pelos dados. O atributo com maior satisfatibilidade é aquele que melhor descreve o grupo de objetos à esquerda de triângulos.

---

## Nível 4 — Raciocínio Condicional e Hipotético

### Pergunta 12 — Implicação Material

> **"É verdade que todo objeto grande é também vermelho?"**

#### Fórmula LTN

$$\forall x \left(IsLarge(x) \Rightarrow IsRed(x)\right)$$

#### Raciocínio

```python
x = ltn.Variable("x", dataset_tensor)

regra_grande_implica_vermelho = Forall(
    x,
    Implies(IsLarge(x), IsRed(x))
)
print(f"Todo grande é vermelho? satAgg = {regra_grande_implica_vermelho.value.item():.4f}")
```

**O caso da Vacuous Truth (Verdade Vacuosa):** Se não houver nenhum objeto grande na cena, o quantificador universal `∀x` agrega sobre um conjunto vazio. Com o `AggregPMeanError`, isso retorna 1.0 — a regra é **trivialmente verdadeira** porque não há contraexemplos.

**Implicação de Reichenbach:** $A \Rightarrow B = 1 - A + A \cdot B$. Quando $A = 0$ (objeto não é grande), a implicação vale 1.0. Somente quando $A = 1$ e $B = 0$ (grande mas não vermelho) a implicação vale 0.0.

**Interpretação:** Um `satAgg` alto com muitos objetos grandes sugere correlação real. Um `satAgg` alto com poucos ou nenhum objeto grande pode ser vacuosamente verdadeiro — deve-se verificar a contagem antes de aceitar a conclusão.

---

### Pergunta 13 — Validação de Regra Complexa

> **"Todo quadrado está à esquerda de algum círculo OU à direita de algum cilindro?"**

#### Fórmula LTN

$$\forall x \left(IsSquare(x) \Rightarrow \left(\exists y \left(IsCircle(y) \wedge LeftOf(x, y)\right) \vee \exists z \left(IsCylinder(z) \wedge RightOf(x, z)\right)\right)\right)$$

#### Raciocínio

```python
x = ltn.Variable("x", dataset_tensor)
y = ltn.Variable("y", dataset_tensor)
z = ltn.Variable("z", dataset_tensor)

regra_quadrado = Forall(
    x,
    Implies(
        IsSquare(x),
        Or(
            Exists(y, And(IsCircle(y),   pred["LeftOf"](x, y))),
            Exists(z, And(IsCylinder(z), pred["RightOf"](x, z)))
        )
    )
)
print(f"Regra satisfeita? satAgg = {regra_quadrado.value.item():.4f}")
```

**Interpretação:** O operador `Or` com `OrProbSum` ($A \vee B = A + B - A \cdot B$) permite que baste *uma* das condições ser verdadeira. Um quadrado que não tem nenhum círculo à direita pode ainda satisfazer a regra se tiver um cilindro à sua esquerda.

**Importância prática:** Este tipo de regra é comum em cenários reais — "um funcionário deve ter ou uma certificação A ou uma certificação B". O LTN permite codificar e verificar essas políticas diretamente sobre representações neurais.

---

## Nível 5 — Detecção de Anomalia ("Odd-One-Out")

### Pergunta 14 — Detecção de Objeto Único

> **"Qual objeto tem uma forma diferente de todos os outros?"**

#### Fórmula LTN

Para cada candidato $c$ (constante), avaliar:

$$Score(c) = \text{SatAgg}\!\left[\forall y \left(y \neq c \Rightarrow \neg SameShape(c, y)\right)\right]$$

O objeto com maior $Score$ é o outlier.

#### Raciocínio

```python
SameShape = ltn.Predicate(
    func=lambda a, b: 1.0 - torch.sum(torch.abs(a[..., 5:10] - b[..., 5:10]), dim=-1) / 2.0
)
NotEqual = ltn.Predicate(
    func=lambda a, b: 1.0 - torch.exp(-5.0 * torch.sum((a - b)**2, dim=-1))
)

y = ltn.Variable("y", dataset_tensor)
scores = []
for i in range(len(dataset_np)):
    c = ltn.Constant(dataset_tensor[i])
    score = Forall(
        y,
        Implies(NotEqual(c, y), Not(SameShape(c, y)))
    )
    scores.append((i, score.value.item()))

scores.sort(key=lambda t: -t[1])
print(f"Maior outlier: Objeto #{scores[0][0]} (satAgg = {scores[0][1]:.4f})")
for idx, val in scores[:5]:
    print(f"  Objeto #{idx}: {val:.4f}")
```

**Interpretação:** Em um dataset com 5 objetos de cada forma, nenhum será outlier. Mas se um único objeto de uma forma incomum existir, ele terá `satAgg` alto porque "todos os outros têm forma diferente dele". Este é um exemplo de **raciocínio abdutivo** — o modelo identifica o objeto mais *singular* na cena.

---

### Pergunta 15 — Quebra de Padrão Espacial

> **"Qual objeto 'quebra' o padrão de uma sequência ordenada da esquerda para a direita?"**

#### Fórmula LTN

Para cada candidato $c$, avaliar o custo de removê-lo da sequência:

$$Score(c) = 1 - \text{SatAgg}\!\left[\forall x, y \left(\left(LeftOf(x, c) \wedge LeftOf(c, y)\right) \Rightarrow LeftOf(x, y)\right)\right]$$

*(O objeto que mais viola a transitividade é o "intruso".)*

#### Raciocínio

Um padrão espacial ordenado satisfaz completamente a transitividade: se A está antes de B e B está antes de C, então A está antes de C. Um objeto intruso "corta" esta cadeia.

```python
x = ltn.Variable("x", dataset_tensor)
y = ltn.Variable("y", dataset_tensor)

scores_quebra = []
for i in range(len(dataset_np)):
    c = ltn.Constant(dataset_tensor[i])
    # Consistência transitiva passando por c
    consistencia = Forall(
        [x, y],
        Implies(
            And(pred["LeftOf"](x, c), pred["LeftOf"](c, y)),
            pred["LeftOf"](x, y)
        )
    )
    scores_quebra.append((i, 1.0 - consistencia.value.item()))

scores_quebra.sort(key=lambda t: -t[1])
print(f"Objeto que mais quebra o padrão: #{scores_quebra[0][0]}")
```

**Interpretação:** O **score de quebra** é o inverso da satisfatibilidade da transitividade local. O objeto com maior score é aquele cuja presença na cadeia mais viola a consistência. Isso equivale a encontrar o ponto onde a sequência espacial "não faz sentido".

---

## Resumo Comparativo

| # | Pergunta | Técnica LTN Central | Complexidade |
|---|---|---|---|
| 1 | Contagem de cilindros | Soma de valores fuzzy | ⭐ |
| 2 | Mais quadrados que círculos? | Sigmoid da diferença de contagens | ⭐⭐ |
| 3 | Exatamente um grande verde? | Existência + Unicidade + Igualdade | ⭐⭐⭐ |
| 4 | Pelo menos 3 triângulos? | Operador `AtLeast-k` diferenciável | ⭐⭐ |
| 5 | Forma do objeto 2-hop | Encadeamento de quantificadores | ⭐⭐⭐⭐ |
| 6 | Pequeno abaixo de cilindro E à esquerda de quadrado | Múltiplas restrições simultâneas | ⭐⭐⭐ |
| 7 | Mesma cor em referências cruzadas | SameColor + 2 âncoras relacionais | ⭐⭐⭐⭐ |
| 8 | Cone vermelho no meio? | InBetween com constante de cor | ⭐⭐ |
| 9 | Cilindros não-vermelhos acima de cones? | Negação + implicação + existencial aninhado | ⭐⭐⭐ |
| 10 | Objeto com forma de A e cor de B? | Constantes LTN + SameShape/SameColor | ⭐⭐ |
| 11 | Atributo comum à esquerda de triângulos | Busca abdutiva sobre candidatos | ⭐⭐⭐⭐ |
| 12 | Todo grande é vermelho? | Implicação + Vacuous Truth | ⭐⭐ |
| 13 | Todo quadrado: à esq. de círculo OU à dir. de cilindro? | ∀ + ∨ de existenciais | ⭐⭐⭐ |
| 14 | Objeto com forma diferente de todos | Busca por outlier + NotEqual + SameShape | ⭐⭐⭐⭐⭐ |
| 15 | Objeto que quebra padrão espacial | Violação de transitividade local | ⭐⭐⭐⭐⭐ |

---

## Considerações Finais sobre os Limites do LTN

### Onde o LTN se destaca
- **Raciocínio relacional:** Perguntas 5–8, 13 — encadeia relações espaciais com atributos de forma natural e diferenciável.
- **Implicações condicionais:** Perguntas 9, 12, 13 — o operador de Reichenbach lida corretamente com premissas falsas (vacuous truth).
- **Integração de conhecimento prévio:** Os axiomas da KB guiam o aprendizado sem exemplos rotulados de cada relação.

### Onde o LTN tem limitações
- **Contagem exata** (P1, P4): requer saída do framework lógico puro para operações numéricas Python/PyTorch.
- **Comparação numérica** (P2): a função `>` não é diferenciável nativamente — precisa de aproximação sigmoide.
- **Busca sobre candidatos** (P11, P14, P15): são loops Python sobre instâncias individuais, não fórmulas LTN puras.
- **Padrões espaciais complexos** (P15): LTN não modela "linhas" ou "sequências" — aproximamos via transitividade local.

---

*Referências: Badreddine et al. (2022). Logic Tensor Networks. Artificial Intelligence, 303. | Johnson et al. (2017). CLEVR. CVPR.*