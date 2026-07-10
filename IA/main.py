# ==============================================================================
# TRABALHO FINAL DE IA -- RACIOCÍNIO NEURO-SIMBÓLICO COM LOGIC TENSOR NETWORKS
# Professor: Edjard Mota | Julho de 2026
#
import matplotlib
matplotlib.use("Agg")   # backend sem janela -- não bloqueia o pipeline
#
# Estrutura do vetor de atributos (11 posições -- conforme instruções):
#   [0]      -> Posição X (normalizada, 0.0 a 1.0)
#   [1]      -> Posição Y (normalizada, 0.0 a 1.0)
#   [2..4]   -> Cor one-hot   (Vermelho, Verde, Azul)
#   [5..9]   -> Forma one-hot (Círculo, Quadrado, Cilindro, Cone, Triângulo)
#   [10]     -> Tamanho (0.0 = Pequeno, 1.0 = Grande)
#
# Dependências: torch, ltn (LTNtorch), numpy, matplotlib, sklearn
# ==============================================================================

import itertools
import statistics

import numpy as np
import torch
import torch.nn as nn
import ltn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import dataset as ds


# ==============================================================================
# SEÇÃO 1 -- CONECTIVOS E QUANTIFICADORES LTN
# ==============================================================================

Not     = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And     = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or      = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Forall  = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Exists  = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2),      quantifier="e")
SatAgg  = ltn.fuzzy_ops.SatAgg()


# ==============================================================================
# SEÇÃO 2 -- PREDICADOS UNÁRIOS (Cor, Forma, Tamanho)
#            Índices conforme o vetor de 11 atributos.
# ==============================================================================

# Cor (índices 2, 3, 4)
IsRed   = ltn.Predicate(func=lambda x: x[..., 2])
IsGreen = ltn.Predicate(func=lambda x: x[..., 3])
IsBlue  = ltn.Predicate(func=lambda x: x[..., 4])

_PREDICADOS_COR = [IsRed, IsGreen, IsBlue]

# Forma (índices 5, 6, 7, 8, 9)
IsCircle   = ltn.Predicate(func=lambda x: x[..., 5])
IsSquare   = ltn.Predicate(func=lambda x: x[..., 6])
IsCylinder = ltn.Predicate(func=lambda x: x[..., 7])
IsCone     = ltn.Predicate(func=lambda x: x[..., 8])
IsTriangle = ltn.Predicate(func=lambda x: x[..., 9])

_PREDICADOS_FORMA = [IsCircle, IsSquare, IsCylinder, IsCone, IsTriangle]

# Tamanho (índice 10)
IsSmall = ltn.Predicate(func=lambda x: 1.0 - x[..., 10])
IsLarge = ltn.Predicate(func=lambda x: x[..., 10])

# SameSize (predicado binário diferenciável)
SameSize = ltn.Predicate(
    func=lambda x, y: 1.0 - torch.abs(x[..., 10] - y[..., 10])
)


# ==============================================================================
# SEÇÃO 3 -- PREDICADOS ESPACIAIS (Redes Neurais Treináveis)
# ==============================================================================

class RelacaoEspacialNet(nn.Module):
    """MLP que aprende relações espaciais binárias (LeftOf, RightOf, Above, Below)."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        entrada = torch.cat([x[..., 0:2], y[..., 0:2]], dim=-1)
        return torch.sigmoid(self.mlp(entrada)).squeeze(-1)


class ProximidadeNet(nn.Module):
    """Kernel Gaussiano treinável para CloseTo: exp(-β * ||x_pos - y_pos||²)."""

    def __init__(self):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(1.5))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.sum((x[..., 0:2] - y[..., 0:2]) ** 2, dim=-1)
        beta = torch.nn.functional.softplus(self.log_beta)
        return torch.exp(-beta * dist_sq)


def construir_predicados_espaciais() -> dict:
    return {
        "LeftOf":  ltn.Predicate(model=RelacaoEspacialNet()),
        "RightOf": ltn.Predicate(model=RelacaoEspacialNet()),
        "Above":   ltn.Predicate(model=RelacaoEspacialNet()),
        "Below":   ltn.Predicate(model=RelacaoEspacialNet()),
        "CloseTo": ltn.Predicate(model=ProximidadeNet()),
    }


# ==============================================================================
# SEÇÃO 4 -- BASE DE CONHECIMENTO (KB) E AXIOMAS
# ==============================================================================

def construir_axiomas_base(dataset_tensor: torch.Tensor, pred: dict) -> list:
    """
    Retorna a lista completa de axiomas LTN (Tarefa 1, 2 e 3) para treinamento.
    """
    x = ltn.Variable("x", dataset_tensor)
    y = ltn.Variable("y", dataset_tensor)
    z = ltn.Variable("z", dataset_tensor)

    axiomas = []

    # ------------------------------------------------------------------
    # TAREFA 1 -- Taxonomia de Cores e Formas
    # ------------------------------------------------------------------

    # 1a. Unicidade de cor: ¬(isA(x) ∧ isB(x))
    for pa, pb in itertools.combinations(_PREDICADOS_COR, 2):
        axiomas.append(Forall(x, Not(And(pa(x), pb(x)))))

    # 1b. Unicidade de forma: ¬(isA(x) ∧ isB(x))
    for pa, pb in itertools.combinations(_PREDICADOS_FORMA, 2):
        axiomas.append(Forall(x, Not(And(pa(x), pb(x)))))

    # 1c. Cobertura de cor: ∀x (isRed ∨ isGreen ∨ isBlue)
    axiomas.append(Forall(x, Or(IsRed(x), Or(IsGreen(x), IsBlue(x)))))

    # 1d. Cobertura de forma: ∀x (isCircle ∨ isSquare ∨ ... ∨ isTriangle)
    cobertura_forma = IsCircle(x)
    for pred_f in [IsSquare, IsCylinder, IsCone, IsTriangle]:
        cobertura_forma = Or(cobertura_forma, pred_f(x))
    axiomas.append(Forall(x, cobertura_forma))

    # ------------------------------------------------------------------
    # TAREFA 2 -- Raciocínio Horizontal (LeftOf / RightOf)
    # ------------------------------------------------------------------

    # 2a. Irreflexividade: ¬LeftOf(x,x) e ¬RightOf(x,x)
    axiomas.append(Forall(x, Not(pred["LeftOf"](x, x))))
    axiomas.append(Forall(x, Not(pred["RightOf"](x, x))))

    # 2b. Assimetria: LeftOf(x,y) ⇒ ¬LeftOf(y,x)
    axiomas.append(Forall([x, y], Implies(pred["LeftOf"](x, y), Not(pred["LeftOf"](y, x)))))

    # 2c. Inverso (bicondicional): LeftOf(x,y) ⟺ RightOf(y,x)
    axiomas.append(Forall([x, y], Implies(pred["LeftOf"](x, y),  pred["RightOf"](y, x))))
    axiomas.append(Forall([x, y], Implies(pred["RightOf"](x, y), pred["LeftOf"](y, x))))

    # 2d. Transitividade: LeftOf(x,y) ∧ LeftOf(y,z) ⇒ LeftOf(x,z)
    axiomas.append(Forall(
        [x, y, z],
        Implies(And(pred["LeftOf"](x, y), pred["LeftOf"](y, z)), pred["LeftOf"](x, z))
    ))

    # ------------------------------------------------------------------
    # TAREFA 3 -- Raciocínio Vertical (Above / Below)
    # ------------------------------------------------------------------

    # 3a. Irreflexividade
    axiomas.append(Forall(x, Not(pred["Above"](x, x))))
    axiomas.append(Forall(x, Not(pred["Below"](x, x))))

    # 3b. Inverso: Below(x,y) ⟺ Above(y,x)
    axiomas.append(Forall([x, y], Implies(pred["Below"](x, y), pred["Above"](y, x))))
    axiomas.append(Forall([x, y], Implies(pred["Above"](x, y), pred["Below"](y, x))))

    # 3c. Transitividade: Below(x,y) ∧ Below(y,z) ⇒ Below(x,z)
    axiomas.append(Forall(
        [x, y, z],
        Implies(And(pred["Below"](x, y), pred["Below"](y, z)), pred["Below"](x, z))
    ))

    return axiomas


# ==============================================================================
# SEÇÃO 5 -- LOOP DE TREINAMENTO
# ==============================================================================

def treinar_base_de_conhecimento(
    dataset_np: np.ndarray,
    pred: dict,
    epochs: int = 500,
    lr: float = 0.01,
    log_interval: int = 100,
    verbose: bool = True,
) -> float:
    dataset_tensor = torch.tensor(dataset_np, dtype=torch.float32)

    parametros = (
        list(pred["LeftOf"].parameters())
        + list(pred["RightOf"].parameters())
        + list(pred["Above"].parameters())
        + list(pred["Below"].parameters())
        + list(pred["CloseTo"].parameters())
    )
    optimizer = torch.optim.Adam(parametros, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    sat_final = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        axiomas  = construir_axiomas_base(dataset_tensor, pred)
        sat_level = SatAgg(*axiomas)
        loss = 1.0 - sat_level
        loss.backward()
        optimizer.step()
        scheduler.step()

        sat_final = sat_level.item()
        if verbose and (epoch + 1) % log_interval == 0:
            print(
                f"    Época {epoch + 1:04d}/{epochs} | "
                f"Satisfação KB: {sat_final:.4f} | "
                f"Loss: {loss.item():.4f}"
            )

    return sat_final


# ==============================================================================
# SEÇÃO 6 -- FÓRMULAS COMPLEXAS (Tarefas 2 e 4)
# ==============================================================================

def in_between(x, y, z, pred: dict):
    """
    inBetween(x, y, z): x está entre y e z horizontalmente.
    Fórmula: (leftOf(y,x) ∧ rightOf(z,x)) ∨ (leftOf(z,x) ∧ rightOf(y,x))
    """
    caso_1 = And(pred["LeftOf"](y, x), pred["RightOf"](z, x))
    caso_2 = And(pred["LeftOf"](z, x), pred["RightOf"](y, x))
    return Or(caso_1, caso_2)


def can_stack(x, y) -> object:
    """
    canStack(x, y): x pode ser empilhado sobre y se y não for cone nem triângulo
    e ambos tiverem o mesmo tamanho (condição de equilíbrio estável).
    """
    base_estavel = And(Not(IsCone(y)), Not(IsTriangle(y)))
    return And(base_estavel, SameSize(x, y))


def avaliar_formulas_complexas(dataset_np: np.ndarray, pred: dict) -> dict:
    """
    Avalia as fórmulas das Tarefas 2 e 4 após o treinamento.
    Retorna um dicionário com a satisfatibilidade de cada fórmula.
    """
    T = torch.tensor(dataset_np, dtype=torch.float32)
    x1 = ltn.Variable("x1", T)
    y1 = ltn.Variable("y1", T)
    z1 = ltn.Variable("z1", T)

    # ------------------------------------------------------------------
    # TAREFA 2 -- Consultas de Raciocínio Espacial
    # ------------------------------------------------------------------

    # lastOnTheLeft: ∃x ∀y leftOf(x, y)
    last_left = Exists(x1, Forall(y1, pred["LeftOf"](x1, y1)))

    # lastOnTheRight: ∃x ∀y rightOf(x, y)
    last_right = Exists(x1, Forall(y1, pred["RightOf"](x1, y1)))

    x2 = ltn.Variable("x2", T)
    y2 = ltn.Variable("y2", T)
    z2 = ltn.Variable("z2", T)

    # inBetween -- avaliação existencial: ∃x,y,z inBetween(x,y,z)
    sat_inbetween = Exists([x2, y2, z2], in_between(x2, y2, z2, pred))

    # ------------------------------------------------------------------
    # TAREFA 3.3 -- canStack
    # ------------------------------------------------------------------
    x3 = ltn.Variable("x3", T)
    y3 = ltn.Variable("y3", T)

    # canStack: ∃x,y canStack(x, y)
    sat_canstack = Exists([x3, y3], can_stack(x3, y3))

    # ------------------------------------------------------------------
    # TAREFA 4 -- Raciocínio Composto
    # ------------------------------------------------------------------
    x4 = ltn.Variable("x4", T)
    y4 = ltn.Variable("y4", T)
    z4 = ltn.Variable("z4", T)

    # Fórmula 4.1: ∃x (isSmall(x) ∧ ∃y (isCylinder(y) ∧ Below(x,y)) ∧ ∃z (isSquare(z) ∧ LeftOf(x,z)))
    Formula_4_1 = Exists(
        x4,
        And(
            IsSmall(x4),
            And(
                Exists(y4, And(IsCylinder(y4), pred["Below"](x4, y4))),
                Exists(z4, And(IsSquare(z4),   pred["LeftOf"](x4, z4))),
            ),
        ),
    )

    x5 = ltn.Variable("x5", T)
    y5 = ltn.Variable("y5", T)
    z5 = ltn.Variable("z5", T)

    # Fórmula 4.2: ∃x,y,z (isCone(x) ∧ isGreen(x) ∧ inBetween(x,y,z))
    Formula_4_2 = Exists(
        [x5, y5, z5],
        And(IsCone(x5), And(IsGreen(x5), in_between(x5, y5, z5, pred))),
    )

    x6 = ltn.Variable("x6", T)
    y6 = ltn.Variable("y6", T)

    # Fórmula 4.3: ∀x,y ((isTriangle(x) ∧ isTriangle(y) ∧ closeTo(x,y)) ⇒ sameSize(x,y))
    Formula_4_3 = Forall(
        [x6, y6],
        Implies(
            And(IsTriangle(x6), And(IsTriangle(y6), pred["CloseTo"](x6, y6))),
            SameSize(x6, y6),
        ),
    )

    return {
        # Tarefa 2
        "LastOnTheLeft":  last_left.value.item(),
        "LastOnTheRight": last_right.value.item(),
        "InBetween":      sat_inbetween.value.item(),
        # Tarefa 3.3
        "CanStack":       sat_canstack.value.item(),
        # Tarefa 4
        "Formula_4_1":    Formula_4_1.value.item(),
        "Formula_4_2":    Formula_4_2.value.item(),
        "Formula_4_3":    Formula_4_3.value.item(),
    }


# ==============================================================================
# SEÇÃO 7 -- MÉTRICAS DE CLASSIFICAÇÃO BINÁRIA
# ==============================================================================

def _ground_truth_leftof(dataset_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gera pares (i,j) e rótulos binários para LeftOf (x_i < x_j)."""
    n = len(dataset_np)
    pares_x, pares_y, rotulos = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pares_x.append(dataset_np[i])
            pares_y.append(dataset_np[j])
            rotulos.append(1 if dataset_np[i, 0] < dataset_np[j, 0] else 0)
    return np.array(pares_x, dtype=np.float32), np.array(pares_y, dtype=np.float32), np.array(rotulos)


def _ground_truth_below(dataset_np: np.ndarray) -> tuple:
    """Rótulos binários para Below (y_i < y_j)."""
    n = len(dataset_np)
    pares_x, pares_y, rotulos = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pares_x.append(dataset_np[i])
            pares_y.append(dataset_np[j])
            rotulos.append(1 if dataset_np[i, 1] < dataset_np[j, 1] else 0)
    return np.array(pares_x, dtype=np.float32), np.array(pares_y, dtype=np.float32), np.array(rotulos)


def _ground_truth_closeto(dataset_np: np.ndarray, limiar: float = 0.3) -> tuple:
    """Rótulos binários para CloseTo (distância euclidiana < limiar)."""
    n = len(dataset_np)
    pares_x, pares_y, rotulos = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(dataset_np[i, 0:2] - dataset_np[j, 0:2])
            pares_x.append(dataset_np[i])
            pares_y.append(dataset_np[j])
            rotulos.append(1 if dist < limiar else 0)
    return np.array(pares_x, dtype=np.float32), np.array(pares_y, dtype=np.float32), np.array(rotulos)


def _calcular_metricas(y_true: np.ndarray, y_pred_fuzzy: np.ndarray, limiar: float = 0.5) -> dict:
    y_pred = (y_pred_fuzzy >= limiar).astype(int)
    # Proteção contra classes ausentes
    labels = [0, 1] if len(np.unique(y_true)) > 1 else [y_true[0]]
    return {
        "Acuracia":  round(accuracy_score(y_true, y_pred), 4),
        "Precisao":  round(precision_score(y_true, y_pred, zero_division=0, average="binary"), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0, average="binary"), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0, average="binary"), 4),
    }


def calcular_metricas_classificacao(dataset_np: np.ndarray, pred: dict) -> dict:
    """
    Calcula Acurácia, Precisão, Recall e F1-Score para os predicados
    LeftOf, Below e CloseTo comparando os modelos neurais com as
    verdades-terreno geométricas.
    """
    metricas = {}

    with torch.no_grad():
        for nome, gt_fn, pred_key in [
            ("LeftOf",  _ground_truth_leftof,  "LeftOf"),
            ("Below",   _ground_truth_below,   "Below"),
            ("CloseTo", _ground_truth_closeto, "CloseTo"),
        ]:
            px, py, rotulos = gt_fn(dataset_np)
            tx = torch.tensor(px, dtype=torch.float32)
            ty = torch.tensor(py, dtype=torch.float32)
            saidas = pred[pred_key].model(tx, ty).numpy()
            metricas[nome] = _calcular_metricas(rotulos, saidas)

    return metricas


# ==============================================================================
# SEÇÃO 8 -- PIPELINE COMPLETO
# ==============================================================================

def executar_pipeline_completo(
    dataset_np: np.ndarray,
    epochs: int = 500,
    lr: float = 0.01,
    verbose: bool = True,
) -> dict:
    predicados = construir_predicados_espaciais()
    sat_kb = treinar_base_de_conhecimento(
        dataset_np, predicados, epochs=epochs, lr=lr, verbose=verbose
    )
    formulas = avaliar_formulas_complexas(dataset_np, predicados)
    metricas = calcular_metricas_classificacao(dataset_np, predicados)

    return {
        "SatLevel_KB": sat_kb,
        **formulas,
        "Metricas": metricas,
    }


def validar_robustez_multi_dataset(
    num_execucoes: int = 5,
    num_objetos: int = 25,
    epochs: int = 500,
    lr: float = 0.01,
) -> list:
    resultados = []
    for rodada in range(1, num_execucoes + 1):
        print(f"\n{'-'*60}")
        print(f" Rodada {rodada}/{num_execucoes} -- dataset aleatório independente")
        print(f"{'-'*60}")
        dados = ds.gerar_dataset_aleatorio(num_objetos=num_objetos)
        resultado = executar_pipeline_completo(dados, epochs=epochs, lr=lr, verbose=False)
        m = resultado["Metricas"]
        print(
            f"  KB={resultado['SatLevel_KB']:.4f} | "
            f"F4.1={resultado['Formula_4_1']:.4f} | "
            f"F4.2={resultado['Formula_4_2']:.4f} | "
            f"F4.3={resultado['Formula_4_3']:.4f}"
        )
        print(
            f"  LeftOf  -> Ac={m['LeftOf']['Acuracia']:.3f} "
            f"Pr={m['LeftOf']['Precisao']:.3f} "
            f"Re={m['LeftOf']['Recall']:.3f} "
            f"F1={m['LeftOf']['F1']:.3f}"
        )
        print(
            f"  Below  -> Ac={m['Below']['Acuracia']:.3f} "
            f"Pr={m['Below']['Precisao']:.3f} "
            f"Re={m['Below']['Recall']:.3f} "
            f"F1={m['Below']['F1']:.3f}"
        )
        print(
            f"  CloseTo -> Ac={m['CloseTo']['Acuracia']:.3f} "
            f"Pr={m['CloseTo']['Precisao']:.3f} "
            f"Re={m['CloseTo']['Recall']:.3f} "
            f"F1={m['CloseTo']['F1']:.3f}"
        )
        resultados.append(resultado)
    return resultados


# ==============================================================================
# SEÇÃO 9 -- RELATÓRIO FINAL
# ==============================================================================

def imprimir_relatorio_final(
    resultados_validacao: list,
    resultado_fixo: dict,
) -> None:

    SEP  = "=" * 78
    sep2 = "-" * 78

    print(f"\n{SEP}")
    print(" RELATÓRIO FINAL -- TRABALHO DE IA NEURO-SIMBÓLICA (LTNtorch)".center(78))
    print(" Professor Edjard Mota -- Julho de 2026".center(78))
    print(SEP)

    # -- CENÁRIO FIXO --------------------------------------------------
    print("\n[1] CENÁRIO FIXO DE CALIBRAÇÃO")
    print(sep2)
    f = resultado_fixo
    print(f"  Satisfação global da KB            : {f['SatLevel_KB']:.4f}")
    print(f"  Tarefa 2 -- lastOnTheLeft           : {f['LastOnTheLeft']:.4f}")
    print(f"  Tarefa 2 -- lastOnTheRight          : {f['LastOnTheRight']:.4f}")
    print(f"  Tarefa 2 -- inBetween (existencial) : {f['InBetween']:.4f}")
    print(f"  Tarefa 3.3 -- canStack (existencial): {f['CanStack']:.4f}")
    print(f"  Fórmula 4.1 (Filtragem Composta)   : {f['Formula_4_1']:.4f}")
    print(f"  Fórmula 4.2 (Posição Absoluta)     : {f['Formula_4_2']:.4f}")
    print(f"  Fórmula 4.3 (Prox. Triângulos)     : {f['Formula_4_3']:.4f}")

    m = f["Metricas"]
    print(f"\n  Métricas de Classificação -- Cenário Fixo:")
    print(f"  {'Predicado':<12} {'Acurácia':>10} {'Precisão':>10} {'Recall':>10} {'F1-Score':>10}")
    print(f"  {'-'*54}")
    for pred_nome in ["LeftOf", "Below", "CloseTo"]:
        mv = m[pred_nome]
        print(
            f"  {pred_nome:<12} "
            f"{mv['Acuracia']:>10.4f} "
            f"{mv['Precisao']:>10.4f} "
            f"{mv['Recall']:>10.4f} "
            f"{mv['F1']:>10.4f}"
        )

    # -- VALIDAÇÃO 5 DATASETS -----------------------------------------
    print(f"\n[2] VALIDAÇÃO DE ROBUSTEZ -- 5 DATASETS ALEATÓRIOS INDEPENDENTES")
    print(sep2)

    chaves_sat = ["SatLevel_KB", "Formula_4_1", "Formula_4_2", "Formula_4_3"]
    header = f"{'Rd':<4}{'KB':>8}{'F4.1':>8}{'F4.2':>8}{'F4.3':>8}"
    for pred_nome in ["LeftOf", "Below", "CloseTo"]:
        header += f"  {pred_nome+':Ac':>10}{pred_nome+':F1':>8}"
    print(f"  {header}")
    print(f"  {'-'*76}")

    for i, res in enumerate(resultados_validacao, 1):
        linha = (
            f"  {i:<4}"
            f"{res['SatLevel_KB']:>8.4f}"
            f"{res['Formula_4_1']:>8.4f}"
            f"{res['Formula_4_2']:>8.4f}"
            f"{res['Formula_4_3']:>8.4f}"
        )
        for pred_nome in ["LeftOf", "Below", "CloseTo"]:
            mv = res["Metricas"][pred_nome]
            linha += f"  {mv['Acuracia']:>10.4f}{mv['F1']:>8.4f}"
        print(linha)

    # -- MÉDIAS E DESVIOS ---------------------------------------------
    print(f"\n[3] MÉTRICAS CONSOLIDADAS (Média ± Desvio-Padrão -- 5 Execuções)")
    print(sep2)

    for chave in chaves_sat:
        vals = [r[chave] for r in resultados_validacao]
        mu  = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {chave:<22}: {mu:.4f} ± {std:.4f}")

    print()
    for pred_nome in ["LeftOf", "Below", "CloseTo"]:
        for metrica in ["Acuracia", "Precisao", "Recall", "F1"]:
            vals = [r["Metricas"][pred_nome][metrica] for r in resultados_validacao]
            mu  = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {pred_nome+'.'+metrica:<26}: {mu:.4f} ± {std:.4f}")

    print(f"\n{SEP}\n")


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 78)
    print(" PIPELINE NEURO-SIMBÓLICO -- LOGIC TENSOR NETWORKS SOBRE CLEVR SIMPLIFICADO")
    print("=" * 78)

    # Etapa A: Plota o cenário fixo (requisito visual da Tarefa 1.1)
    print("\n[Etapa A] Gerando e plotando o cenário fixo...")
    dataset_fixo = ds.gerar_dataset_fixo()
    ds.plotar_cenario(dataset_fixo, titulo="Cenário Fixo de Calibração (25 objetos, 5 formas)")
    print("  -> Figura salva como 'cenario_clevr.png' (modo não-interativo)")

    # Etapa B: Treina sobre o dataset fixo
    print("\n[Etapa B] Treinando a KB sobre o dataset fixo...")
    resultado_fixo = executar_pipeline_completo(
        dataset_fixo, epochs=500, lr=0.01, verbose=True
    )

    # Etapa C: Validação em 5 datasets aleatórios
    print("\n[Etapa C] Validação de robustez -- 5 datasets aleatórios independentes...")
    resultados_validacao = validar_robustez_multi_dataset(
        num_execucoes=5, num_objetos=25, epochs=500, lr=0.01
    )

    # Etapa D: Relatório final
    print("\n[Etapa D] Gerando relatório final...")
    imprimir_relatorio_final(resultados_validacao, resultado_fixo)