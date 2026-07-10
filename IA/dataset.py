import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sem janela
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import matplotlib.patches as mpatches

# ==============================================================================
# DATASET — ESTRUTURA CORRETA CONFORME AS INSTRUÇÕES DO TRABALHO
# ==============================================================================
#
# Vetor do Objeto (Tamanho: 11)
#   [0, 1]      -> Posição x, y  (normalizadas, 0.0 a 1.0)
#   [2, 3, 4]   -> Cor one-hot   (Vermelho, Verde, Azul)
#   [5, 6, 7, 8, 9] -> Forma one-hot (Círculo, Quadrado, Cilindro, Cone, Triângulo)
#   [10]         -> Tamanho (Pequeno = 0.0, Grande = 1.0)
#
# ==============================================================================

CORES = {
    "vermelho": [1.0, 0.0, 0.0],
    "verde":    [0.0, 1.0, 0.0],
    "azul":     [0.0, 0.0, 1.0],
}

FORMAS = {
    "circulo":   [1.0, 0.0, 0.0, 0.0, 0.0],
    "quadrado":  [0.0, 1.0, 0.0, 0.0, 0.0],
    "cilindro":  [0.0, 0.0, 1.0, 0.0, 0.0],
    "cone":      [0.0, 0.0, 0.0, 1.0, 0.0],
    "triangulo": [0.0, 0.0, 0.0, 0.0, 1.0],
}

TAMANHOS = {
    "pequeno": 0.0,
    "grande":  1.0,
}

# Índices no vetor para fácil referência
IDX_X, IDX_Y = 0, 1
IDX_COR_INICIO, IDX_COR_FIM = 2, 5        # [2, 3, 4]
IDX_FORMA_INICIO, IDX_FORMA_FIM = 5, 10   # [5, 6, 7, 8, 9]
IDX_TAMANHO = 10


def criar_objeto_manual(x: float, y: float, cor: str, forma: str, tamanho: str) -> np.ndarray:
    """Monta o vetor de características de tamanho 11 para um único objeto."""
    vetor = [x, y] + CORES[cor] + FORMAS[forma] + [TAMANHOS[tamanho]]
    return np.array(vetor, dtype=np.float32)


def gerar_dataset_fixo() -> np.ndarray:
    """
    Gera um cenário controlado contendo ao menos dois objetos de cada forma
    (Círculo, Quadrado, Cilindro, Cone, Triângulo), totalizando 25 objetos,
    com cores, tamanhos e posições variadas para cobrir o espaço de estados.
    """
    objetos = []

    # --- CÍRCULOS (5 objetos) ---
    objetos.append(criar_objeto_manual(0.10, 0.10, "vermelho",  "circulo",  "grande"))
    objetos.append(criar_objeto_manual(0.85, 0.80, "azul",      "circulo",  "pequeno"))
    objetos.append(criar_objeto_manual(0.50, 0.50, "verde",     "circulo",  "grande"))
    objetos.append(criar_objeto_manual(0.30, 0.70, "vermelho",  "circulo",  "pequeno"))
    objetos.append(criar_objeto_manual(0.70, 0.30, "azul",      "circulo",  "grande"))

    # --- QUADRADOS (5 objetos) ---
    objetos.append(criar_objeto_manual(0.20, 0.80, "azul",      "quadrado", "pequeno"))
    objetos.append(criar_objeto_manual(0.80, 0.20, "verde",     "quadrado", "grande"))
    objetos.append(criar_objeto_manual(0.05, 0.50, "vermelho",  "quadrado", "grande"))
    objetos.append(criar_objeto_manual(0.95, 0.50, "azul",      "quadrado", "pequeno"))
    objetos.append(criar_objeto_manual(0.50, 0.05, "verde",     "quadrado", "grande"))

    # --- CILINDROS (5 objetos) ---
    objetos.append(criar_objeto_manual(0.15, 0.40, "verde",     "cilindro", "grande"))
    objetos.append(criar_objeto_manual(0.60, 0.90, "vermelho",  "cilindro", "pequeno"))
    objetos.append(criar_objeto_manual(0.40, 0.60, "azul",      "cilindro", "grande"))
    objetos.append(criar_objeto_manual(0.85, 0.55, "verde",     "cilindro", "pequeno"))
    objetos.append(criar_objeto_manual(0.25, 0.25, "vermelho",  "cilindro", "grande"))

    # --- CONES (5 objetos) ---
    objetos.append(criar_objeto_manual(0.90, 0.10, "vermelho",  "cone",     "pequeno"))
    objetos.append(criar_objeto_manual(0.10, 0.90, "azul",      "cone",     "grande"))
    objetos.append(criar_objeto_manual(0.55, 0.35, "verde",     "cone",     "pequeno"))
    objetos.append(criar_objeto_manual(0.35, 0.55, "vermelho",  "cone",     "grande"))
    objetos.append(criar_objeto_manual(0.70, 0.75, "azul",      "cone",     "pequeno"))

    # --- TRIÂNGULOS (5 objetos) ---
    objetos.append(criar_objeto_manual(0.45, 0.15, "azul",      "triangulo","pequeno"))
    objetos.append(criar_objeto_manual(0.75, 0.45, "verde",     "triangulo","grande"))
    objetos.append(criar_objeto_manual(0.20, 0.60, "vermelho",  "triangulo","pequeno"))
    objetos.append(criar_objeto_manual(0.60, 0.70, "azul",      "triangulo","grande"))
    objetos.append(criar_objeto_manual(0.90, 0.90, "verde",     "triangulo","pequeno"))

    return np.array(objetos, dtype=np.float32)


def gerar_dataset_aleatorio(num_objetos: int = 25) -> np.ndarray:
    """
    Gera uma massa de dados randômica com vetores de tamanho 11.
    Usa 3 cores (Vermelho, Verde, Azul) e 5 formas (Círculo, Quadrado,
    Cilindro, Cone, Triângulo) conforme as instruções do trabalho.
    """
    dataset = []
    for _ in range(num_objetos):
        x, y = np.random.rand(), np.random.rand()

        # One-hot para 3 cores e 5 formas
        cor   = np.eye(3)[np.random.randint(0, 3)]
        forma = np.eye(5)[np.random.randint(0, 5)]
        tamanho = float(np.random.choice([0.0, 1.0]))

        vetor = np.concatenate(([x, y], cor, forma, [tamanho]))
        dataset.append(vetor)

    return np.array(dataset, dtype=np.float32)


def plotar_cenario(dataset: np.ndarray, titulo: str = "Cenário CLEVR Simplificado") -> None:
    """
    Plota graficamente o cenário com cada objeto representado pelo marcador
    correspondente à sua forma, colorido pela sua cor one-hot.
    Exporta a figura como 'cenario_clevr.png'.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.set_xlabel("Eixo X (Esquerda → Direita)", fontsize=11)
    ax.set_ylabel("Eixo Y (Abaixo → Acima)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Mapeamento: [Círculo, Quadrado, Cilindro, Cone, Triângulo]
    marcadores = ["o", "s", "p", "v", "^"]
    nomes_formas = ["Círculo", "Quadrado", "Cilindro", "Cone", "Triângulo"]
    cores_visuais = [[0.9, 0.2, 0.2], [0.2, 0.75, 0.2], [0.2, 0.4, 0.9]]  # R, G, B

    for idx, obj in enumerate(dataset):
        px, py = obj[IDX_X], obj[IDX_Y]

        cor_idx   = int(np.argmax(obj[IDX_COR_INICIO:IDX_COR_FIM]))
        forma_idx = int(np.argmax(obj[IDX_FORMA_INICIO:IDX_FORMA_FIM]))
        tamanho_val = obj[IDX_TAMANHO]

        tamanho_marcador = 420 if tamanho_val == 1.0 else 140
        label_tamanho = "G" if tamanho_val == 1.0 else "P"

        ax.scatter(
            px, py,
            color=cores_visuais[cor_idx],
            marker=marcadores[forma_idx],
            s=tamanho_marcador,
            edgecolors="black",
            linewidths=1.2,
            alpha=0.85,
            zorder=3,
        )
        ax.text(px, py + 0.035, f"#{idx}({label_tamanho})",
                fontsize=8, ha="center", fontweight="bold", color="#222")

    # Legenda de formas
    handles = [
        mpatches.Patch(color="gray", label=f"{marcadores[i]} = {nomes_formas[i]}")
        for i in range(len(nomes_formas))
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              title="Formas", title_fontsize=9)

    plt.tight_layout()
    plt.savefig("cenario_clevr.png", dpi=150)
    print("  -> Figura salva em 'cenario_clevr.png'")
    plt.close()


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    print("=" * 60)
    print(" Pipeline de Dados — Trabalho IA (NeSy / LTNtorch)")
    print("=" * 60)

    print("\n[1] Gerando cenário fixo (25 objetos controlados)...")
    dados_fixos = gerar_dataset_fixo()
    print(f"    Dimensão da matriz: {dados_fixos.shape}  (esperado: 25 x 11)")

    print("\n[2] Gerando cenário aleatório (25 objetos)...")
    dados_aleatorios = gerar_dataset_aleatorio(num_objetos=25)
    print(f"    Dimensão da matriz: {dados_aleatorios.shape}  (esperado: 25 x 11)")

    print("\n[3] Plotando cenário fixo...")
    plotar_cenario(dados_fixos, titulo="Cenário Fixo de Calibração — 25 objetos (5 formas × 5)")

    print("\n[Concluído]")