import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PASTA_RESULTADOS = Path("resultados")
PASTA_RESULTADOS.mkdir(exist_ok=True)

# Leitura em escala de cinza (1 canal)
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

def normalizar(matriz):
    """Reescala os valores para [0, 255] (uint8) para poder exibir e salvar."""
    return cv2.normalize(matriz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def criar_grade(imagens, titulos):
    """Monta uma figura com as imagens lado a lado e devolve a figura."""
    fig, eixos = plt.subplots(1, len(imagens), figsize=(16, 4))
    for ax, im, titulo in zip(eixos, imagens, titulos):
        ax.imshow(im, cmap="gray")
        ax.set_title(titulo)
        ax.axis("off")
    fig.tight_layout()
    return fig

# Gradientes. A saída em CV_64F mantém os valores negativos das derivadas;
# com saída uint8 eles seriam cortados em 0 e metade das bordas se perderia.

# Sobel: kernel com pesos 1, 2, 1 (maior peso na linha/coluna central)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
mag_sobel = np.hypot(sobel_x, sobel_y)  # sqrt(Gx² + Gy²)

# Prewitt: pesos uniformes 1, 1, 1. O OpenCV não tem função pronta,
# então o kernel é aplicado com filter2D.
kernel_prewitt_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float64)
kernel_prewitt_y = kernel_prewitt_x.T
prewitt_x = cv2.filter2D(img, cv2.CV_64F, kernel_prewitt_x)
prewitt_y = cv2.filter2D(img, cv2.CV_64F, kernel_prewitt_y)
mag_prewitt = np.hypot(prewitt_x, prewitt_y)

# Scharr: pesos 3, 10, 3, com resposta mais uniforme em bordas diagonais
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
mag_scharr = np.hypot(scharr_x, scharr_y)

# Passos intermediários do Sobel: bordas verticais (Gx), horizontais (Gy) e magnitude
fig_passos = criar_grade(
    [img, normalizar(np.abs(sobel_x)), normalizar(np.abs(sobel_y)), normalizar(mag_sobel)],
    ["Original", "Gradiente X (|Gx|)", "Gradiente Y (|Gy|)", "Magnitude (Sobel)"],
)

# Comparação entre os três operadores
fig_operadores = criar_grade(
    [img, normalizar(mag_prewitt), normalizar(mag_sobel), normalizar(mag_scharr)],
    ["Original", "Prewitt", "Sobel", "Scharr"],
)

# Efeito do tamanho do kernel no Sobel
tamanhos = (3, 5, 7)
sobel_ksize = [
    normalizar(np.hypot(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k),
                        cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)))
    for k in tamanhos
]
fig_ksize = criar_grade(sobel_ksize, [f"Sobel ksize={k}" for k in tamanhos])

# Efeito do limiar na binarização da magnitude do Sobel
limiares = (30, 80, 150)
base = normalizar(mag_sobel)
binarizadas = [cv2.threshold(base, t, 255, cv2.THRESH_BINARY)[1] for t in limiares]
fig_limiar = criar_grade(binarizadas, [f"Limiar T={t}" for t in limiares])

# Salva as figuras no final
resultados = {
    "resultado_passos_sobel.png": fig_passos,
    "resultado_comparacao_operadores.png": fig_operadores,
    "resultado_ksize.png": fig_ksize,
    "resultado_limiar.png": fig_limiar,
}
for nome, fig in resultados.items():
    fig.savefig(PASTA_RESULTADOS / nome, dpi=150)
    plt.close(fig)
print(f"Resultados salvos em: {PASTA_RESULTADOS.resolve()}")
