import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PASTA_RESULTADOS = Path("resultados")
PASTA_RESULTADOS.mkdir(exist_ok=True)

# Leitura da imagem em tons de cinza.
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

# Cálculo dos gradientes Sobel.
# CV_64F preserva os valores negativos das derivadas.
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) # verticais
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) # horizontais
mag_sobel = np.hypot(sobel_x, sobel_y)  # sqrt(Gx² + Gy²)

# Cálculo dos gradientes Prewitt.
kernel_prewitt_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float64)
kernel_prewitt_y = kernel_prewitt_x.T
prewitt_x = cv2.filter2D(img, cv2.CV_64F, kernel_prewitt_x)
prewitt_y = cv2.filter2D(img, cv2.CV_64F, kernel_prewitt_y)
mag_prewitt = np.hypot(prewitt_x, prewitt_y)

# Cálculo dos gradientes Scharr.
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
mag_scharr = np.hypot(scharr_x, scharr_y)

# Visualização dos componentes X e Y e da magnitude Sobel.
fig_passos = criar_grade(
    [img, normalizar(np.abs(sobel_x)), normalizar(np.abs(sobel_y)), normalizar(mag_sobel)],
    ["Original", "Gradiente X (|Gx|)", "Gradiente Y (|Gy|)", "Magnitude (Sobel)"],
)

# Comparação entre Prewitt, Sobel e Scharr.
fig_operadores = criar_grade(
    [img, normalizar(mag_prewitt), normalizar(mag_sobel), normalizar(mag_scharr)],
    ["Original", "Prewitt", "Sobel", "Scharr"],
)

# Comparação de diferentes tamanhos do kernel Sobel.
tamanhos = (3, 9, 15)
sobel_ksize = [
    normalizar(np.hypot(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k),
                        cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)))
    for k in tamanhos
]
fig_ksize = criar_grade(sobel_ksize, [f"Sobel ksize={k}" for k in tamanhos])

# Comparação de diferentes limiares aplicados à magnitude Sobel.
limiares = (30, 80, 150)
base = normalizar(mag_sobel)
binarizadas = [cv2.threshold(base, t, 255, cv2.THRESH_BINARY)[1] for t in limiares]
fig_limiar = criar_grade(binarizadas, [f"Limiar T={t}" for t in limiares])

# Salvamento das figuras na pasta de resultados.
resultados = {
    "resultado_passos_sobel.png": fig_passos,
    "resultado_comparacao_operadores.png": fig_operadores,
    "resultado_ksize.png": fig_ksize,
    "resultado_limiar.png": fig_limiar,
}

for nome, figura in resultados.items():
    figura.savefig(
        PASTA_RESULTADOS / nome,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figura)
