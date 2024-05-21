import cv2

camera = cv2.VideoCapture(0)

# Definindo as coordenadas e dimensões do retângulo
x = 520  # X posição inicial do retângulo
y = 314  # Y posição inicial do retângulo
largura = 100  # Largura do retângulo
altura = 100   # Altura do retângulo
cor_retangulo = (255, 0, 0)  # Cor do retângulo (vermelho)
# Por inspeção 100px = 20 +/- 0,5 mm
espessura = 5

while True:
    ret, imagem = camera.read()

    cv2.rectangle(imagem, (x, y), (x + largura, y + altura), cor_retangulo, 1)

    cv2.imshow("Imagem", imagem)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

camera.release()
cv2.destroyAllWindows()

