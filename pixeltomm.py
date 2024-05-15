import cv2

camera = cv2.VideoCapture(0)

# Definindo as coordenadas e dimensões do retângulo
x = 550  # X posição inicial do retângulo
y = 312  # Y posição inicial do retângulo
largura = 100  # Largura do retângulo
altura = 54   # Altura do retângulo
cor_retangulo = (255, 0, 0)  # Cor do retângulo (vermelho)

while True:
    ret, imagem = camera.read()

    cv2.rectangle(imagem, (x, y), (x + largura, y + altura), cor_retangulo, 2)

    cv2.imshow("Imagem", imagem)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

camera.release()
cv2.destroyAllWindows()

