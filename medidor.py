import cv2
import numpy as np
from collections import deque
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def segmentar_menisco_cor(imagem):
  # Converte a imagem para o espaço de cores HSV
  imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

  # Limiarização em HSV para o verde
  mascara_verde = cv2.inRange(imagem_hsv[:, :, 0], 30, 100) \
                & cv2.inRange(imagem_hsv[:, :, 1], 80, 255) \
                & cv2.inRange(imagem_hsv[:, :, 2], 40, 150)

  kernel_gauss = np.ones((4, 4), np.float32) / 16  # Aumente o tamanho do kernel
  mascara_verde_filtrada = cv2.filter2D(mascara_verde.astype(np.float32), -1, kernel_gauss)
  mascara_verde_filtrada = cv2.bilateralFilter(mascara_verde_filtrada.astype(np.float32), 15, 150, 100)
  mascara_verde_filtrada = mascara_verde_filtrada.astype(np.uint8)

  return mascara_verde_filtrada


def calcular_altura(segmentacao, imagem):
  # Encontra os contornos do menisco
  contornos, _ = cv2.findContours(segmentacao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Assume que o menisco é o maior contorno
  maior_contorno = max(contornos, key=cv2.contourArea)

  # Encontra o centroide do menisco
  momentos = cv2.moments(maior_contorno)
  centroide = (momentos["m10"] / momentos["m00"], momentos["m01"] / momentos["m00"])

  # Calcula a altura do menisco como a distância do centroide à borda inferior da imagem
  altura_base, i, x = imagem.shape[0]
  altura = imagem.shape[0] - centroide[1]

  return altura

def find_peaks_positions(mask_image, imagem):
    # Find contours in the binary mask
    _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to store the positions of the peaks
    peak_positions = []
    
    # Define the y-threshold to filter out lower points
    y_threshold = 530 # Adjust this based on your image
    
    # Iterate over the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Filter contours based on their area to identify the peaks
        if area > 200: # Example threshold, adjust as needed
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Generate a guideline based on the centroid's X coordinate
                start_col = cX - 10 # Adjust the range as needed
                end_col = cX + 10
                
                # Ensure the guideline is within the image bounds
                start_col = max(0, start_col)
                end_col = min(mask_image.shape[1], end_col)
                
                # Extract the guideline from the mask image
                guideline_mask = mask_image[:, start_col:end_col]
                
                # Find the highest white point in the guideline
                highest_point = np.unravel_index(np.argmax(guideline_mask), guideline_mask.shape)
                
                # Check if the y-coordinate of the highest point is above the y-threshold
                if highest_point[0] < y_threshold:
                    # Calculate the x-coordinate of the peak
                    peak_x = start_col + highest_point[1]
                    
                    # Store the peak position
                    peak_positions.append((peak_x, highest_point[0]))
                    
                    # Plot the peak on the original image
                    cv2.circle(imagem, (peak_x, highest_point[0]-7), 3, (255, 0, 255), -1) # Green circle
                    cv2.putText(imagem, f"({peak_x}, {highest_point[0]})", (peak_x, highest_point[0]-17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    if len(peak_positions) >= 2:
        # Assuming the points are sorted in the order they appear on the screen
        h_diff = abs(peak_positions[0][1] - peak_positions[-1][1])
    else:
        h_diff = 0    
    
    return mask_image, imagem, h_diff

def media_movel(nova_medida, janela=10):
  """
  Calcula a média móvel de uma série de medidas.

  Args:
      nova_medida: A nova medida a ser adicionada à janela.
      janela: O tamanho da janela da média móvel (número de medidas).

  Returns:
      A média móvel atualizada.
  """
  # Inicializa a fila (deque) para armazenar as medidas da janela
  global medidas  # Torna a variável acessível dentro da função
  if "medidas" not in globals():  # Verifica se a variável já existe
    medidas = deque(maxlen=janela)
  
  #Verificad a validade da nova medida
  if nova_medida == 0:
    if len(medidas) == 0:  # Verifica se a fila está vazia
      return 0  # Retorna 0 se não houver medidas válidas
    else:
      return sum(medidas) / len(medidas)

  # Adiciona a nova medida à fila
  medidas.append(nova_medida)
  
  # Calcula e retorna a média móvel
  return sum(medidas) / len(medidas)

def calcular_velocidade(D, d, Beta, DeltaP, rho, Ny, g, V_chute, F_1, F_2):
  # Iterações para encontrar o Cd verdadeiro
  tol = 1e-6  # Tolerância para a convergência
  max_iter = 100  # Número máximo de iterações
  it = 0  # Contador de iterações
  convergiu = False

  while not convergiu and it < max_iter:
    it += 1

    # Calcula o número de Reynolds com o chute inicial de V
    R_d = ((V_chute * (D * 10**(-3)))/ Ny)
    
    #R_d = R_d if R_d > 0 else 0.000001

    # Calcula o coeficiente de descarga com o Reynolds atual
    Cd = 0.5959 + 0.0312 * Beta**2.1 - 0.184 * Beta**8 + 91.71 * Beta**2.5 * (R_d)**(-0.75) + (((0.09 * Beta**4) / (1 - Beta**4)) *F_1) - ((0.0337 * Beta**3) * F_2)

    # Calcula a nova velocidade do fluxo com o Cd atual
    V_orifício = Cd * ((2 * (DeltaP) / rho) / (1 - Beta**4))**(1/2)

    # Novo chute para V antes da placa
    V_nova = V_orifício * (Beta**2)

    # Verifica a convergência
    erro = abs(V_nova - V_chute )/ V_chute
    if erro < tol:
      convergiu = True

    # Atualiza o chute inicial de V para a próxima iteração
    V_chute = V_nova

  return V_chute

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
  # Captura uma imagem do manômetro
  #camera = cv2.VideoCapture(0)  #camera
  camera = cv2.VideoCapture("WhatsApp Video 2024-03-28 at 09.45.25.mp4") #video para teste offline

  # Parâmetros do problema
  D = 43.2  # Diâmetro interno da tubulação [mm]
  escolha = 3 # "Vena-contractive taps" (~1D à montante e no local da Vena à jusante) se aproximam de Flange taps
  d_1 = 8.0   # [mm]
  d_2 = 15.0  # [mm]
  d_3 = 22.0  # [mm]
  Beta_1 = d_1 / D
  Beta_2 = d_2 / D
  Beta_3 = d_3 / D

  F_1 = 0.4333  # Cria uma variável global
  F_2 = 0.5880  # Cria uma variável global

  # Parâmetros do fluido
  # Considerando ar a 20 graus celcius -> https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and_kinematic_14483.htm
  rho_ar = 1.184  # [kg/m^3]
  Ny = 1.516E-5  # [m^2/s]
  g = 9.81  # [m/s^2]

  # Chute inicial de velocidade
  V_chute = 2  # [m/s]  -> Chute inicial para a velocidade

  # Considerando água a 20 graus celcius -> https://www.engineersedge.com/physics/water__density_viscosity_specific_weight_13146.htm
  rho_agua = 998.21 # [kg/m^3]
  
  while True:
    ret, imagem = camera.read()

  # Segmenta o menisco
    segmentacao = segmentar_menisco_cor(imagem)

  # Encontra os pontos de menisco
    segmentacao, imagem, h_diff = find_peaks_positions(segmentacao, imagem)
    altura_filtrada = media_movel(h_diff)
    hdiff_mm = 20/100 * altura_filtrada

  # Cálculo da diferença de pressão
    DeltaP = rho_agua*g*(hdiff_mm)*10**(-3) #[Pa]
    print(DeltaP)

  # Calcular velocidade
    V = calcular_velocidade(D, d_2, Beta_2, DeltaP, rho_ar, Ny, g, V_chute, F_1, F_2)

  # Calcular vazao
    Vazao = math.pi * (D**2)/4 *  V*(10**(-3))
    
  # Especificando o tipo de fonte e outras configurações do texto
    posicao_centro_superior = (int(imagem.shape[1] / 2), int(imagem.shape[0] * 0.1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    escala_fonte = 1
    cor_texto = (0, 0, 0) 
    espessura_linha = 2

# Adicionando o primeiro valor à imagem
    hdiff_mm = round(hdiff_mm, 3)
    hdiff_mm_str = str(hdiff_mm)
    imagem = cv2.putText(imagem, hdiff_mm_str, posicao_centro_superior, font, escala_fonte, cor_texto, espessura_linha, cv2.LINE_AA)

# Adicionando o segundo valor à imagem
    Vazao = round(Vazao, 3)
    vazao_str = str(Vazao)
    print(Vazao)
    print(hdiff_mm)
    imagem = cv2.putText(imagem, vazao_str, (posicao_centro_superior[0], posicao_centro_superior[1] + 25), font, escala_fonte, cor_texto, espessura_linha, cv2.LINE_AA)


  # Mostra a imagem segmentada e a altura do menisco
    cv2.imshow("Mascara", segmentacao)
    cv2.imshow("Imagem", imagem)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()