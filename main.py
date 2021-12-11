import os
import cv2
import numpy as np
import tensorflow.keras as tf
import math

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def main():
    # abrir o arquivo label.txt
    labelsfile = open(f"{DIR_PATH}/labels.txt", 'r')  # caminho do arquivo de classes, tipo de abertura

    # inicializar classes e ler linhas
    classes = []
    line = labelsfile.readline()
    while line:
        # capturar apenas o nome da classe e anexar às classes
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsfile.readline()
    # fechar arquivo de classes
    labelsfile.close()

    # carregar o arquivo treinado no site teachable machine
    model_path = f"{DIR_PATH}/keras_model.h5"
    model = tf.models.load_model(model_path, compile=False)

    # inicializar webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # altere o número para modificar a webcam

    # largura e altura do vídeo da webcam em pixels -> você pode ajustar ao seu tamanho
    # ajuste os valores caso você veja barras pretas nas laterais da janela de captura
    frameWidth = 640
    frameHeight = 480

    # definir largura e altura em pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    # habilitar ganho automático
    cap.set(cv2.CAP_PROP_GAIN, 0)

    # mantém o programa em execução até apertar a tecla esc
    while True:

        # desative a notação científica para maior clareza
        np.set_printoptions(suppress=True)

        # Matriz para alimentar o modelo keras.
        # Inserir uma imagem RGB de 1x 224x224 pixels.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # capture a imagem
        check, frame = cap.read()
        # imagem espelhada - normalmente é espelhada
        # dependendo do seu computador/webcam, você pode ter que virar o vídeo
        frame = cv2.flip(frame, 1)

        # recortar para enquadrar
        margin = int(((frameWidth - frameHeight) / 2))
        square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        # redimensionar para 224x224 para uso com o modelo TM
        resized_img = cv2.resize(square_frame, (224, 224))
        # inverta a cor da imagem que irá para o modelo
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # transformar a imagem em uma matriz numpy
        image_array = np.asarray(model_img)
        # normalizar imagem
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # carregue a imagem no array
        data[0] = normalized_image_array

        # executar previsão
        predictions = model.predict(data)

        # o limite de confiança é 90%.
        conf_threshold = 90
        confidence = []
        conf_label = ""
        threshold_class = ""

        # criar borda preta na parte inferior para os rótulos
        per_line = 2  # número de classes por linha de texto
        bordered_frame = cv2.copyMakeBorder(
            square_frame,
            top=0,
            bottom=30 + 15 * math.ceil(len(classes) / per_line),
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        # para cada uma das classes
        for i in range(0, len(classes)):
            # dimensionar a confiança da previsão para % e aplicar à lista 1-D
            confidence.append(int(predictions[0][i] * 100))
            # colocar o texto por linha com base no número de classes por linha
            if i != 0 and not i % per_line:
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight + 25 + 15 * math.ceil(i / per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # anexar classes e confidências ao texto para rótulo
            conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
            # mostrar última linha
            if i == (len(classes) - 1):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight + 25 + 15 * math.ceil((i + 1) / per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # se acima do limite de confiança, enviar para a sequência/fila
            if confidence[i] > conf_threshold:
                threshold_class = "Probabilidade: " + classes[i]
        # adicionar classes de rótulo acima do limite de confiança
        cv2.putText(
            img=bordered_frame,
            text=threshold_class,
            org=(int(0), int(frameHeight + 20)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 255, 255)
        )

        cv2.imshow("Capturing", bordered_frame)

        tecla = cv2.waitKey(2)
        # mandar ele parar se o usuário clicar em "Esc"
        if tecla == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
