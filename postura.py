import cv2
from cv2 import cuda

# Crea l'oggetto VideoCapture per acquisire il video in tempo reale dalla webcam o da un file video
cap = cv2.VideoCapture(0)  # 0 indica la webcam integrata nel computer

# Verifica se la cattura del video è avvenuta correttamente
if not cap.isOpened():
    print("Errore nell'apertura del video.")
    exit()

# Crea un oggetto cv2.cuda_GpuMat per leggere i frame video dalla GPU
gpuFrame = cuda.GpuMat()

# Loop per leggere i frame video in tempo reale
while True:
    # Leggi un frame video dalla sorgente (webcam o file video)
    ret, frame = cap.read()

    # Verifica se il frame è stato letto correttamente
    if not ret:
        print("Errore nella lettura del frame.")
        break

    # Sposta il frame sulla GPU
    gpuFrame.upload(frame)

    # Applica le operazioni di elaborazione desiderate utilizzando le funzioni CUDA di OpenCV
    # Ad esempio, converti il frame a scala di grigi sulla GPU
    gpuGray = cuda.GpuMat()
    cuda.cvtColor(gpuFrame, cv2.COLOR_BGR2GRAY, gpuGray)

    # Sposta il frame di output dalla GPU alla RAM
    gray = gpuGray.download()

    # Visualizza il frame elaborato
    cv2.imshow('Frame in Scala di Grigi', gray)

    # Interruzione del loop se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()