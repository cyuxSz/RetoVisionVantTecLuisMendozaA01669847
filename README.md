# Reto de Vision VantTec LuisMendoza A01669847
# Detección de obstáculos para un SDV en YOLO26 + OpenCV 5

**Autor:** Luis Eduardo Mendoza Menéndez: A01669847

Aplicación en C++ que captura video en vivo de una webcam, detecta obstáculos
(personas, mochilas, sillas, etc.) usando el detector **YOLO26-nano** (se elige nano para menor carga computacional) exportado
a ONNX, y dibuja sobre cada cuadro la clase, la confianza y la posición de cada
detección.

## Contenido del repositorio

```
├── CMakeLists.txt      # configuración de compilación (CMake + OpenCV)
├── main.cpp            # código fuente
└── yolo26n.onnx        # no se sube al repositorio por tamaño. Se genera localmente siguiendo el paso 6 de la instalación, y `CMakeLists.txt` ya se encarga de copiarlo junto al ejecutable en cada compilación.
```

## Cómo funciona:

1. **Letterbox**: cada cuadro de la webcam se reescala a 640×640 sin deformar
   la imagen (se rellena el sobrante con gris), porque la red espera una
   entrada cuadrada.
2. **Inferencia**: el cuadro ajustado se convierte en un *blob* (tensor
   normalizado) y se pasa por la red mediante el módulo `cv::dnn` de OpenCV 5.
3. **Post-proceso**: YOLO26 es *NMS-free* — la salida `[1, 300, 6]` ya viene
   sin cajas duplicadas, así que solo se filtra por umbral de confianza y se
   revierte el letterbox para ubicar cada caja en las coordenadas del cuadro
   original.
4. **Visualización**: se dibuja un rectángulo, la clase (traducida al español
   para las clases más comunes de interior; el resto se etiqueta como
   "objeto") y el porcentaje de confianza.

El código fuente (`main.cpp`) incluye comentarios explicando funciones,
parámetros y decisiones de diseño.

## Requisitos

- CMake ≥ 3.16
- OpenCV 5 (con el módulo `dnn`)
- Compilador con soporte C++17
- Python 3 + [Ultralytics](https://github.com/ultralytics/ultralytics) (solo
  para exportar el modelo, no es dependencia del programa)
- Una webcam (externa o ya integrada en la laptop)
- VS Code (preferiblemente, pero se puede replicar en cualquier IDE/terminal)

## Instalación y compilación

### 1. Instalar CMake y OpenCV

- CMake: https://cmake.org/download/
- OpenCV: https://opencv.org/releases/

### 2. Configurar variables de entorno

Agregar al `PATH` del sistema:
- La carpeta `bin` de CMake.
- La carpeta `bin` de OpenCV.

Y agregar la variable de librería:
- La carpeta `lib` de OpenCV.

### 3. Extensiones de Visual Studio Code

Instalar:
- **C/C++** (Microsoft)
- **C/C++ Compile Run**
- **CMake** 
- **CMake Tools**

### 4. Generar el proyecto con CMake

  Usando el comando de VS Code: **CMake: Quick Start**.

### 5. Exportar el modelo YOLO26-nano a ONNX

Instalar Ultralytics en terminal (solo se usa una vez):

```bash
pip install ultralytics
```

Exportar el modelo:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", imgsz=640, opset=12)
```

Esto genera `yolo26n.onnx`.

### 6. Colocar el modelo en el proyecto

Copiar `yolo26n.onnx` a la raíz del proyecto (junto a `CMakeLists.txt` y
`main.cpp`). El `CMakeLists.txt` ya incluye un paso de post-build que lo copia
automáticamente junto al ejecutable cada vez que se compila:

```cmake
add_custom_command(TARGET OpenCVPruebaVantTec POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/yolo26n.onnx"
        "$<TARGET_FILE_DIR:OpenCVPruebaVantTec>/yolo26n.onnx"
)
```

## Ejecución

Desde la carpeta donde quedó el ejecutable o haciendo build en `CMakeLists.txt` en vscode y posteriormente compilando ahi mismo (no en el .cpp)

```bash
./OpenCVPruebaVantTec
```

Se abrirá una ventana con el video de la webcam y las detecciones dibujadas en
tiempo real.

- Presiona **ESC** para terminar el programa.

## Configuración

Dentro de `main.cpp`, en `main()`, se pueden ajustar dos valores sin tocar el
resto del código:

| Constante | Qué controla | Valor actual |
|---|---|---|
| `tentrada (main)` | Tamaño de entrada de la red (debe coincidir con el usado al exportar el ONNX) | `640` |
| `uconfianza` | Umbral mínimo de confianza para mostrar una detección | `0.4` |

## Clases detectadas

El modelo reconoce las 80 clases del dataset COCO. Se usan únicamente las más relevantes para un entorno de interior para una prueba sencilla
(persona, mochila, botella, taza, silla, tv, laptop, teclado, celular, libro);
cualquier otra clase detectada se etiqueta genéricamente como `"objeto"`. véase `main.cpp` en la linea 116 (si bien únicamente se debería escribir el nombre entre comillas para ahorrar algo de tiempo se tomó esta decisión, también quedando más compacto) 

Además he de decir que lo probé con las clases mencionadas ahí arriba y efectivamente reconoce los objetos en la cámara web interna de mi laptop.

## Referencias

- Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F. C., & Kalfaoglu, M. E. (2026). *Ultralytics YOLO26: Unified real-time end-to-end vision models*. arXiv. https://arxiv.org/html/2606.03748v1
- OpenCV. (s. f.). *Deep neural networks (dnn module)*. OpenCV Documentation. https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
- OpenCV. (2026). *OpenCV 5.0 release announcement*. https://opencv.org/opencv-5/
- Chakrabarty, S. (2026, 9 de julio). How to run object detection with OpenCV 5. LearnOpenCV. https://learnopencv.com/how-to-run-object-detection-with-opencv-5/#32-export-yolo26-to-onnx
- Chakrabarty, S. (2026, 28 de julio). Object detection with OpenCV 5 in C++: YOLO26 pose and segmentation. LearnOpenCV. https://learnopencv.com/opencv-5-cpp-object-detection-yolo26/
