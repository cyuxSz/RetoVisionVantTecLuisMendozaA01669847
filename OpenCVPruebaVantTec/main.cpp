// Carga un modelo YOLO26 exportado a ONNX, corre inferencia en vivo sobre la
// webcam, y dibuja cada detección (clase, confianza) sobre el video.

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <unordered_map>
#include <string>

using namespace cv;
using namespace cv::dnn;

static const std::unordered_map<int, std::string> clases = {
    {0, "persona"}, {24, "mochila"}, {39, "botella"}, {41, "taza"},
    {56, "silla"},  {62, "tv"},      {63, "laptop"},  {66, "teclado"},
    {67, "celular"},{73, "libro"}
};

// Una sola detección ya lista para dibujar: dónde está, qué tan segura está
// la red, y de qué clase es.
struct Deteccion {
    Rect2f caja;       
    float confianza;   
    int clase;         
};

// r, dx y dy son parámetros por referencia porque una función en C++ solo
// puede "retornar" un valor con `return`; para regresar 3 datos más, se
// modifican directamente las variables que (detectar()) le pasó.

Mat letterbox(const Mat& img, int tam, float& r, int& dx, int& dy) {
    // Se toma el menor de los dos factores de escala posibles, para que el
    // lado más grande de la imagen quepa justo en "tam" sin salirse.
    r = std::min((float)tam / img.cols, (float)tam / img.rows);

    int nw = (int)std::round(img.cols * r); 
    int nh = (int)std::round(img.rows * r);  

    // El espacio sobrante se reparte mitad y mitad, para centrar la imagen.
    dx = (tam - nw) / 2;
    dy = (tam - nh) / 2;

    Mat escalada;
    resize(img, escalada, Size(nw, nh));

    // Crea un lienzo cuadrado final, mismo tipo de píxel que la imagen original, relleno de gris.
    Mat salida(tam, tam, img.type(), Scalar(114, 114, 114));

    // Pega la imagen ya escalada dentro del lienzo, en la posición (dx, dy).
    // out(Rect()) crea una "ventana" (ROI) sobre el lienzo grande.
    escalada.copyTo(salida(Rect(dx, dy, nw, nh)));
    return salida;
}


// Corre un cuadro de video por la red y devuelve las detecciones que superan el umbral de confianza.
std::vector<Deteccion> detectar(Net& red, const Mat& frame, int tam, float uconfianza) {
    float r; int dx, dy;
    Mat entrada = letterbox(frame, tam, r, dx, dy);
    Mat blob = blobFromImage(entrada, 1.0 / 255.0, Size(tam, tam), Scalar(), true, false);
    red.setInput(blob);
    Mat salida = red.forward();  
    int filas = salida.size[salida.dims - 2];
    int cols  = salida.size[salida.dims - 1];

    // salida.data es un puntero genérico a los bytes crudos de la matriz; este cast le dice a C++ "interpreta esos bytes como floats".
    const float* datos = reinterpret_cast<const float*>(salida.data);

    std::vector<Deteccion> resultado;
    for (int i = 0; i < filas; ++i) {
        const float* fila = datos + (size_t)i * cols;  
        float confianza = fila[4];
        if (confianza < uconfianza) continue;  // descarta detecciones débiles

        // Revierte el letterbox: resta el relleno y divide entre el factor de escala, para regresar la caja al sistema de coordenadas del frame
        // original (no del cuadro cuadrado con relleno gris).
        Rect2f caja(
            (fila[0] - dx) / r,
            (fila[1] - dy) / r,
            (fila[2] - fila[0]) / r,
            (fila[3] - fila[1]) / r
        );

        resultado.push_back({caja, confianza, (int)fila[5]});
    }
    return resultado;
}

int main() {
    // Carga el modelo YOLO26-nano ya exportado a ONNX
    //ENGINE_AUTO deja que OpenCV 5 elija el motor 
    Net red = readNetFromONNX(String("yolo26n.onnx"), (int)ENGINE_AUTO);
    if (red.empty()) {
        std::cerr << "No se pudo cargar yolo26n.onnx. Verifica que este junto al ejecutable.\n";
        return -1;
    }

    VideoCapture camara(0);
    if (!camara.isOpened()) {
        std::cerr << "No se pudo abrir la webcam.\n";
        return -1;
    }

    const int tentrada    = 640;
    const float uconfianza = 0.4f;
    const char* ventana    = "Deteccion de obstaculos SDV";

    Mat frame;
    while (true) {
        camara >> frame;            
        if (frame.empty()) break;

        for (const auto& d : detectar(red, frame, tentrada, uconfianza)) {
            rectangle(frame, d.caja, Scalar(0, 255, 0), 2);
            auto it = clases.find(d.clase);
            std::string etiqueta = (it != clases.end()) ? it->second : "objeto";

            putText(frame, etiqueta + " " + std::to_string((int)(d.confianza * 100)) + "%",
                    Point((int)d.caja.x, (int)d.caja.y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        imshow(ventana, frame);

        if (waitKey(1) == 27 || getWindowProperty(ventana, WND_PROP_VISIBLE) < 1) break;
    }

    camara.release();
    destroyAllWindows();
    return 0;
}