// Prueba de vision vanttec, hecho por: Luis Eduardo Mendoza Menendez A01669847
// Carga un modelo YOLO26 exportado a ONNX, corre inferencia en vivo sobre la
// webcam, y dibuja cada detección (clase, confianza) sobre el video.

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <unordered_map>
#include <string>

using namespace cv;
using namespace cv::dnn;

//uso solo unas clases del datases coco para que sea un poco mas breve el codigo y ademas asi no hayq ue nombrar a todas para español
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

// escala, relleno_x y relleno_y son parámetros por referencia porque una función en C++ solo
// puede "retornar" un valor con `return`; para regresar 3 datos más, se
// modifican directamente las variables que (detectar()) le pasó.

Mat letterbox(const Mat& img, int tentrada, float& escala, int& relleno_x, int& relleno_y) {
    // Se toma el menor de los dos factores de escala posibles, para que el
    // lado más grande de la imagen quepa justo en "tam" sin salirse.
    escala = std::min((float)tentrada / img.cols, (float)tentrada / img.rows);

    int nuevoAncho = (int)std::round(img.cols * escala); 
    int nuevoAlto = (int)std::round(img.rows * escala);  

    // El espacio sobrante se reparte mitad y mitad, para centrar la imagen.
    relleno_x = (tentrada - nuevoAncho) / 2;
    relleno_y = (tentrada - nuevoAlto) / 2;

    Mat escalada;
    resize(img, escalada, Size(nuevoAncho, nuevoAlto));

    // Crea un lienzo cuadrado final, mismo tipo de píxel que la imagen original, relleno de gris.
    Mat salida(tentrada, tentrada, img.type(), Scalar(114, 114, 114));

    // Pega la imagen ya escalada dentro del lienzo, en la posición (dx, dy).
    // out(Rect()) crea una "ventana" (ROI) sobre el lienzo grande.
    escalada.copyTo(salida(Rect(relleno_x , relleno_y, nuevoAncho, nuevoAlto)));
    return salida;
}


// Corre un cuadro de video por la red y devuelve las detecciones que superan el umbral de confianza.
std::vector<Deteccion> detectar(Net& red, const Mat& frame, int tentrada, float uconfianza) {
    float escala; int relleno_x, relleno_y;
    Mat entrada = letterbox(frame, tentrada, escala, relleno_x, relleno_y);
    Mat blob = blobFromImage(entrada, 1.0 / 255.0, Size(tentrada, tentrada), Scalar(), true, false);
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
            (fila[0] - relleno_x) / escala,
            (fila[1] - relleno_y) / escala,
            (fila[2] - fila[0]) / escala,
            (fila[3] - fila[1]) / escala
        );

        resultado.push_back({caja, confianza, (int)fila[5]});
    }
    return resultado;
}

int main() {
    // Carga el modelo YOLO26-nano ya exportado a ONNX
    // ENGINE_AUTO deja que OpenCV 5 elija el motor 
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

    const int tentrada    = 640; //tamaño de entrada que debe coincidir con el exportado en este caso 640, vease script de exportación a onnx en el readme
    const float uconfianza = 0.4f; //umbral de confianza para que detecte mas preciso y no cosas que no este tan seguro
    const char* ventana    = "Deteccion de obstaculos SDV";

    Mat frame;
    while (true) {
        camara >> frame;            
        if (frame.empty()) break;

        for (const auto& d : detectar(red, frame, tentrada, uconfianza)) {
            rectangle(frame, d.caja, Scalar(0, 255, 0), 2);
            auto it = clases.find(d.clase);
            std::string etiqueta = (it != clases.end()) ? it->second : "objeto"; //para el resto de clases se le llamara simplemente objeto 

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
