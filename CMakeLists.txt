cmake_minimum_required(VERSION 3.15)
project(gwo_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Le dice al compilador que genere código específico para la CPU en la que estás compilando,
# usando todas sus instrucciones modernas (AVX, AVX2, FMA, etc.).
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Activa la Optimización en Tiempo de Enlace (LTO). Permite al compilador optimizar
# a través de diferentes archivos .cpp, encontrando oportunidades de inlining.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
# 1. Inclusión de directorios estándar (bin/lib) para install()
include(GNUInstallDirs)

# 2. Encontrar Python y pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# 3. Encontrar Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# -------------------------------------------------------------------
# Definición de Targets (Módulo y Ejecutable)
# -------------------------------------------------------------------

# 4. Crear el módulo Python
pybind11_add_module(gwo_py MODULE bindings.cpp)
target_link_libraries(gwo_py PRIVATE Eigen3::Eigen)

# -------------------------------------------------------------------
# Reglas de Instalación
# -------------------------------------------------------------------

# 6. Instalación del módulo Python
install(TARGETS gwo_py
    COMPONENT python
    LIBRARY DESTINATION "${Python3_SITELIB}"
)

# 7. Añadir el subdirectorio de ejemplos
add_subdirectory(examples)
