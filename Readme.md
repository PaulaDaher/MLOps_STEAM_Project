


# <h1 align=left> MACHINE LEARNING OPERATIONS (MLOps) </h1>
# <h3 align=left>**`PAULA DAHER`**</h3>

<p align="center">
<img src="image.png" height=500>
</p>

# <h3 align=left>**`DESCRIPCIÓN DEL PROYECTO`**</h3>
Este proyecto consiste en desarrollar un sistema de recomendación de videojuegos para los usuarios de Steam, utilizando técnicas de MLOps para asegurar que el modelo y la API sean escalables, reproducibles y mantenibles.
Conclusión
Este proyecto demostró la aplicación práctica de técnicas de MLOps para desarrollar un sistema de recomendación de videojuegos. Se completaron todas las etapas del ciclo de vida del proyecto, desde el tratamiento de datos hasta el despliegue de una API funcional.
</p>

# <h3 align=left>**`OBJETIVOS`**</h3>

**Transformaciones de Datos:** Leer y limpiar el dataset, eliminando columnas innecesarias para optimizar el rendimiento. 

**Feature Engineering:** Realizar análisis de sentimiento en las reseñas de usuarios y crear una nueva columna 'sentiment_analysis'. 

**Desarrollo de API:** Implementar una API con FastAPI que permita consultar datos y recomendaciones. 

**Despliegue:** Desplegar la API en un servicio web para que sea accesible públicamente. 

**Análisis Exploratorio de Datos (EDA):** Explorar y visualizar los datos para obtener insights valiosos.

**Modelo de Aprendizaje Automático:** Desarrollar un sistema de recomendación basado en la similitud del coseno.


# <h3 align=left>**`ESTRUCTURA DEL PROYECTO`**</h3>
- data: Contiene los datasets utilizados.
- notebooks: Jupyter notebooks para EDA y desarrollo del modelo.
- app: Código fuente de la API.
- main.py: Archivo principal de la API.
- models: Contiene el modelo de recomendación entrenado.
- requirements.txt: Lista de dependencias del proyecto.
- README.md: Descripción y guía del proyecto.


# <h3 align=left>**`PASOS REALIZADOS`**</h3>
1. Transformaciones de Datos
Se realizó la limpieza y preprocesamiento de los datos:
Se eliminaron columnas innecesarias.
Se manejaron valores nulos y datos anidados.

2. Feature Engineering
Se aplicó análisis de sentimiento en las reseñas de usuarios:
Se creó la columna 'sentiment_analysis' con valores 0 (negativo), 1 (neutral) y 2 (positivo).
Se asignó un valor de 1 a las reseñas ausentes.

3. Desarrollo de API
Se implementaron los siguientes endpoints en la API utilizando FastAPI:
developer(desarrollador: str): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
userdata(User_id: str): Cantidad de dinero gastado por el usuario, porcentaje de recomendación y cantidad de items.
UserForGenre(genero: str): Usuario con más horas jugadas para el género dado y acumulación de horas jugadas por año de lanzamiento.
best_developer_year(año: int): Top 3 de desarrolladores con juegos más recomendados por usuarios para el año dado.
developer_reviews_analysis(desarrolladora: str): Análisis de reseñas de usuarios categorizados con análisis de sentimiento (positivo o negativo).
recomendacion_juego(id de producto: str): Lista de 5 juegos recomendados similares al ingresado.

4. Despliegue
La API se desplegó utilizando Render para que sea accesible públicamente desde cualquier dispositivo conectado a Internet.

5. Análisis Exploratorio de Datos (EDA)
Se realizó un análisis exploratorio de los datos para entender mejor las relaciones entre las variables, detectar outliers y patrones interesantes. Se utilizaron técnicas como nubes de palabras para visualizar las palabras más frecuentes en los títulos de los juegos.

6. Modelo de Aprendizaje Automático
Se desarrolló un sistema de recomendación basado en la similitud del coseno:
Input: ID de un producto.
Output: Lista de 5 juegos recomendados similares al ingresado.

7. Video de Demostración
Se creó un video demostrando el funcionamiento de las consultas de la API y el modelo de ML entrenado. El video muestra cómo se realizan las consultas y explica brevemente el modelo utilizado para el sistema de recomendación.

# <h3 align=left>**`TECNOLOGÍA UTILIZADA`**</h3>
Python: Lenguaje de programación principal.
FastAPI: Framework para el desarrollo de la API.
Pandas: Manipulación y análisis de datos.
Scikit-learn: Desarrollo del modelo de recomendación.
NLTK: Análisis de sentimiento.
Render: Despliegue de la API.


# <h3 align=left>**`API Y VIDEO`**</h3>
## Para más detalles, puedes consultar el código y los notebooks en este repositorio. Además, puedes ver el video de demostración aquí:
### Acceder a la API en https://proyecto-mlops-steam-4gux.onrender.com
### Ver video  




