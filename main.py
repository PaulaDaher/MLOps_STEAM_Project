from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


# Indicamos título y descripción de la API
app = FastAPI(title='PROYECTO INDIVIDUAL 1 - MLOps - Paula Daher',
            description='STEAM: API de datos y recomendaciones de videos juegos')



#librerias necesarias para portada
from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory='Templates')
@app.get('/', tags=['Pagina Principal'])
async def read_root(request: Request):
    return templates.TemplateResponse('portada.html', {'request': request})

@app.get('/about/')
async def about():
    return {'PROYECTO INDIVIDUAL 1 - Machine Learning Operations - MVP para STEAM'}



# Cargamos los archivos parquet
df1 = pd.read_parquet('consulta1.parquet')
games = pd.read_parquet('games.parquet')
reviews = pd.read_parquet('reviews.parquet')
items = pd.read_parquet('items.parquet')
recomendacion = pd.read_parquet('recomendacion.parquet')



@app.get('/Developer', tags=['GET'])
def developer(desarrollador : str=Query(default='Kotoshiro')):
    """
    Devuelve cantidad de items y porcentaje de contenido Free por año según desarrollador\n.

    Parametro
    ---------
    str
        developer: nombre del desarrollador
    
    Retorna
    -------
        Año                     : Años en que la empresa desarrolló contenido
        Cantidad de items       : Cantidad de items que se desarrolló en esos años
        Contenido Free          : Porcentaje de contenido gratis hecho cada año
    
    """

    #filtrar juego por desarrollador
    filter_games = df1[df1['developer'] == desarrollador]

    if filter_games.empty:
        return f"No hay datos disponibles para el desarrollador '{desarrollador}', asegurese de haber escrito correctamente el nombre"

    # Agrupar por año de lanzamiento y calcular cantidad de items y juegos gratuitos
    filter_games = filter_games.groupby('release_date').agg(total_items=('content_id', 'count'), free_game=('price', lambda x: (x == 0.0).sum())).reset_index()
    
    # Calcular el porcentaje de contenido gratuito
    filter_games['Contenido Free'] = round((filter_games['free_game'] / filter_games['total_items']) * 100, 2)
    
    # Formatear los resultados
    año = filter_games['release_date'].tolist()
    cantidad_items = filter_games['total_items'].tolist()
    cont_free = filter_games['Contenido Free'].tolist()
    
    # Retornar el resultado en el formato solicitado
    return {'Año': año, 'Cantidad de items': cantidad_items, 'Contenido Free': cont_free}






@app.get('/Userdata', tags=['GET'])
def userdata(User_id: str = Query(default='MeaTCompany')):
    """
    Devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.

    Parametro
    ---------
    str
        user_id: identificador del usuario
    
    Retorna
    -------
        Usuario                : ID del usuario
        Dinero gastado         : Total de dinero gastado por el usuario
        Cantidad de items      : Cantidad de items que el usuario tiene
        % de recomendacion     : Porcentaje de recomendación basado en reviews
    """
    
    # Selección de columnas
    game = games[['content_id', 'price']]
    
    # Filtro por usuario
    user_items = items[items['user_id'] == User_id]

    if user_items.empty:
        return {'Usuario': User_id, 'Dinero gastado': 0.0, 'Cantidad de items': 0, '% de recomendacion': 0.0}

    # Calcular el porcentaje de recomendación
    if reviews.empty:
        recomend = 0.0
    else:
        recomend = round(reviews['review_pos'].sum() / (reviews['review_pos'].sum() + reviews['review_neg'].sum()) * 100, 2)

    # Unión de los datos
    user_items = user_items.merge(game, left_on='item_id', right_on='content_id', how='inner')
    if user_items.empty:
        return {'Usuario': User_id, 'Dinero gastado': 0.0, 'Cantidad de items': 0, '% de recomendacion': recomend}

    # Agrupación y sumatoria de datos
    aggregated_data = user_items.groupby('user_id').agg({'price': 'sum', 'item_count': 'sum'}).reset_index()

    if aggregated_data.empty:
        return {'Usuario': User_id, 'Dinero gastado': 0.0, 'Cantidad de items': 0, '% de recomendacion': recomend}

    # Variables:
    usuario = aggregated_data['user_id'].iloc[0]
    total_dinero = float(aggregated_data['price'].iloc[0])  # Convertir a float nativo de Python

    # Obtener el valor único de item_count
    unique_item_count = user_items['item_count'].unique()
    if len(unique_item_count) > 1:
        print(f"Advertencia: Hay más de un valor único en item_count para el usuario {User_id}. Usando el primero.")
    total_items = int(unique_item_count[0]) if len(unique_item_count) > 0 else 0  # Convertir a int nativo de Python

    print(f"Total ítems: {total_items}, Dinero gastado: {total_dinero}")

    return {
        'Usuario': usuario,
        'Dinero gastado': round(total_dinero, 2),
        'Cantidad de items': total_items,
        '% de recomendacion': recomend
    }





@app.get('/UserForGenre', tags=['GET'])
def UserForGenre(genre: str=Query(default='Indie')):
    
    """
    Ingresa un género y devuelve el usuario que acumula más horas jugadas para él, y una lista de la acumulación de horas jugadas por año de lanzamiento.\n.
    
    Parametro
    ---------
    str
        genre: Ej > Action, Casual, Indie, Simulation, Strategy, RPG, Sports, Adventure, Racing
    
    Retorna
    -------
        Usuario con más horas jugadas para Género                    
        Llista de la acumulación de horas jugadas por año de lanzamiento

    """

    if genre not in games['genre'].unique():
        return f"El género '{genre}' no existe. Prueba ingresando alguno de los generos descriptos en Parámetros"
    
    # Filtrar juegos por género
    filter_games = games[games['genre'] == genre]
    
    # Unir con el DataFrame de items utilizando content_id y item_id
    merge_df = filter_games.merge(items, left_on='content_id', right_on='item_id')
    
    if merge_df.empty:
        return {"Usuario con más horas jugadas para Género": None, "Horas jugadas": []}
    
    # Calcular el usuario con más horas jugadas
    user_playtime = merge_df.groupby('user_id')['playtime_forever'].sum().reset_index()
    max_playtime_user = user_playtime.loc[user_playtime['playtime_forever'].idxmax()]['user_id']
    
    # Agrupar las horas jugadas por año de lanzamiento
    merge_df['release_year'] = pd.to_datetime(merge_df['release_date']).dt.year
    playtime_por_anio = merge_df.groupby('release_year')['playtime_forever'].sum().reset_index()
    
    # Convertir a la estructura requerida
    playtime_por_anio_list = playtime_por_anio.to_dict('records')
    
    return {"Usuario con más horas jugadas para Género": max_playtime_user, "Horas jugadas": playtime_por_anio_list}




@app.get('/BestDeveloperYear', tags=['GET'])
def best_developer_year(year: int=Query(default=2017)):

    """
    Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. 
    
    Parametro
    ---------
    int
        year: año
    
    Retorna
    -------
        Puesto 1, 2 y 3 de mejores desarrolladores del año

    """

    #Filtrar columnas del df1
    game = df1[['developer','title','content_id','release_date']]

    if game.empty:
        return "No han habido desarrollos para ese año, prueba ingresando otro año"
    
    # Filtrar los juegos por año dado
    game = df1[df1['release_date'] == year]

    #Filtramos columnas del df reviews
    review = reviews[['item_id', 'review_pos']]

    #Filtramos reviews positivas
    reviews_filter = game.merge(review, left_on='content_id', right_on='item_id', how='inner')

    if reviews_filter.empty:
        return "No hay reviews para los juegos lanzados en ese año"
    
    reviews_filter = reviews_filter[['developer', 'title', 'review_pos']]
    reviews_filter = reviews_filter.groupby(['developer','title']).agg({'review_pos': 'sum'}).reset_index()
    reviews_filter = reviews_filter.sort_values(by='review_pos', ascending=False)

    result = []

    # Obtener los puestos
    puestos = ['Puesto1', 'Puesto2', 'Puesto3']
    for puesto in puestos:
        if len(reviews_filter) >= int(puesto[-1]):
            developer_name = reviews_filter.iloc[int(puesto[-1]) - 1, 0]
            result.append({puesto: developer_name})
        else:
            result.append({puesto: reviews_filter.iloc[0, 0]})

    return result




@app.get('/DeveloperReviewsAnalysis', tags=['GET'])
def developer_reviews_analysis(desarrollador: str = Query(default='SCS Software')):
    """
    Devuelve la cantidad de reseñas positivas y negativas para el desarrollador ingresado

    Parametro
    ---------
    str
        desarrollador: nombre del desarrollador
    
    Retorna
    -------
        Devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas positivas y negativas
    """
    develop = df1[['developer', 'content_id']]

    # Filtrar los juegos por el desarrollador dado
    develop = develop[develop['developer'] == desarrollador]

    if develop.empty:
        return {"error": "No existe este desarrollador, asegurese de haber escrito bien el nombre"}

    # Filtramos columnas del df reviews
    review = reviews[['item_id', 'review_pos', 'review_neg']]

    # Unimos dataframes
    develop_filter = develop.merge(review, left_on='content_id', right_on='item_id', how='inner')

    if develop_filter.empty:
        return {"error": "No hay reviews para los juegos lanzados por este desarrollador en este MVP, pruebe con otro desarrollador"}
    
    develop_filter = develop_filter.groupby(['developer']).agg({'review_pos': 'sum', 'review_neg': 'sum'}).reset_index()
    
    developer = develop_filter['developer'].iloc[0]
    reviews_positivas = int(develop_filter['review_pos'].iloc[0])  # Convertir a int nativo de Python
    reviews_negativas = int(develop_filter['review_neg'].iloc[0])  # Convertir a int nativo de Python
    
    return {developer: {'reviews positivas': reviews_positivas, 'reviews negativas': reviews_negativas}}




@app.get('/Recomendacion juego', tags=['GET'])
def calcular_similitud(id_producto : int=Query(default=761140)):
    """
    Sistema de recomendación por similitud del coseno 
    Ingresando el id de un producto, recibimos 5 juegos recomendados similares al ingresado.

    Parametro
    ---------
    int
        id_producto: id del item
    
    Retorna
    -------
        Devuelve lista con 5 juegos recomendados similares al ingresado

    """

    # Combina las columnas 'genre', 'tags' y 'specs' en una sola columna
    recomendacion['combined'] = recomendacion.apply(lambda row: f"{row['genre']}, {row['tags']}, {row['specs']}", axis=1)

    # Comenzamos con el vectorizado 
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(recomendacion['combined'])

    product_index = recomendacion[recomendacion['content_id'] == id_producto].index[0]

    if product_index is None:
        return {'No se esncuentra un juego con el ID proporcionado, pruebe con otro ID'}

    product_vector = matrix[product_index]

    # Cambia la forma de product_vector a una matriz 2D
    product_vector_2d = product_vector.reshape(1, -1)

    cosine_similarity_matrix = cosine_similarity(product_vector_2d, matrix)

    # Obtenemos la similitud con otros items
    product_similarities = cosine_similarity_matrix[0]

    # Obtenemos los indices de los primeros 5 items mas similares 
    most_similar_products_indices = np.argsort(-product_similarities)[1:6]

    # Obtenemos los nombres de los items mas similares
    most_similar_products = recomendacion.loc[most_similar_products_indices, 'title']

    return most_similar_products