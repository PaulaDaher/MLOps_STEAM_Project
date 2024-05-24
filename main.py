from fastapi import FastAPI, Query
import pandas as pd
import numpy as np



# Indicamos título y descripción de la API
app = FastAPI(title='PROYECTO INDIVIDUAL 1 - MLOps - Paula Daher',
            description='STEAM: API de datos y recomendaciones de videos juegos')

# Datasets
#df = pd.read_csv('movies_final.csv')
#df1 = pd.read_csv('movies_ml.csv')

# Función para reconocer el servidor local

from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory='Templates')

@app.get('/', tags=['Pagina Principal'])
async def read_root(request: Request):
    return templates.TemplateResponse('portada.html', {'request': request})

@app.get('/about/')
async def about():
    return {'PROYECTO INDIVIDUAL 1 - Machine Learning Operations - STEAM Simulation'}


# Cargamos los archivos parquet
df1 = pd.read_parquet('consulta1.parquet')
games = pd.read_parquet('games.parquet')
reviews = pd.read_parquet('reviews.parquet')
items = pd.read_parquet('items.parquet')


@app.get('/consulta1 - Developer', tags=['GET'])
def developer():
    """Ingrese el nombre de algún desorrollador y devuelve cantidad de items y porcentaje de contenido Free por año según la empresa empresa que infresó\n.
    """

    # Calcular la cantidad de items y el porcentaje de contenido gratuito por año y desarrollador
    df1['is_free'] = df1['price'] == 0.00
    grouped = df1.groupby(['release_date', 'developer']).agg(
        total_items=('content_id', 'count'),
        free_items=('is_free', 'sum')
    ).reset_index()
    grouped['free_percentage'] = (grouped['free_items'] / grouped['total_items']) * 100
    
    # organizar resultados en la estructura deseada
    result = []
    for _, row in grouped.iterrows():
        result.append({
            'Año': row['release_date'],
            'Cantidad de items': row['total_items'],
            'Contenido Free': f"{row['free_percentage']:.2f}%"
        })
    
    return result



@app.get('/consulta2 - Userdata', tags=['GET'])
def userdata(User_id: str):
    """Ingrese la id_del usuario y devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items\n.
    """
    # Selección de columnas
    game = games[['content_id', 'price']]
    
    # Filtro por usuario
    user = items[items['user_id'] == User_id]

    if user.empty:
        return {'Usuario': User_id, 'Dinero gastado': 0.0, 'Horas juego': 0, '% de recomendacion': 0.0}

    # Calcular el porcentaje de recomendación
    if reviews.empty:
        recomend = 0.0
    else:
        recomend = round(reviews['review_pos'].sum() / (reviews['review_pos'].sum() + reviews['review_neg'].sum()) * 100, 2)

    # Cantidad de juegos por usuario
    cantidad_items = items[items['user_id'] == User_id]

    # Unión de los datos
    cantidad_items = cantidad_items.merge(game, left_on='item_id', right_on='content_id', how='inner')
    if cantidad_items.empty:
        return {'Usuario': User_id, 'Dinero gastado': 0.0, 'Horas juego': 0, '% de recomendacion': recomend}

    cantidad_items = cantidad_items.groupby('user_id').agg({'playtime_forever': 'sum', 'price': 'sum'}).reset_index()

    if cantidad_items.empty:
        return {'Usuario': User_id, 'Dinero gastado': 0.0, 'Horas juego': 0, '% de recomendacion': recomend}

    # Variables:
    usuario = cantidad_items['user_id'].iloc[0]
    tiempo = cantidad_items['playtime_forever'].iloc[0]
    dinero = cantidad_items['price'].iloc[0]

    return {'Usuario': usuario, 'Dinero gastado': round(dinero, 2), 'Horas juego': tiempo, '% de recomendacion': recomend}