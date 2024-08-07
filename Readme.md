
# <h1 align=left> MACHINE LEARNING OPERATIONS (MLOps) </h1>
# <h3 align=left>**`POL DAJER (PAULA DAHER)`**</h3>

<p align="center">
<img src="image.png" height=300>
</p>

# <h3 align=left>**`PROJECT DESCRIPTION`**</h3>

In this project, we will go through all the stages of the lifecycle of a Machine Learning project, resulting in the development of an API deployed on Render, through which queries can be made to the records of a Steam (videogames) platform database. Additionally, a machine learning recommendation model for video games based on cosine similarity is developed, which can also be accessed through the API.


### The project is divided into two parts:

**Part I:** Data Engineering
It starts from scratch, quickly working as a Data Engineer with data collection and extraction from files, as well as their processing, transformation, and modeling.

**Part II:** Machine Learning
The model is created, the cleaned data is consumed, and it is trained under certain conditions. As a result, a video game recommendation system for Steam users is created, using MLOps techniques to ensure that the model and the API are scalable, reproducible, and maintainable.


</p>

# <h3 align=left>**`OBJECTIVES`**</h3>

**Data Transformations:** Read and clean the dataset, removing unnecessary columns to optimize performance, knowing that data maturity is low: nested data, raw type, no automated processes for updating new products, among other things.   
[Download link for the Datasets to which ETL was applied](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj)   
[Data Dictionary](https://docs.google.com/spreadsheets/d/1-t9HLzLHIGXvliq56UE_gMaWBVTPfrlTf2D9uAtLGrk/edit#gid=0)

**Feature Engineering:** Perform sentiment analysis on user reviews and create a new column 'sentiment_analysis'.

**API Development:** Implement an API with FastAPI that allows querying data and recommendations.

**Deployment:** Deploy the API on a web service to make it publicly accessible.

**Exploratory Data Analysis (EDA):** Explore and visualize the data to gain valuable insights.
[Article of interest](https://medium.com/swlh/introduction-to-exploratory-data-analysis-eda-d83424e47151)

**Machine Learning Model:**
Develop a recommendation system based on cosine similarity. 

# <h3 align=left>**`PROJECT STRUCTURE`**</h3>

- **Data:**  .parquet In the repository, you will find 5 parquet files, results of the ETL, based on which the querying and modeling functions operate.
- **Notebooks:** Jupyter notebooks for ETL and EDA.
- **main.py:** Main file of the API (queries and recommendation model).
- **requirements.txt:** List of project dependencies.
- **README.md:** Project description and guide.

# <h3 align=left>**`STEPS TAKEN`**</h3>

1. Data Transformations:  
The datasets were read in the correct format, and the data from the three databases used were cleaned and preprocessed. Unnecessary columns were removed, null values and nested data were handled, among other things.
The transformations are documented in the notebooks [etl_steam](https://github.com/PaulaDaher/Proyecto_MLOps_STEAM/blob/main/EDA_steam.ipynb), [etl_user_items](https://github.com/PaulaDaher/Proyecto_MLOps_STEAM/blob/main/ETL_user_items.ipynb), and [etl_user_reviews](https://github.com/PaulaDaher/Proyecto_MLOps_STEAM/blob/main/ETL_user_reviews.ipynb)

2. Feature Engineering:  
Sentiment analysis was applied to user reviews:
The 'sentiment_analysis' column was created with values 0 (negative), 1 (neutral), and 2 (positive) using the NLTK library: Sentiment analysis.
A value of 1 was assigned to absent reviews.

3. API Development:  
The following endpoints were implemented in the API using FastAPI:

- **developer(developer: str):** Number of items and percentage of Free content per year by developer.  
- **userdata(User_id: str):** Amount of money spent by the user, recommendation percentage, and number of items.  
- **UserForGenre(genre: str):** User with the most hours played for the given genre and accumulation of hours played by release year.  
- **best_developer_year(year: int):** Top 3 developers with the most user-recommended games for the given year.  
- **developer_reviews_analysis(developer: str):** Analysis of user reviews categorized by sentiment analysis (positive or negative).  
- **game_recommendation(product_id: str):** List of 5 recommended games similar to the entered one.  

4. Deployment:  
The API was deployed using Render to make it publicly accessible from any internet-connected device.

5. Exploratory Data Analysis (EDA):  
An exploratory data analysis was conducted to better understand the relationships between variables, detect outliers, and find interesting patterns.

6. Machine Learning Model:  
A recommendation system based on cosine similarity was developed:
- Input: Product ID.
- Output: List of 5 recommended games similar to the entered one.

7. Demonstration Video:  
A video was created demonstrating the functionality of the API queries and the trained ML model. The video shows how queries are made and briefly explains the project's deployment.

# <h3 align=left>**`TECH STACK`**</h3>
- Python: Main programming language.
- FastAPI: Framework for API development.
- Pandas: Data manipulation and analysis.
- Scikit-learn: Development of the recommendation model.
- NLTK: Sentiment analysis.
- Render: API deployment.


# <h3 align=left>**`API AND VIDEO`**</h3>
### For more details, you can check the code and notebooks in this repository. Additionally, you can access the API and watch the demonstration video here:
### [Access the API](https://proyecto-mlops-steam-4gux.onrender.com)
### [Watch the video](https://drive.google.com/drive/u/0/folders/1QQB0huSYZECBoJL1wp-4G3Q5RqbGJpaG) (Spanish language)




