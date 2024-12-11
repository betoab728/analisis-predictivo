import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

def get_engine():
    # Crear una conexión con SQLAlchemy usando pymysql
    return create_engine("mysql+pymysql://root:@localhost/campera_ganado")

def get_health_data():
    # Crear un motor SQLAlchemy
    engine = get_engine()

    # Consulta SQL
    query = """
    SELECT 
        g.idganado, 
        g.nombre, 
        g.sexo, 
        g.estadoReproductivo, 
        es.fecha, 
        es.peso, 
        es.temperatura, 
        tg.idtipoganado,
        tg.nombre AS tipoGanado
    FROM 
        ganado g
    INNER JOIN 
        estadosalud es ON g.idganado = es.idganado
    INNER JOIN 
        tipoganado tg ON g.idtipoganado = tg.idtipoganado
    """
    
    # Leer datos con pandas
    df = pd.read_sql(query, con=engine)

    # Convertir la columna fecha a datetime
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Calcular los días desde la primera fecha registrada
    df['dias'] = df.groupby('idganado')['fecha'].transform(lambda x: (x - x.min()).dt.days)

    return df
