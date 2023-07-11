
from flask import Flask, jsonify, request
import joblib
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

app = Flask(__name__)
app.config["DEBUG"] = True

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta de la base de datos
db_path = os.path.join(current_dir,'advertising.csv')

# Configuración de la base de datos
engine = create_engine("sqlite:///" + db_path)

Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Definición de la tabla de registros
class Record(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True)
    tv = Column(Float)
    radio = Column(Float)
    newspaper = Column(Float)
    sales = Column(Float)

# Construir la ruta del modelo
model_path = os.path.join(current_dir, 'data','advertising_model')

# Carga del modelo entrenado
model = joblib.load(model_path)

@app.route('/')
def home():
    return 'Welcome to the API'

# Ruta para realizar predicciones
@app.route('/v2/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tv = data['tv']
    radio = data['radio']
    newspaper = data['newspaper']

    # Realizar la predicción utilizando el modelo cargado
    prediction = model.predict([[tv, radio, newspaper]])

    return jsonify({'prediction': prediction.tolist()})

# Ruta para almacenar nuevos registros en la base de datos
@app.route('/v2/ingest_data', methods=['POST'])
def ingest_data():
    data = request.get_json()
    tv = data['tv']
    radio = data['radio']
    newspaper = data['newspaper']
    sales = data['sales']

    # Crear un nuevo registro en la base de datos
    record = Record(tv=tv, radio=radio, newspaper=newspaper, sales=sales)
    session.add(record)
    session.commit()

    return jsonify({'message': 'Record added successfully'})

# Ruta para reentrenar el modelo con los nuevos registros
@app.route('/v2/retrain', methods=['POST'])
def retrain():
    # Recuperar todos los registros de la base de datos
    records = session.query(Record).all()

    # Crear matrices X y y para reentrenar el modelo
    X = [[record.tv, record.radio, record.newspaper] for record in records]
    y = [record.sales for record in records]

    # Reentrenar el modelo con los nuevos datos
    model.fit(X, y)

    return jsonify({'message': 'Model retrained successfully'})

if __name__ == '__main__':
    app.run()



# -------------------------------------------------------



from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
import sqlite3


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

@app.route('/v2/predict', methods=['GET'])
def predict():
    try:
        model = pickle.load(open('data/advertising_model','rb'))

        tv = request.args.get('tv', None)
        radio = request.args.get('radio', None)
        newspaper = request.args.get('newspaper', None)

        if tv is None or radio is None or newspaper is None:
            return "Missing args, the input values are needed to predict"
        else:
            prediction = model.predict([[tv,radio,newspaper]])
            return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'
    except Exception as err:
        return jsonify({"status": 500})

@app.route('/v2/ingest_data', methods=['POST'])
def post_ingest_data():
    try:
        connection = sqlite3.connect("./data/advertising.db")
        data = request.get_json()
        data_df = pd.DataFrame(data)
        data_df.to_sql("advertising", con=connection, if_exists="append")

    except Exception as err:
        return jsonify({"status": 500})


@app.route('/v2/retrain', methods=['PUT'])
def put_retrain():
    try:
        model = pickle.load(open('./data/advertising_model','rb'))
        connection = sqlite3.connect("./data/advertising.db")
        crsr = connection.cursor()
        crsr.execute("SELECT * FROM advertising")
        data = crsr.fetchall()
        # Obtenemos los nombres de las columnas de la tabla
        names = [description[0] for description in crsr.description]
        data_df =  pd.DataFrame(data,columns=names)
        X = data_df.drop("sales", axis=1)
        y = data_df["sales"]
        model.fit(X, y)
        with open('./data/advertising_model', "wb") as f:
            pickle.dump(data, f)

    except Exception as err:
        return jsonify({"status": 500})

app.run()