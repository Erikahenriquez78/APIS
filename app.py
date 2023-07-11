
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
model_path = os.path.join(current_dir, 'advertising_model')

# Carga del modelo entrenado
model = joblib.load(model_path)

@app.route('/')
def home():
    return 'Welcome to the API'

# Ruta para realizar predicciones
@app.route('/v2/predict', methods=['GET'])
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
