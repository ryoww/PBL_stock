from flask import Flask, request, jsonify, make_response
from flask_marshmallow import Marshmallow
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "stock"),
    'password': os.getenv("DB_PASSWORD", "ryotaro1212"),
    'database': os.getenv("DB_NAME", "stock")
}

# Flask setup
app = Flask(__name__)
ma = Marshmallow(app)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Marshmallow Schemas
class PostRowDataSchema(ma.Schema):
    class Meta:
        fields = ("stock_name", "stock_code", "datetime", "content")

class PostMLDataSchema(ma.Schema):
    class Meta:
        fields = ("despair", "optimism", "concern", "excitement", "stability")

class UpdateColumnDataSchema(ma.Schema):
    class Meta:
        fields = ("date", "column_name", "update_value", "stock_code")

post_row_data_schema = PostRowDataSchema()
post_ml_data_schema = PostMLDataSchema()
update_column_data_schema = UpdateColumnDataSchema()

# Helper function to connect to the database
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        raise

# Function to convert timedelta to string
def timedelta_to_str(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# Routes
@app.route("/row_data", methods=["POST"])
def add_row_data():
    rowdata = request.get_json()
    errors = post_row_data_schema.validate(rowdata)
    if errors:
        return make_response(jsonify(errors), 400)
    data = post_row_data_schema.load(rowdata)

    logger.info(f"Adding data: {data}")

    try:
        datetime_obj = datetime.strptime(data['datetime'], "%Y-%m-%d %H:%M:%S")
        date = datetime_obj.date()
        time = datetime_obj.time()
    except ValueError:
        return make_response("Invalid datetime format", 400)

    query = """
    INSERT INTO stock_dataset (stock_name, stock_code, date, time, content)
    VALUES (%s, %s, %s, %s, %s)
    """

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(query, (data['stock_name'], data['stock_code'], date, time, data['content']))
        connection.commit()
        return make_response("Data added successfully", 201)
    except Error as e:
        logger.error(f"Error adding data: {e}")
        return make_response("Failed to add data", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/row_data/<int:id>", methods=["GET"])
def get_row_data(id):
    logger.info(f"Fetching data for ID: {id}")

    query = """
    SELECT id, stock_name, stock_code, date, time, content
    FROM stock_dataset
    WHERE id = %s
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query, (id,))
        row = cursor.fetchone()
        if row:
            row['date'] = row['date'].isoformat()
            row['time'] = timedelta_to_str(row['time']) if isinstance(row['time'], timedelta) else row['time'].isoformat()
            return jsonify(row)
        else:
            return make_response("Data not found", 404)
    except Error as e:
        logger.error(f"Error fetching data: {e}")
        return make_response("Failed to fetch data", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/get_len", methods=["GET"])
def get_len():
    logger.info("Getting the max id of the table")

    query = "SELECT MAX(id) AS max_id FROM stock_dataset"

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query)
        result = cursor.fetchone()
        return jsonify(result)
    except Error as e:
        logger.error(f"Error fetching max id: {e}")
        return make_response("Failed to get max id", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/ml_data/<int:id>", methods=["POST"])
def post_ml_data(id):
    ml_data = request.get_json()
    errors = post_ml_data_schema.validate(ml_data)
    if errors:
        return make_response(jsonify(errors), 400)
    data = post_ml_data_schema.load(ml_data)

    logger.info(f"Posting ML data for ID: {id}")

    query = """
    UPDATE stock_dataset
    SET despair = %s, optimism = %s, concern = %s, excitement = %s, stability = %s
    WHERE id = %s
    """

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(query, (data['despair'], data['optimism'], data['concern'], data['excitement'], data['stability'], id))
        connection.commit()
        if cursor.rowcount == 0:
            return make_response("Data not found", 404)
        return make_response("ML data updated successfully", 200)
    except Error as e:
        logger.error(f"Error updating ML data: {e}")
        return make_response("Failed to update ML data", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/ml_data/<string:stock_code>", methods=["GET"])
def get_ml_data(stock_code):
    logger.info(f"Fetching ML data for stock_code: {stock_code}")

    query = """
    SELECT date, time, despair, optimism, concern, excitement, stability, value, vix, SP_500, NY_Dow
    FROM stock_dataset
    WHERE stock_code = %s
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query, (stock_code,))
        rows = cursor.fetchall()
        for row in rows:
            row['date'] = row['date'].isoformat()
            row['time'] = timedelta_to_str(row['time']) if isinstance(row['time'], timedelta) else row['time'].isoformat()
        return jsonify(rows)
    except Error as e:
        logger.error(f"Error fetching ML data: {e}")
        return make_response("Failed to fetch ML data", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/update_column", methods=["POST"])
def update_column():
    update_data = request.get_json()
    errors = update_column_data_schema.validate(update_data)
    if errors:
        return make_response(jsonify(errors), 400)
    data = update_column_data_schema.load(update_data)

    logger.info(f"Updating column: {data['column_name']} on date: {data['date']}")

    query = f"""
    UPDATE stock_dataset
    SET {data['column_name']} = %s
    WHERE date = %s
    """

    if data.get('stock_code'):
        query += " AND stock_code = %s"

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        if data.get('stock_code'):
            cursor.execute(query, (data['update_value'], data['date'], data['stock_code']))
        else:
            cursor.execute(query, (data['update_value'], data['date']))
        connection.commit()
        if cursor.rowcount == 0:
            return make_response("No record found for the given date and stock_code", 404)
        return make_response("Column updated successfully", 200)
    except Error as e:
        logger.error(f"Error updating column: {e}")
        return make_response("Failed to update column", 500)
    finally:
        cursor.close()
        connection.close()

@app.route("/getdays/<string:stock_code>", methods=["GET"])
def get_days(stock_code):
    logger.info(f"Fetching datetime for stock_code: {stock_code}")

    query = """
    SELECT date, time
    FROM stock_dataset
    WHERE stock_code = %s
    """

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute(query, (stock_code,))
        rows = cursor.fetchall()
        for row in rows:
            row['date'] = row['date'].isoformat()
            row['time'] = timedelta_to_str(row['time']) if isinstance(row['time'], timedelta) else row['time'].isoformat()
        return jsonify(rows)
    except Error as e:
        logger.error(f"Error fetching datetime: {e}")
        return make_response("Failed to fetch datetime", 500)
    finally:
        cursor.close()
        connection.close()

# Run the server
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8999)
