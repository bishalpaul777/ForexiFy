from flask import Flask, render_template, request, redirect, jsonify, Response
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import KMeans
import seaborn as sns
import os
import json
from werkzeug.utils import secure_filename 
from flask import send_file
import plotly.express as px
import tempfile
from io import BytesIO
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global_dataset = None

@app.route('/')
def index():
    return render_template('index.html')

def upload_dataset(file):
    global global_dataset

    if file and file.filename.endswith(('.csv', '.xlsx')):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        try:
            dataset = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], filename), engine='openpyxl')

            global_dataset = dataset  
            return True 
        except UnicodeDecodeError:
            return False  

    return False 

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    dataset_uploaded = global_dataset is not None

    if request.method == 'POST':
      
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if upload_dataset(file):
            return redirect('/dashboard')  # Redirect to the dashboard after successful upload
        else:
            return "Error: Unable to decode the uploaded file. Please make sure it's in a compatible format."

    return render_template('dashboard.html', dataset_uploaded=dataset_uploaded, dataset=global_dataset)

@app.route('/sapo')
def sapo():
    return render_template('sapo.html')

@app.route('/get_sales_data')
def get_sales_data():
    global_dataset['Year'] = global_dataset['Order_Date'].dt.year
    global_dataset['Month'] = global_dataset['Order_Date'].dt.month  # Extract the month from the Order_Date column
    monthly_sales = global_dataset.groupby(['Year', 'Month'])['Amount'].sum().reset_index()

    # Prepare the data in a format that Chart.js can use
    chart_data = {
        "labels": [f"{year}-{month:02}" for year, month in zip(monthly_sales['Year'], monthly_sales['Month'])],
        "datasets": [
            {
                "label": "Total Sales Amount",
                "data": list(monthly_sales['Amount']),
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "borderColor": "rgba(75, 192, 192, 1)",
                "borderWidth": 1,
            }
        ],
    }

    return jsonify({"ChartData": chart_data})


@app.route('/get_profit_data')
def get_profit_data():
    global_dataset['Year'] = global_dataset['Order_Date'].dt.year
    global_dataset['Month'] = global_dataset['Order_Date'].dt.month  # Extract the month from the Order_Date column
    monthly_profit = global_dataset.groupby(['Year', 'Month'])['Profit'].sum().reset_index()

    # Prepare the data in a format that Chart.js can use
    chart_data = {
        "labels": [f"{year}-{month:02}" for year, month in zip(monthly_profit['Year'], monthly_profit['Month'])],
        "datasets": [
            {
                "label": "Total Profit Amount",
                "data": list(monthly_profit['Profit']),
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "borderColor": "rgba(75, 192, 192, 1)",
                "borderWidth": 1,
            }
        ],
    }

    return jsonify({"ChartData": chart_data})



@app.route('/topcustomers')
def topcustomers():
    return render_template('topcustomers.html')


@app.route('/get_top_customers_data', methods=['GET'])
def get_top_customers_data():
    global global_dataset

    if global_dataset is not None:
        customer_amount = global_dataset.groupby('CustomerName')['Amount'].sum()

        top_customers = customer_amount.sort_values(ascending=False).head(7)  # You can adjust the number as needed

        data = {
            "labels": top_customers.index.tolist(),
            "values": top_customers.values.tolist()
        }

        return jsonify({"message": "Success", "data": data})

    return jsonify({"message": "Data not available"})



@app.route('/topstates')
def topstates():
    return render_template('topstates.html')


@app.route('/get_top_states_data', methods=['GET'])
def get_top_states_data():
    global salesdata

    if global_dataset is not None:
        state_profit = global_dataset.groupby('State')['Profit'].sum()
        top_states = state_profit.sort_values(ascending=False).head(7)

        data = {
            "labels": top_states.index.tolist(),
            "values": top_states.values.tolist()
        }

        return jsonify({"message": "Success", "data": data})

    return jsonify({"message": "Data not available"})


@app.route('/topcategories')
def topcategories():
    return render_template('topcategories.html')

@app.route('/get_top_categories_data', methods=['GET'])
def get_top_categories_data():
    global global_dataset

    if global_dataset is not None:
        category_sales = global_dataset.groupby('Sub-Category')['Amount'].sum()

        top_categories = category_sales.sort_values(ascending=False).head(10)  # You can adjust the number as needed

        data = {
            "labels": top_categories.index.tolist(),
            "values": top_categories.values.tolist()
        }

        return jsonify({"message": "Success", "data": data})

    return jsonify({"message": "Data not available"})


@app.route('/payment')
def payment():
    return render_template('payment.html')

@app.route('/get_payment_modes_data', methods=['GET'])
def get_payment_modes_data():
    global global_dataset

    if global_dataset is not None:
        payment_modes_data = global_dataset['PaymentMode'].value_counts()

        data = {
            "labels": payment_modes_data.index.tolist(),
            "values": payment_modes_data.values.tolist()
        }

        return jsonify({"message": "Success", "data": data})

    return jsonify({"message": "Data not available"})


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

@app.route('/linear')
def linear():
    return render_template('linear.html')

@app.route('/demand')
def demand():
    return render_template('demand.html')

@app.route('/correlation')
def correlation():
    return render_template('correlation.html')

if __name__ == '__main__':
    app.run(debug=True)







