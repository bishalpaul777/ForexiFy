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
import plotly.graph_objs as go 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

            print("Column Names in Dataset:", dataset.columns.tolist())

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

@app.route('/get_analysis_data')
def get_analysis_data():
    global_dataset['Order_Date'] = pd.to_datetime(global_dataset['Order_Date'])  # Convert to datetime if not already
    monthly_sales = global_dataset.set_index('Order_Date').resample('M')['Amount'].sum()

    # Prepare the data in a format that Plotly can use
    chart_data = {
        "labels": [str(date) for date in monthly_sales.index],
        "datasets": [
            {
                "label": "Total Sales Amount",
                "data": monthly_sales.values.tolist(),
            }
        ],
    }

    return jsonify({"ChartData": chart_data})



@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

@app.route('/create_forecast_chart', methods=['POST'])
def create_forecast_chart():
    amount_series = global_dataset['Amount']
    ts_model = ExponentialSmoothing(amount_series, seasonal='add', seasonal_periods=12)
    ts_fit = ts_model.fit()
    forecast_2022 = ts_fit.forecast(steps=12)
    forecast_2023 = ts_fit.forecast(steps=12)
    forecast_2024 = ts_fit.forecast(steps=12)
    forecast_2025 = ts_fit.forecast(steps=12)

    date_range_2022 = pd.date_range(start='2022-01-01', periods=12, freq='M')
    date_range_2023 = pd.date_range(start='2023-01-01', periods=12, freq='M')
    date_range_2024 = pd.date_range(start='2024-01-01', periods=12, freq='M')
    date_range_2025 = pd.date_range(start='2025-01-01', periods=12, freq='M')

    # Create the chart
    plt.figure(figsize=(12, 6))
    plt.plot(date_range_2022, amount_series[-24:-12], label='Actual Data (2022)', color='green', marker='o')
    plt.plot(date_range_2023, forecast_2023, label='Forecast (2023)', color='red', linestyle='dashed', marker='s')
    plt.plot(date_range_2024, forecast_2024, label='Forecast (2024)', color='orange', linestyle='dashed', marker='^')
    plt.plot(date_range_2025, forecast_2025, label='Forecast (2025)', color='blue', linestyle='dashed', marker='v')

    for x, y in zip(date_range_2023, forecast_2023):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', color='black')
    for x, y in zip(date_range_2024, forecast_2024):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', color='black')
    for x, y in zip(date_range_2025, forecast_2025):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', color='black')

    plt.title('Time Series Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the chart as an image in memory
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image as base64
    chart_image = base64.b64encode(buffer.read()).decode()

    # Calculate and store the 2025 forecast data
    forecast_2025 = ts_fit.forecast(steps=12)

    return jsonify({"chart_image": chart_image, "chart_image2025": chart_image, "forecast_2025": forecast_2025.tolist()})




@app.route('/linear')
def linear():
    return render_template('linear.html')

import matplotlib.pyplot as plt
import base64
from io import BytesIO

@app.route('/linear_regression_plot', methods=['GET'])
def linear_regression_plot():
    X = global_dataset[['Amount']]
    y = global_dataset['Profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Create the linear regression plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression')
    plt.title('Linear Regression')
    plt.xlabel('Amount')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image in memory
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image as base64
    plot_image = base64.b64encode(buffer.read()).decode()

    return jsonify({"plot_image": plot_image})


@app.route('/demand')
def demand():
    return render_template('demand.html')

@app.route('/get_demand_prediction_data')
def get_demand_prediction_data():
    global global_dataset

    if global_dataset is not None:
        # Clustering for Demand Prediction
        X_clustering = global_dataset[['Amount', 'Profit']]
        kmeans_model = KMeans(n_clusters=5, random_state=42)
        kmeans_model.fit(X_clustering)
        cluster_labels = kmeans_model.labels_
        global_dataset['Cluster'] = cluster_labels

        # Create a bar plot for cluster means
        palette = sns.color_palette("Set1", n_colors=len(global_dataset['Cluster'].unique()))
        cluster_means = global_dataset.groupby('Cluster').mean()

        plt.figure(figsize=(10, 6))
        cluster_means[['Amount', 'Profit']].plot(kind='bar', color=palette)
        plt.title('Clustering for Demand Prediction')
        plt.xlabel('Cluster')
        plt.ylabel('Mean Value')
        plt.legend(title='Feature', loc='upper right')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
        print(cluster_means)

        # Prepare the data to be sent as JSON
        demand_prediction_data = {
            "labels": cluster_means.index.tolist(),
            "values": cluster_means['Amount'].tolist(),  # You can choose 'Amount' or 'Profit' as per your preference
        }

        return jsonify(demand_prediction_data)

    return jsonify({"message": "Data not available"})



@app.route('/correlation')
def correlation():
    return render_template('correlation.html')

@app.route('/get_correlation_data')
def get_correlation_data():
    # Calculate the correlation matrix
    correlation_matrix = global_dataset.corr(numeric_only=True)

    # Convert the correlation matrix to a JSON-compatible format
    correlation_data = {
        'labels': correlation_matrix.columns.tolist(),
        'data': correlation_matrix.values.tolist()
    }

    return jsonify({'correlationData': correlation_data})

if __name__ == '__main__':
    app.run(debug=True)
