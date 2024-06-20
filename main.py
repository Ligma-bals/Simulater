from flask import Flask, render_template, jsonify, make_response
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import hashlib

app = Flask(__name__)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

factors = {
    'Pharma': ['Sales Price','MRP', 'Comp Price', 'Inventory levels', 'Seasonal', 'Expiry days', 'Demand Score','Govt regulations', 'Cost of Manufacturing'],
    'CPG': ['Sales Price','MRP', 'Comp Price', 'Inventory levels', 'Seasonal', 'Expiry days', 'Demand Score','Govt regulations','Campaign','Sales revenue','Cost of Manufacturing'],
    'Wholesale': ['Sales Price','MRP', 'Comp Price', 'Inventory levels', 'Seasonal', 'Offers & Discount', 'Demand Score','Customer segment','Campaign','Sales revenue','Inflation'],
    'Retail': ['Sales Price','MRP', 'Comp Price', 'Inventory levels', 'Seasonal', 'Offers & Discount', 'Demand Score','Customer segment','Campaign','Sales revenue','Inflation']
    # Add more industries and their factors as needed
}
influencing_factors = {
    'Pharma': ['MRP', 'Comp Price', 'Inventory levels', 'Seasonal', 'Expiry days', 'Demand Score','Govt regulations', 'Cost of Manufacturing'],
    'CPG': ['MRP', 'Comp Price', 'Inventory levels', 'Seasonal', 'Expiry days', 'Demand Score','Govt regulations','Campaign','Sales revenue','Cost of Manufacturing'],
    'Wholesale': ['MRP', 'Comp Price', 'Inventory levels', 'Seasonal', 'Offers & Discount', 'Demand Score','Customer segment','Campaign','Sales revenue','Inflation'],
    'Retail': ['MRP', 'Comp Price', 'Inventory levels', 'Seasonal', 'Offers & Discount', 'Demand Score','Customer segment','Campaign','Sales revenue','Inflation']
    # Add more industries and their factors as needed
}

target_variable = {
    'Pharma': 'Sales Price',
    'Wholesale': 'Sales Price',
    'CPG': 'Sales Price',
    'Retail': 'Sales Price'
    # Add more industries and their targets as needed
}

# Dictionary to store the last hash of each file and the corresponding coefficients
file_hashes = {}
coefficients = {}

def return_response(*Value):
    if len(Value) > 1:
        response = make_response(Value[0], Value[1])
    else:
        response = make_response(Value[0])
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

def calculate_file_hash(file_path):
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def get_ridge_coefficients(df, industry):
    X = df[influencing_factors[industry]]
    y = df[target_variable[industry]]
    model = Ridge()
    model.fit(X, y)
    coefs = model.coef_
    const_coef = model.intercept_
    return coefs, const_coef


@app.route('/')
def index():
    # Dynamically get industry names from the data directory
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    industries = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
    return render_template('index1.html', industries=industries, factors= factors, influencing_factors=influencing_factors)

@app.route('/data/<industry>', methods=['GET'])
def get_products(industry):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', industry)
    try:
        products = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.csv')]
        response = return_response(jsonify(products))
        return response
    except FileNotFoundError:
        response = return_response(jsonify({"error": f"Industry directory {industry} not found"}), 404)
        return response

@app.route('/data/<industry>/<product>', methods=['GET'])
def get_default_factors(industry, product):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', industry, f'{product}.csv')
    try:
        df = pd.read_csv(file_path)
        last_row = df.iloc[-1]
        factors_val = factors.get(industry, [])
        default_factors = [last_row[factor].item() if isinstance(last_row[factor], (np.integer, np.floating)) else last_row[factor] for factor in factors_val if factor in last_row]
        response = return_response(jsonify(default_factors))
        return response
    except FileNotFoundError:
        response = return_response(jsonify({"error": f"Product file {product}.csv not found in {industry}"}), 404)
        return response
    except Exception as e:
        response = return_response(jsonify({"error": str(e)}), 500)
        return response

@app.route('/coefficients/<industry>/<product>', methods=['GET','POST'])
def get_coefficients(industry, product):
    file_path = os.path.join(data_dir, industry, f'{product}.csv')
    try:
        current_hash = calculate_file_hash(file_path)
        
        if file_path not in file_hashes or file_hashes[file_path] != current_hash:
            df = pd.read_csv(file_path)
            coefs, const_coef = get_ridge_coefficients(df, industry)
            coefficients[(industry, product)] = dict(zip(influencing_factors[industry], coefs))
            coefficients[(industry, product)]['const'] = const_coef  # Add the constant term to the coefficients
            file_hashes[file_path] = current_hash
        
        response = return_response(jsonify(coefficients[(industry, product)]))
        return response
    except FileNotFoundError:
        response = return_response(jsonify({"error": f"Product file {product}.csv not found in {industry}"}), 404)
        return response
    except Exception as e:
        response = return_response(jsonify({"error": str(e)}), 500)
        return response
    

if __name__ == '__main__':
    app.run(debug=True)