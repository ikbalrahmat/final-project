from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import os
import pandas as pd
from decision_tree_analysis import analyze_decision_tree_imputation
from logistic_regression_analysis import analyze_logistic_regression_imputation
from random_forest_analysis import analyze_random_forest_imputation
from knn_analysis import analyze_knn_imputation
from naive_bayes_analysis import analyze_naive_bayes_imputation
from mean_analysis import analyze_mean_imputation
from median_analysis import analyze_median_imputation

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_UPLOAD_FOLDER'] = 'static/uploads'

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Dummy user
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Create a user_loader callback
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_UPLOAD_FOLDER'], exist_ok=True)

def get_target_column(filepath):
    if 'diabetes' in filepath:
        return 'Outcome'
    elif 'heart' in filepath:
        return 'HeartDisease'
    elif 'hepatitis' in filepath:
        return 'Class'
    elif 'titanic' in filepath:
        return 'Survived'
    elif 'adult' in filepath:
        return 'income'
    else:
        return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            user = User(id='admin')
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        session['filepath'] = filepath  # Simpan filepath ke sesi
        target_column = get_target_column(filepath)
        if not target_column:
            flash('Unsupported file type')
            return redirect(url_for('dashboard'))
        
        df = pd.read_csv(filepath)
        df = df.where(pd.notnull(df), None)  # Replace NaN with None
        missing_values = df[df.isnull().any(axis=1)]
        missing_count = df.isnull().sum().sum()
        missing_percentages = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        table_rows = missing_values.to_dict(orient='records')

        labels = list(missing_percentages.keys())
        data = list(missing_percentages.values())

        return render_template('show_data.html', titles=missing_values.columns.values, filepath=filepath, target_column=target_column, missing_percentages=missing_percentages, labels=labels, data=data, table_rows=table_rows)
    flash('File harus berformat CSV')
    return redirect(request.url)

@app.route('/impute', methods=['GET', 'POST'])
@login_required
def impute():
    filepath = session.get('filepath')
    if not filepath:
        flash('Harap unggah data terlebih dahulu sebelum melakukan imputasi.')
        return redirect(url_for('index'))

    target_column = get_target_column(filepath)
    
    knn_result = analyze_knn_imputation(filepath, app, target_column)
    decision_tree_result = analyze_decision_tree_imputation(filepath, app, target_column)
    logistic_regression_result = analyze_logistic_regression_imputation(filepath, app, target_column)
    random_forest_result = analyze_random_forest_imputation(filepath, app, target_column)
    naive_bayes_result = analyze_naive_bayes_imputation(filepath, app, target_column)
    mean_imputation_result = analyze_mean_imputation(filepath, app, target_column)
    median_imputation_result = analyze_median_imputation(filepath, app, target_column)
    
    datasets = {
        'knn': knn_result,
        'decision_tree': decision_tree_result,
        'logistic_regression': logistic_regression_result,
        'random_forest': random_forest_result,
        'naive_bayes': naive_bayes_result,
        'mean_imputation': mean_imputation_result,
        'median_imputation': median_imputation_result
    }

    return render_template('imputation_results.html', datasets=datasets, filepath=filepath, target_column=target_column)

@app.route('/final_imputation', methods=['POST'])
@login_required
def final_imputation():
    filepath = request.form['filepath']
    target_column = request.form['target_column']
    best_method = request.form['best_method']

    if not os.path.isfile(filepath):
        flash('File tidak ditemukan. Pastikan Anda telah mengunggah file yang benar.')
        return redirect(url_for('dashboard'))
    
    if best_method == 'knn':
        result = analyze_knn_imputation(filepath, app, target_column, final_imputation=True)
    elif best_method == 'decision_tree':
        result = analyze_decision_tree_imputation(filepath, app, target_column, final_imputation=True)
    elif best_method == 'logistic_regression':
        result = analyze_logistic_regression_imputation(filepath, app, target_column, final_imputation=True)
    elif best_method == 'random_forest':
        result = analyze_random_forest_imputation(filepath, app, target_column, final_imputation=True)
    elif best_method == 'naive_bayes':
        result = analyze_naive_bayes_imputation(filepath, app, target_column, final_imputation=True)
    elif best_method == 'mean_imputation':
        result = analyze_mean_imputation(filepath, app, target_column, final_imputation=True)
    elif best_method == 'median_imputation':
        result = analyze_median_imputation(filepath, app, target_column, final_imputation=True)
    else:
        result = {'error': 'Invalid method selected'}
    
    return render_template('final_imputation_results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
