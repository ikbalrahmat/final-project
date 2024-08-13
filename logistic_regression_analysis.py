import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from flask import url_for
import os

def logistic_regression_impute(df, original_df):
    df_imputed = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            df_notnull = original_df[original_df[col].notnull()]
            df_null = df[df[col].isnull()]

            X_train = df_notnull.drop(columns=[col])
            y_train = df_notnull[col]
            X_test = df_null.drop(columns=[col])

            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)

            df_imputed.loc[df_null.index, col] = model.predict(X_test)

    return df_imputed

def analyze_logistic_regression_imputation(filepath, app, target_column, final_imputation=False):
    # Membaca file CSV
    df = pd.read_csv(filepath)
    # Hapus baris yang nilai targetnya hilang jika bukan imputasi akhir
    if not final_imputation:
        df = df.dropna(subset=[target_column])

    # Membuat dataset lengkap tanpa nilai yang hilang untuk evaluasi
    complete_data = df.dropna()

    # Memisahkan fitur dan target
    features = complete_data.columns.drop(target_column)
    X = complete_data[features]
    y = complete_data[target_column]

    # One-Hot Encoding untuk kolom kategorikal
    X_encoded = pd.get_dummies(X, drop_first=True)

    if not final_imputation:
        # Membuat nilai hilang secara acak sekitar 20%
        np.random.seed(42)
        X_missing = X_encoded.copy()
        missing_mask = np.random.rand(*X_missing.shape) < 0.2
        X_missing[missing_mask] = np.nan

        # Imputasi awal menggunakan mean
        imputer = SimpleImputer(strategy='mean')
        X_initial_imputed = imputer.fit_transform(X_missing)

        # Imputasi data yang hilang menggunakan Logistic Regression
        data_imputed_lr = logistic_regression_impute(pd.DataFrame(X_initial_imputed, columns=X_encoded.columns), X_encoded)

        # Menghitung RMSE untuk hasil imputasi
        original_with_missing = X_encoded[missing_mask].to_numpy()
        imputed_values = data_imputed_lr[missing_mask].to_numpy()
        rmse = np.sqrt(mean_squared_error(original_with_missing, imputed_values))

        # Menyimpan dataset yang sudah diimputasi
        df_imputed = pd.DataFrame(data_imputed_lr, columns=X_encoded.columns)
        df_imputed[target_column] = y.values
        imputed_csv_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'logistic_regression_imputed.csv')
        df_imputed.to_csv(imputed_csv_path, index=False)
        imputed_excel_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'logistic_regression_imputed.xlsx')
        df_imputed.to_excel(imputed_excel_path, index=False)
    else:
        # Imputasi data asli yang hilang menggunakan Logistic Regression
        X_encoded_full = pd.get_dummies(df[features], drop_first=True)
        imputer = SimpleImputer(strategy='mean')
        X_initial_imputed = imputer.fit_transform(X_encoded_full)

        data_imputed_lr = logistic_regression_impute(pd.DataFrame(X_initial_imputed, columns=X_encoded_full.columns), X_encoded_full)

        # Tidak perlu menghitung RMSE untuk imputasi akhir
        rmse = None

        # Menyimpan dataset yang sudah diimputasi
        df_imputed = pd.DataFrame(data_imputed_lr, columns=X_encoded_full.columns)
        df_imputed[target_column] = df[target_column].values
        imputed_csv_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'final_logistic_regression_imputed.csv')
        df_imputed.to_csv(imputed_csv_path, index=False)
        imputed_excel_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'final_logistic_regression_imputed.xlsx')
        df_imputed.to_excel(imputed_excel_path, index=False)

    return {
        'rmse': rmse,
        'method': 'Logistic Regression',
        'imputed_csv_path': url_for('static', filename='uploads/final_logistic_regression_imputed.csv' if final_imputation else 'uploads/logistic_regression_imputed.csv'),
        'imputed_excel_path': url_for('static', filename='uploads/final_logistic_regression_imputed.xlsx' if final_imputation else 'uploads/logistic_regression_imputed.xlsx'),
        'imputed_data': df_imputed.to_html(classes="table table-striped", index=False)
    }

