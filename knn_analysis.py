import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from flask import url_for
import os

def analyze_knn_imputation(filepath, app, target_column, final_imputation=False):
    if not filepath or not filepath.lower().endswith('.csv'):
        return {'error': 'File harus berformat CSV'}

    df = pd.read_csv(filepath)
    
    if not final_imputation:
        df = df.dropna(subset=[target_column])  # Hapus baris yang nilai targetnya hilang

    # Membuat dataset lengkap tanpa nilai yang hilang untuk evaluasi
    complete_data = df.dropna()

    # Pisahkan fitur dan target
    features = complete_data.columns.drop(target_column)
    X = complete_data[features]
    y = complete_data[target_column]
    X_encoded = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding untuk kolom kategorikal

    if not final_imputation:
        # Membuat nilai hilang secara acak sekitar 20%
        np.random.seed(42)
        X_missing = X_encoded.copy()
        missing_mask = np.random.rand(*X_missing.shape) < 0.2
        X_missing = X_missing.astype(float)  # Pastikan tipe data float
        X_missing[missing_mask] = np.nan

        # Imputasi data yang hilang menggunakan KNN
        imputer = KNNImputer(n_neighbors=5)
        data_imputed = imputer.fit_transform(X_missing)

        # Menghitung RMSE untuk hasil imputasi
        mask_flattened = missing_mask.flatten()
        original_with_missing = X_encoded.values.flatten()[mask_flattened]
        imputed_values = data_imputed.flatten()[mask_flattened]
        rmse = np.sqrt(mean_squared_error(original_with_missing, imputed_values))
        print(f"RMSE for KNN Imputation: {rmse}")

        # Simpan dataset yang sudah diimputasi
        df_imputed = pd.DataFrame(data_imputed, columns=X_encoded.columns)
        df_imputed[target_column] = y.values
        imputed_csv_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'knn_imputed.csv')
        df_imputed.to_csv(imputed_csv_path, index=False)
        imputed_excel_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'knn_imputed.xlsx')
        df_imputed.to_excel(imputed_excel_path, index=False)
    else:
        # Imputasi data asli yang hilang menggunakan KNN
        X_encoded_full = pd.get_dummies(df[features], drop_first=True)
        imputer = KNNImputer(n_neighbors=5)
        data_imputed = imputer.fit_transform(X_encoded_full)

        # Tidak perlu menghitung RMSE untuk imputasi akhir
        rmse = None

        # Simpan dataset yang sudah diimputasi
        df_imputed = pd.DataFrame(data_imputed, columns=X_encoded_full.columns)
        df_imputed[target_column] = df[target_column].values
        imputed_csv_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'final_knn_imputed.csv')
        df_imputed.to_csv(imputed_csv_path, index=False)
        imputed_excel_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'final_knn_imputed.xlsx')
        df_imputed.to_excel(imputed_excel_path, index=False)

    return {
        'rmse': rmse,
        'method': 'KNN Imputation',
        'imputed_csv_path': url_for('static', filename='uploads/final_knn_imputed.csv' if final_imputation else 'uploads/knn_imputed.csv'),
        'imputed_excel_path': url_for('static', filename='uploads/final_knn_imputed.xlsx' if final_imputation else 'uploads/knn_imputed.xlsx'),
        'imputed_data': df_imputed.to_html(classes="table table-striped", index=False)
    }
