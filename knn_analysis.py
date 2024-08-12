# import pandas as pd
# import numpy as np
# from sklearn.impute import KNNImputer
# from sklearn.metrics import mean_squared_error
# from flask import url_for
# import os

# def analyze_knn_imputation(filepath, app, target_column):
#     df = pd.read_csv(filepath)
#     df = df.dropna(subset=[target_column])  # Hapus baris yang nilai targetnya hilang

#     # Membuat dataset lengkap tanpa nilai yang hilang untuk evaluasi
#     complete_data = df.dropna()

#     # Pisahkan fitur dan target
#     features = complete_data.columns.drop(target_column)
#     X = complete_data[features]
#     y = complete_data[target_column]
#     X = pd.get_dummies(X, drop_first=True)

#     # Membuat nilai hilang secara acak sekitar 20%
#     np.random.seed(42)
#     X_missing = X.copy()
#     missing_mask = np.random.rand(*X_missing.shape) < 0.2
#     X_missing[missing_mask] = np.nan

#     # Imputasi data yang hilang menggunakan KNN
#     imputer = KNNImputer(n_neighbors=5)
#     data_imputed_knn = imputer.fit_transform(X_missing)

#     # Menghitung RMSE untuk hasil imputasi
#     numerical_cols = X.select_dtypes(include=[np.number]).columns
#     rmse = np.sqrt(mean_squared_error(X[numerical_cols], data_imputed_knn[:, :len(numerical_cols)]))

#     # Simpan dataset yang sudah diimputasi
#     df_imputed = pd.DataFrame(data_imputed_knn, columns=X.columns)
#     df_imputed[target_column] = y.values
#     imputed_csv_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'knn_imputed.csv')
#     df_imputed.to_csv(imputed_csv_path, index=False)

#     return {
#         'rmse': rmse,
#         'imputed_csv_path': url_for('static', filename='uploads/knn_imputed.csv'),
#         'imputed_data': df_imputed.to_html(classes="table table-striped", index=False)
#     }


# import pandas as pd
# import numpy as np
# from sklearn.impute import KNNImputer
# from sklearn.metrics import mean_squared_error
# from flask import url_for
# import os

# def analyze_knn_imputation(filepath, app, target_column):
#     # Membaca file CSV
#     df = pd.read_csv(filepath)
#     # Menghapus baris yang nilai targetnya hilang
#     df = df.dropna(subset=[target_column])

#     # Membuat dataset lengkap tanpa nilai yang hilang untuk evaluasi
#     complete_data = df.dropna()

#     # Memisahkan fitur dan target
#     features = complete_data.columns.drop(target_column)
#     X = complete_data[features]
#     y = complete_data[target_column]
    
#     # One-Hot Encoding untuk kolom kategorikal
#     X_encoded = pd.get_dummies(X, drop_first=True)

#     # Membuat nilai hilang secara acak sekitar 20%
#     np.random.seed(42)
#     X_missing = X_encoded.copy()
#     missing_mask = np.random.rand(*X_missing.shape) < 0.2
#     X_missing[missing_mask] = np.nan

#     # Fungsi untuk imputasi menggunakan KNN
#     def knn_impute(df):
#         imputer = KNNImputer(n_neighbors=5)
#         return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

#     # Imputasi data
#     data_imputed_knn = knn_impute(X_missing)

#     # Menghitung RMSE untuk hasil imputasi
#     original_with_missing = X_encoded[missing_mask].to_numpy()
#     imputed_values = data_imputed_knn[missing_mask].to_numpy()
#     rmse = np.sqrt(mean_squared_error(original_with_missing, imputed_values))

#     # Menyimpan dataset yang sudah diimputasi
#     df_imputed = pd.DataFrame(data_imputed_knn, columns=X_encoded.columns)
#     df_imputed[target_column] = y.values
#     imputed_csv_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'knn_imputed.csv')
#     df_imputed.to_csv(imputed_csv_path, index=False)

#     return {
#         'rmse': rmse,
#         'imputed_csv_path': url_for('static', filename='uploads/knn_imputed.csv'),
#         'imputed_excel_path': url_for('static', filename='uploads/knn_imputed.xlsx'),
#         'imputed_data': df_imputed.to_html(classes="table table-striped", index=False)
#     }


# import pandas as pd
# import numpy as np
# from sklearn.impute import KNNImputer
# from sklearn.metrics import mean_squared_error
# from flask import url_for
# import os

# def analyze_knn_imputation(filepath, app, target_column, final_imputation=False):
#     if not filepath or not filepath.lower().endswith('.csv'):
#         return {'error': 'File harus berformat CSV'}

#     print("Loading dataset...")
#     df = pd.read_csv(filepath)
#     print("Dataset loaded.")
    
#     if not final_imputation:
#         print(f"Dropping rows with missing target column: {target_column}")
#         df = df.dropna(subset=[target_column])  # Hapus baris yang nilai targetnya hilang
#         print("Rows dropped.")

#     # Membuat dataset lengkap tanpa nilai yang hilang untuk evaluasi
#     complete_data = df.dropna()
#     print("Complete data created.")

#     # Pisahkan fitur dan target
#     features = complete_data.columns.drop(target_column)
#     X = complete_data[features]
#     y = complete_data[target_column]
#     X_encoded = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding untuk kolom kategorikal
#     print("Features and target separated and encoded.")

#     if not final_imputation:
#         # Membuat nilai hilang secara acak sekitar 20%
#         np.random.seed(42)
#         X_missing = X_encoded.copy()
#         missing_mask = np.random.rand(*X_missing.shape) < 0.2
#         X_missing[missing_mask] = np.nan
#         print("Missing values introduced.")

#         # Imputasi data yang hilang menggunakan KNN
#         imputer = KNNImputer(n_neighbors=5)
#         data_imputed_knn = imputer.fit_transform(X_missing)
#         print("Missing values imputed.")

#         # Menghitung RMSE untuk hasil imputasi
#         original_with_missing = X_encoded[missing_mask]
#         imputed_values = pd.DataFrame(data_imputed_knn, columns=X_encoded.columns)[missing_mask]
#         rmse = np.sqrt(mean_squared_error(original_with_missing, imputed_values))
#         print(f"RMSE calculated: {rmse}")

#         # Simpan dataset yang sudah diimputasi
#         df_imputed = pd.DataFrame(data_imputed_knn, columns=X_encoded.columns)
#         df_imputed[target_column] = y.values
#         imputed_csv_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'knn_imputed.csv')
#         df_imputed.to_csv(imputed_csv_path, index=False)
#         imputed_excel_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'knn_imputed.xlsx')
#         df_imputed.to_excel(imputed_excel_path, index=False)
#         print(f"Imputed dataset saved to {imputed_csv_path} and {imputed_excel_path}.")
#     else:
#         # Imputasi data asli yang hilang menggunakan KNN
#         X_encoded_full = pd.get_dummies(df[features], drop_first=True)
#         imputer = KNNImputer(n_neighbors=5)
#         data_imputed_knn = imputer.fit_transform(X_encoded_full)
#         print("Final missing values imputed.")

#         # Tidak perlu menghitung RMSE untuk imputasi akhir
#         rmse = None

#         # Simpan dataset yang sudah diimputasi
#         df_imputed = pd.DataFrame(data_imputed_knn, columns=X_encoded_full.columns)
#         df_imputed[target_column] = df[target_column].values
#         imputed_csv_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'final_knn_imputed.csv')
#         df_imputed.to_csv(imputed_csv_path, index=False)
#         imputed_excel_path = os.path.join(app.config['STATIC_UPLOAD_FOLDER'], 'final_knn_imputed.xlsx')
#         df_imputed.to_excel(imputed_excel_path, index=False)
#         print(f"Final imputed dataset saved to {imputed_csv_path} and {imputed_excel_path}.")

#     return {
#         'rmse': rmse,
#         'method': 'KNN Imputer',
#         'imputed_csv_path': url_for('static', filename='uploads/final_knn_imputed.csv' if final_imputation else 'uploads/knn_imputed.csv'),
#         'imputed_excel_path': url_for('static', filename='uploads/final_knn_imputed.xlsx' if final_imputation else 'uploads/knn_imputed.xlsx'),
#         'imputed_data': df_imputed.to_html(classes="table table-striped", index=False)
#     }

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
