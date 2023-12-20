import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# Menambahkan pilihan antara 'Regresi Linear' dan 'SVR' di sidebar
selected_option = st.sidebar.selectbox('Pilih jenis model:', ['Regresi Linear', 'SVR'])

# Menambahkan kolom pemilihan tanggal ke dalam sidebar
selected_date = st.sidebar.date_input('Pilih tanggal', datetime.date.today())

# Menambahkan tombol ke dalam sidebar
button_clicked = st.sidebar.button('Hitung')

# Menambahkan konten ke halaman utama
st.title('Aplikasi Prediksi Dollar Amerika Ke Rupiah')
st.write(f'Opsi yang dipilih: {selected_option}')
st.write(f'Tanggal yang dipilih: {selected_date}')

# Mengatur perhitungan berdasarkan pilihan
if button_clicked:
    if selected_option == 'Regresi Linear':
        df = pd.read_csv('C:/Users/User/PycharmProjects/pythonProject/github/data.csv', index_col=0, parse_dates=True,
                         skipinitialspace=True)
        last_date = df.index[-1].date()
        delta = selected_date - last_date  # Selisih tanggal
        st.write(f'Selisih hari antara {selected_date} dan hari ini adalah {delta.days} hari')
        selisih_hari = delta.days
        if selisih_hari <= len(df):
            df['Kursjuals'] = df['KursJual'].shift(selisih_hari)
            reversed_datas = df['KursJual'].iloc[::-1]  # Membalik urutan data
            df['Kursjuals'][:selisih_hari] = reversed_datas.iloc[:selisih_hari].values
            df['Kursbelis'] = df['KursBeli'].shift(selisih_hari)
            reversed_data = df['KursBeli'].iloc[::-1]  # Membalik urutan data
            df['Kursbelis'][:selisih_hari] = reversed_data.iloc[:selisih_hari].values
        else:
            print("Jumlah data yang ingin diisi melebihi jumlah total data dalam DataFrame.")
        X = np.array(df[['Kursjuals', 'Kursbelis']])
        y = np.array(df['KursTgh'])
        y = y.reshape(-1, 1)
        actual_10days_array = np.array(df['KursJual'])[-selisih_hari:]
        actual_11days_array = np.array(df['KursBeli'])[-selisih_hari:]
        df.drop(columns=['KursBeli', 'KursJual', 'KursTengah', 'KursJual2'], inplace=True)
        scalerX = MinMaxScaler()
        scalerX.fit(X)
        X = scalerX.transform(X)
        scalery = MinMaxScaler()
        scalery.fit(y)
        y = scalery.transform(y)
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)
        X_train_scaled = scalerX.fit_transform(X_train)
        X_test_scaled = scalerX.transform(X_test)
        from sklearn.linear_model import LinearRegression

        regressor = LinearRegression()
        regressor.fit(X_train_scaled, y_train)
        y_pred_test = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)
        y_pred_test2d = y_pred_test.reshape(-1, 1)
        y_test2d = y_test.reshape(-1, 1)
        nilai_prediksitest = scalery.inverse_transform(y_pred_test2d)
        nilai_test = scalery.inverse_transform(y_test2d)
        datanew = actual_10days_array.reshape(-1, 1)
        datanew1 = actual_11days_array.reshape(-1, 1)
        sc_Actualarray = MinMaxScaler()
        Actual10_scale = sc_Actualarray.fit_transform(datanew)
        Actual11_scale = sc_Actualarray.fit_transform(datanew1)
        features_for_prediction = np.concatenate((Actual10_scale, Actual11_scale), axis=1)
        prediction10d = regressor.predict(features_for_prediction)
        nilai_baru = sc_Actualarray.inverse_transform(prediction10d)
        from datetime import datetime, timedelta

        # Ambil tanggal terakhir dari indeks dataset
        last_date = df.index[-1]

        # Hitung tanggal 7 hari ke depan
        future_dates = [last_date + timedelta(days=i) for i in range(1, selisih_hari + 1)]
        print(type(df))
        # Buat indeks tanggal untuk tanggal-tanggal prediksi
        indeks_prediksi = pd.date_range(start=last_date + timedelta(days=1), periods=selisih_hari)
        Actual10_scale_reshaped = Actual10_scale.reshape(-1, 1)
        Actual11_scale_reshaped = Actual11_scale.reshape(-1, 1)
        features_for_prediction1 = np.concatenate((Actual10_scale_reshaped, Actual11_scale_reshaped), axis=1)

        # Lakukan prediksi untuk data masa depan (pastikan X_prediksi sesuai)
        prediksi_series = regressor.predict(features_for_prediction1)
        nilai_baru11 = sc_Actualarray.inverse_transform(
            prediksi_series)  # Ganti X_prediksi sesuai dengan data input yang benar
        nilai_baru11 = nilai_baru11.reshape(-1)

        df_prediksi = pd.DataFrame({
            'Kursjuals': actual_10days_array,
            'Kursbelis': actual_11days_array,
            'KursTgh': nilai_baru11
        }, index=future_dates)
        print(df_prediksi)
        df_combined = pd.concat([df, df_prediksi])

        # Tampilkan DataFrame hasil gabungan
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Ambil data untuk tahun 2023
        df_2023 = df_combined.loc['2023']

        # Tentukan indeks untuk 7 data terakhir
        indeks_7_terakhir = df_2023.index[-selisih_hari:]

        # Gabungkan data untuk memastikan indeks terhubung
        df_combineds = pd.DataFrame({'KursTghAsli': df_2023['KursTgh'], 'KursTgh7Terakhir': df_2023['KursTgh']})
        df_combineds['KursTgh7Terakhir'].loc[~df_combineds.index.isin(indeks_7_terakhir)] = np.nan

        # Visualisasi data asli (warna biru dan merah terhubung)
        plt.plot(df_combineds.index, df_combineds['KursTghAsli'], label='Kurs Tengah Asli', color='blue')

        # Visualisasi 7 data terakhir (warna merah)
        plt.plot(df_combineds.index, df_combineds['KursTgh7Terakhir'],
                 label='Kurs Tengah 7 Terakhir', color='red')

        # Format tanggal pada sumbu x menjadi bulan
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        # Tambahkan label sumbu dan legenda
        plt.xlabel('Bulan Tahun 2023')
        plt.ylabel('Nilai Kurs Tengah')
        plt.grid(True)
        plt.legend()

        # Tampilkan plot
        plt.show()
        st.pyplot(plt)
        print(df_2023)
        nilai_baru11_bulat = nilai_baru11.round(0)
        nilai_baru111 = nilai_baru11_bulat.astype(int)
        st.table({'Tanggal': future_dates, 'Kurs': nilai_baru111})

    elif selected_option == 'SVR':
        df = pd.read_csv('C:/Users/User/PycharmProjects/pythonProject/github/data.csv', index_col=0, parse_dates=True,
                         skipinitialspace=True)
        last_date = df.index[-1].date()
        delta = selected_date - last_date  # Selisih tanggal
        selisih_hari = delta.days
        if selisih_hari <= len(df):
            df['Kursjuals'] = df['KursJual'].shift(selisih_hari)
            reversed_datas = df['KursJual'].iloc[::-1]  # Membalik urutan data
            df['Kursjuals'][:selisih_hari] = reversed_datas.iloc[:selisih_hari].values
            df['Kursbelis'] = df['KursBeli'].shift(selisih_hari)
            reversed_data = df['KursBeli'].iloc[::-1]  # Membalik urutan data
            df['Kursbelis'][:selisih_hari] = reversed_data.iloc[:selisih_hari].values
        else:
            print("Jumlah data yang ingin diisi melebihi jumlah total data dalam DataFrame.")
        X = np.array(df[['Kursjuals', 'Kursbelis']])
        y = np.array(df['KursTgh'])
        y = y.reshape(-1, 1)
        actual_10days_array = np.array(df['KursJual'])[-selisih_hari:]
        actual_11days_array = np.array(df['KursBeli'])[-selisih_hari:]
        df.drop(columns=['KursBeli', 'KursJual', 'KursTengah', 'KursJual2'], inplace=True)
        scalerX = MinMaxScaler()
        scalerX.fit(X)
        X = scalerX.transform(X)
        scalery = MinMaxScaler()
        scalery.fit(y)
        y = scalery.transform(y)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)
        modellin = SVR(kernel='rbf', C=10, epsilon=0.001, gamma=0.01)
        modellin.fit(X_train, np.ravel(y_train))
        y_pred_testlin = modellin.predict(X_test)
        y_pred_trainlin = modellin.predict(X_train)
        # Hitung MSE (Mean Squared Error)
        datanew = actual_10days_array.reshape(-1, 1)
        datanew1 = actual_11days_array.reshape(-1, 1)
        sc_Actualarray = MinMaxScaler()
        Actual10_scale = sc_Actualarray.fit_transform(datanew)
        Actual11_scale = sc_Actualarray.fit_transform(datanew1)
        features_for_prediction = np.concatenate((Actual10_scale, Actual11_scale), axis=1)
        prediction10d1 = modellin.predict(features_for_prediction)
        prediction10d1 = prediction10d1.reshape(-1, 1)
        nilai_baru1 = sc_Actualarray.inverse_transform(prediction10d1)
        from datetime import datetime, timedelta

        # Ambil tanggal terakhir dari indeks dataset
        last_date = df.index[-1]

        # Hitung tanggal 7 hari ke depan
        future_dates = [last_date + timedelta(days=i) for i in range(1, selisih_hari + 1)]
        # Buat indeks tanggal untuk tanggal-tanggal prediksi
        indeks_prediksi = pd.date_range(start=last_date + timedelta(days=1), periods=selisih_hari)
        Actual10_scale_reshaped = Actual10_scale.reshape(-1, 1)
        Actual11_scale_reshaped = Actual11_scale.reshape(-1, 1)
        features_for_prediction1 = np.concatenate((Actual10_scale_reshaped, Actual11_scale_reshaped), axis=1)

        # Lakukan prediksi untuk data masa depan (pastikan X_prediksi sesuai)
        prediksi_series = modellin.predict(features_for_prediction1)
        prediksi_series = prediksi_series.reshape(-1, 1)
        nilai_baru11 = sc_Actualarray.inverse_transform(
            prediksi_series)  # Ganti X_prediksi sesuai dengan data input yang benar
        nilai_baru11 = nilai_baru11.reshape(-1)

        df_prediksi = pd.DataFrame({
            'Kursjuals': actual_10days_array,
            'Kursbelis': actual_11days_array,
            'KursTgh': nilai_baru11
        }, index=future_dates)

        df_combined = pd.concat([df, df_prediksi])
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Ambil data untuk tahun 2023
        df_2023 = df_combined.loc['2023']

        df_2023_september_onward = df_2023['2023-08-20':]

        # Tentukan indeks untuk 7 data terakhir
        indeks_7_terakhir = df_2023_september_onward.index[-7:]

        # Gabungkan data untuk memastikan indeks terhubung
        df_combined = pd.DataFrame({'KursTghAsli': df_2023_september_onward['KursTgh'],
                                    'KursTgh7Terakhir': df_2023_september_onward['KursTgh']})
        df_combined['KursTgh7Terakhir'].loc[~df_combined.index.isin(indeks_7_terakhir)] = np.nan

        # Visualisasi data asli (warna biru dan merah terhubung)
        plt.plot(df_combined.index, df_combined['KursTghAsli'], label='Kurs Tengah Asli', color='blue')

        # Visualisasi 7 data terakhir (warna merah)
        plt.plot(df_combined.index, df_combined['KursTgh7Terakhir'],
                 label='Prediksi', color='red')

        # Format tanggal pada sumbu x menjadi tanggal
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))  # Format tanggal, bulan, dan tahun

        # Tambahkan label sumbu dan legenda
        plt.xlabel('Tanggal September 2023')
        plt.ylabel('Model SVR Poly')
        plt.xticks(rotation=90)  # Rotasi label agar terlihat lebih baik
        plt.tight_layout()
        plt.legend()
        # Tampilkan plot
        plt.show()
        st.pyplot(plt)
        print(df_2023)
        nilai_baru11_bulat = nilai_baru11.round(0)
        nilai_baru111 = nilai_baru11_bulat.astype(int)
        st.table({'Tanggal': future_dates, 'Kurs': nilai_baru111})
