import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score


import joblib

from sklearn.metrics import mean_absolute_error

#Preberemo csv file

result_of_csv_read = pd.read_csv('HousingData.csv')

"Pregled kaj vrne CSV"
#Preberemo podatke iz csv. Tukaj dobimo rezutlat funkcije read_csv()

#print(result_of_csv_read)


#############################################################################
"RAZISKAVA PODATKOV"
#############################################################################
#--------------------------------------------------------------------------------
# head() - vrne prvih pet vrstic | ce dodas stevilko v oklepaje ti vrne toliko vrstic

#print(result_of_csv_read.head())

#print(result_of_csv_read.head(10))

#--------------------------------------------------------------------------------
# info() - povzetek seznama vseh stolpcev z njihovimi podatkovnimi tipi in številom nenull vrednosti v vsakem stolpcu
#print(result_of_csv_read.info())

#print(result_of_csv_read.info(verbose = False))

#print(result_of_csv_read.info(verbose=True, show_counts=False))
#--------------------------------------------------------------------------------
# describe()

#print(result_of_csv_read.describe())

#############################################################################
"PRIRPAVA PODATKOV"
#############################################################################

#print(result_of_csv_read.isnull().sum()) # pogledamo katere vrednosti imajo NA/Null vrednosti


result_of_csv_read = result_of_csv_read.fillna(result_of_csv_read.mean(numeric_only=True))

#print(result_of_csv_read.isnull().sum()) # zdej mamo prazne podatke zafilane s povprecno vrednostjo stolpca


#Razdelimo podatke na X in pa Y os

X = result_of_csv_read.drop("MEDV", axis = 1)
Y = result_of_csv_read["MEDV"]

#Razdelimo podatke na učno enoto in testno množico

X_train, X_test, Y_train, Y_test = train_test_split (
    X, Y, test_size=0.2, random_state=42

)

#Standardizacija


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"Pomembno: Vedno uporabi fit_transform() samo na učni množici – testno le transform()!"

#############################################################################
"GRAJENJE IN TRENIRRANJE MODELOV"
#############################################################################

#Uporabili bomo Linearno Regresijo 

model = LinearRegression()

model.fit(X_train, Y_train)


#############################################################################
"EVALVACIJA MODELA"
#############################################################################

#Izmerili bomo natančnost

predictions = model.predict(X_test)

mse = mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

#Prikaz kako dobro model deluje
plt.figure(figsize=(10,6))
plt.scatter(Y_test, predictions, color='blue', alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--')
plt.xlabel("Dejanska cena (MEDV)")
plt.ylabel("Napovedana cena")
plt.title("Primerjava dejanske in napovedane cene")
plt.grid()
plt.show()



#############################################################################
"SHRANIMO MODEL"
#############################################################################


#joblib.dump(model, "house_price_model.pkl")

#joblib.dump(model, 'linear_model.pkl')
#joblib.dump(scaler, 'scaler.pkl')  # Ne pozabi shraniti tudi scalerja!



"Analiza značilnosti: Kaj vpliva na ceno?"

coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Koeficient'])
print(coefficients.sort_values(by="Koeficient", ascending=False))
#Tako vidiš, kateri faktorji najbolj vplivajo na ceno (pozitivno ali negativno).

"Napredna metrika (npr. RMSE ali MAE)"

mae = mean_absolute_error(Y_test, predictions)
print(f"\n\nMAE: {mae:.2f}")