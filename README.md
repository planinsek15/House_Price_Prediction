# House_Price_Prediction

### **Cilj projekta**

Napovedati ceno hiše (spremenljivka `MEDV`) na podlagi različnih lastnosti nepremičnine (npr. število sob, kriminal v okolici, bližina rek, davki ipd.) z uporabo linearne regresije.

---

## ✅ 1. **Priprava podatkov**

- Uporabili smo podatkovni niz `HousingData.csv`, ki vsebuje različne značilnosti nepremičnin.

- Naložili smo podatke s knjižnico `pandas`.

- Manjkajoče vrednosti smo nadomestili z **aritmetično sredino** (`mean()`).

- Podatke smo razdelili na značilnosti (X) in ciljno spremenljivko (Y = `MEDV`).


---

## ✂️ 2. **Delitev podatkov**

- Podatke smo razdelili na **učno množico** (80 %) in **testno množico** (20 %) z uporabo `train_test_split`.


---

## ⚖️ 3. **Standardizacija podatkov**

- Značilnosti smo standardizirali z `StandardScaler`:

	- `fit_transform()` samo na učni množici.

    - `transform()` na testni množici.

- To je pomembno, saj linearna regresija deluje bolje pri normaliziranih podatkih.

---

## 📈 4. **Treniranje modela**

- Uporabili smo model `LinearRegression` iz knjižnice `sklearn.linear_model`.

- Model smo natrenirali na učni množici (`model.fit(X_train, Y_train)`).


---

## 🧪 5. **Evalvacija modela**

- Model smo testirali na testni množici z uporabo:

    - **MSE** (mean squared error)

    - **R² score** (koeficient determinacije)


Primer:
```
mse = mean_squared_error(Y_test, predictions) r2 = r2_score(Y_test, predictions)
`````

- `R²` nam pove, koliko variance v cenah hiš razloži naš model. Višja kot je vrednost (bližje 1), boljši je model.

---

## 🧮 6. **Napovedovanje cen**

- Model smo uporabili za napovedovanje cen hiš na testni množici.

- Prikazali smo napovedane cene z `print(predictions)`.


---

## 💾 7. **Shranjevanje modela (opcija)**

- Uporabili smo `joblib` za shranjevanje modela in standardizatorja.

    - `joblib.dump(model, 'model.pkl')`

    - `joblib.dump(scaler, 'scaler.pkl')`


---

## 📊 8. **Vizualizacija rezultatov (opcija)**

- Narisali smo graf primerjave med dejanskimi in napovedanimi cenami (scatter plot).

- Dobili občutek o tem, kako dobro model napoveduje.


---

## 🧠 Naučili smo se:

- Kako obdelati in pripraviti podatke za strojno učenje.

- Kako uporabiti linearno regresijo za napovedovanje.

- Kako oceniti kakovost modela s pomočjo MSE in R².

- Kako standardizacija vpliva na kakovost modela.

- Kako shraniti in ponovno uporabiti modele.

- Kako vizualizirati napake in napovedi.

---

## 🚀 Možnosti nadgradnje:

- Poskusi **drugačne modele** (npr. DecisionTreeRegressor, RandomForest, XGBoost).

- Dodaj **interaktivni uporabniški vmesnik** za vnos podatkov (npr. Streamlit).

- Izvedi **feature selection** (izberi le najpomembnejše značilnosti).

- Dodaj **vizualne analize** korelacij med značilnostmi (npr. `sns.heatmap`).
