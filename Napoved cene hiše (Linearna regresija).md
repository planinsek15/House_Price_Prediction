Tukaj se uporabi "scikit-learn" za napoved cen hiš. 

Tehnike, ki se bodo uporabile so: Linearna regresija, podatkovna obdelava(pandas, matplotlib), metrike uspešnosti. 

Iz tukaj naj bi se pobiralo podatke "https://www.kaggle.com/datasets/altavish/boston-housing-dataset". Kasneje lahko narediš tudi za nepremičnine.net. 

- KAJ JE "scikit-learn"?
Najbolj uporabna knjižnica za učenje strojev (ML - machine learning) v Pythonu. Praktično dobiš neke metrike in pa grafe s katerimi ponazoriš neke podatke. 

Spletna stran te knjižnice:
![[Pasted image 20250518144038.png]]

Potek izdealve: 
1. **Priprava okolja** - lahko uporabiš Google Colab, kjer je bolj pregledna koda. Uporabiš Python jezik. inštalacija knjižnic (`pandas`, `numpy`, `matplotlib`, `scikit-learn`)
2. **Pridobitev podatkov** - podatki se nahajajo v CSV datoteki ter iz nje bo potrebno prebrati podatke.  
3. **Cilj:** Razumeti strukturo podatkov.

	Koraki:
	
	- Naloži podatke z `pandas.read_csv()`
	    
	- Uporabi `.head()`, `.info()`, `.describe()` za osnovne informacije
	    
	- Nariši grafe:
	    
	    - Histogrami (`df.hist()`)
	    
	    - Korelacijska matrika (`df.corr()`)
	    
	    - Scatter plot med npr. “površina” in “cena”

4. **Pripravi podatke - Koraki:**

	- Ugotovi manjkajoče vrednosti in jih odpravi (`df.isnull().sum()`)
	
	- Razdeli podatke na **X (vhodni podatki)** in **y (izhod – cena)**
	
	- Standardiziraj vrednosti (npr. `StandardScaler()` iz `sklearn.preprocessing`)
	
	- Razdeli na **trening in testni set** (`train_test_split`)

5. Zgradi in treniraj model - Uporabi linearno regresijo.
6. Evalvacija modela - Izmeri natančnost.
7. Shrani model (bonus)
8. Napiši poročilo (zase)

*********************************************************
**PRIPRAVA OKOLJA** 
""""""""""""""""""""""
Inštalirali smo jupyter notebook (pip install jupyter notebook) ter ostale knjižnice (pip install pandas numpy matplotlib scikit-learn).

Delal bom v VS Code ker sem bolje navajen in tudi bolj sem seznanjen kot z Google Colab, čeprav je tam mogoče bolj pregledno. 

*********************************************************
**PRIDOBITEV PODATKOV**
""""""""""""""""""""""""""""

Ročno sem pridobil CSV datoteko in jo shranil pod ta projekt (C:\Users\david\Desktop\MYGoals\MaterOfAI\Master of AI\Zacetniski_nivo\Napoved_cene_hise).  


*********************************************************
**Razišči podatke (Exploratory Data Analysis – EDA)**
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Za začetek sem pognal to kodo: 
```
import pandas as pd

#Preberemo csv file

result_of_csv_read = pd.read_csv('HousingData.csv')

print(result_of_csv_read)
```

S pomočjo knjižnice pandas sem prebral podatke iz CSV datoteke in pogledal kaj dobim kot rezultat. 

Rezultat je prikazan spodaj na sliki. Vidimo, da so podatki lepo urejeni po vsako vrstico posebej in tudi prikazani imamo koliko je vrstic in pa stolpcev. 

![[Pasted image 20250518165546.png]]

Sedaj bom testiral funkcije `.head()`, `.info()`, `.describe()`:

- **head()** - to je funkcija, ki nam prikaže določeno število vrstic, če vstavimo številko v oklepaje. Če številke ne navedemo nam vrne prvih pet vrstic. Vedno pa prebere vrstice od začetka. 

Se pravi mi smo dobili lepo izpisane vrstice in s pomočjo head() funkcije lahko izpišemo določeno število vrstic, da ne rabimo vedno prikazati vseh. Sej ti tudi sam program, ko ga poženeš ne izpiše vseh 500 vrstic, ampak ti samo par začetnih in par končnih. 

Prvi rezultat je defoltni, drugi rezultat je, ko sem vstavil številko 10.  
![[Pasted image 20250518174627.png]]

```
import pandas as pd 

result_of_csv_read = pd.read_csv('HousingData.csv')

print(result_of_csv_read.head())
print(result_of_csv_read.head(10))
```

- **info()** - kot vidimo na spodnji sliki, nam ta funkcija vrne povzetek seznama vseh stolpcev z njihovimi podatkovnimi tipi in številom nenull vrednosti v vsakem stolpcu. 

![[Pasted image 20250523071724.png]]

```
import pandas as pd 

result_of_csv_read = pd.read_csv('HousingData.csv')

print(result_of_csv_read.info())`
```

Imamo tudi neke dodatne parametre, ki jih lahko dodamo: 

Če nastavimo verbose na False nam ne izpiše tabele ampak poda krajši opis vsega kar imamo. To nam pride prav v primeru če imamo na tisoče vrstic. 

```
print(result_of_csv_read.info(verbose = False))
```

![[Pasted image 20250523082242.png]]

Če pa nastavimo verbose na True, se bo izpisala tabela in pa show_counters damo na False se ne bo izpisalo število vrstic, ki imajo nonull vrednosti. 

```
print(result_of_csv_read.info(verbose=True, show_counts=False))
```


![[Pasted image 20250523082415.png]]


- describe() - Metoda describe() vrne opis podatkov v podatkovnem okviru (DataFrame).

  Če podatkovni okvir vsebuje numerične podatke, opis vsebuje te podatke za vsak stolpec:

  count - Število nepraznih vrednosti.
  mean - Povprečna (aritmetična) vrednost.
  std - Standardni odklon.
  min - Najmanjša vrednost.
  25 % - 25 % percentil*.
  50 % - 50 % percentil*.
  75 % - 75 % percentil*.
  max - Najvišja vrednost.

![[Pasted image 20250523084329.png]]

*********************************************************

**PRIPRAVA PODATKOV**
"""""""""""""""""""""""""

Cilje je da očistimo in popravimo podatke za model. 

Potrebno je: 

- Ugotovi manjkajoče vrednosti in jih odpravi (`df.isnull().sum()`)

- Razdeli podatke na **X (vhodni podatki)** in **y (izhod – cena)**

- Standardiziraj vrednosti (npr. `StandardScaler()` iz `sklearn.preprocessing`)

- Razdeli na **trening in testni set** (`train_test_split`)


1. Odprava manjkajočih vrednosti - df.isnull().sum()
	- Ta funkcija nam omogoča prešteti, koliko podatkov ima NULL vrednost

```
import pandas as pd 

result_of_csv_read = pd.read_csv('HousingData.csv')

print(result_of_csv_read.isnull().sum())

```

![[Pasted image 20250523085629.png]]

2. Razdelimo podatke na **X (vhodni podatki)** in **y (izhod – cena)**
## Razlaga stolpcev (podatki iz Bostonskega stanovanjskega sklopa):

|Stolpec|Pomen|
|---|---|
|**CRIM**|Kriminalna stopnja po naselju|
|**ZN**|Procent stanovanjskih con za stanovanja > 25.000 ft²|
|**INDUS**|Procent poslovnih območij (nekmetijska raba)|
|**CHAS**|Meja s Charles River (1 = da, 0 = ne)|
|**NOX**|Onesnaženost zraka (dušikovi oksidi)|
|**RM**|Povprečno število sob na stanovanjsko enoto|
|**AGE**|Procent stanovanj, zgrajenih pred 1940|
|**DIS**|Razdalja do petih pomembnih delovnih središč v Bostonu|
|**RAD**|Indeks dostopa do avtocest|
|**TAX**|Davčna stopnja na nepremičnine|
|**PTRATIO**|Razmerje učencev na učitelja po občini|
|**B**|1000(Bk - 0.63)^2, kjer je Bk delež temnopoltih prebivalcev|
|**LSTAT**|Procent nižjega socialno-ekonomskega sloja|
|**MEDV**|🔥 Mediana vrednosti hiš v tisočih dolarjev (to je cilj – `y`)|
Ker želimo poizvedeti kako se spreminjajo cene hiš, bomo vzeli podatek MEDV. 

Če želimo poizvedeti kateri podatki najbolje korelirajo med sabo moramo narediti korelacijsko matriko (heatmap): 

````
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

result_of_csv_read = pd.read_csv('HousingData.csv')

corr_matrix = result_of_csv_read.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")

plt.show()

````

Tole je pa rezultat kode:
![[Pasted image 20250523094630.png]]

### Najbolj pomembne povezave s `MEDV`:

|Spremenljivka|Korelacija z `MEDV`|Pomen|
|---|---|---|
|**RM**|**+0.70**|Več sob → višja cena|
|**LSTAT**|**-0.72**|Več revnih → nižja cena|
|**PTRATIO**|-0.51|Več učencev na učitelja → nižja cena|
|**INDUS**|-0.48|Več industrije → nižja cena|
|**TAX**|-0.47|Višji davek → nižja cena|
|**NOX**|-0.43|Onesnaženje → nižja cena|
|**CRIM**|-0.38|Kriminal → nižja cena|
|**AGE**|-0.38|Starejše soseske → nižja cena|
|**B**|+0.33|(kompleksna etnična formula – pogosto manj pomembna)|

Mi bomo uporabili kar vse podatke in dali MDEV na Y os. 

Podatke smo tudi razdelili na učno in testno množico s pomočjo train_test_split funkcije. 

Naredili smo tudi STANDARDIZACIJO, ker nekateri podatki delujejo bolje, če so v isti skali. To smo naredili s pomočjo knjižnice StandardScaler(). 

Tukaj je celotna koda:
```
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

#Preberemo csv file
result_of_csv_read = pd.read_csv('HousingData.csv')


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


```

*****************************************
**GRAJENJE IN TRENIRANJE MODELOV**

Uporabili bomo linearno regresijo. 

```
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
```

*******************************
EVALVACIJA MODELA

Tukaj bomo izmerili natančnost modela.

```
from sklearn.metrics import mean_squared_error, r2_score
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

```

*****************
SHRANIMO MODEL


```
import joblib
joblib.dump(model, "house_price_model.pkl")

```

Celotna koda se nahaja v mapi Napoved cene hiše. 

 DODATEK: 

Dodal sem se Vizualizacijo, kjer primerjaš dejanske in napovedane cene, kako shranimo modele, analiza - kaj vpliva na ceno ter napredno matriko (RMSE ali MAE).

![[Pasted image 20250523101717.png]]

![[Pasted image 20250523101733.png]]

