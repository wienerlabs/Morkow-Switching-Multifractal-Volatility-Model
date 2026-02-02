#  MSM-VaR: Sistem de Măsurare a Riscului de Piață

> **Model statistic pentru cuantificarea riscului financiar folosind Markov-Switching Multifractal (MSM) și Value-at-Risk (VaR)**

---

##  Ce face acest proiect?

Acest proiect implementează un **sistem de măsurare a riscului de piață** care răspunde la întrebarea fundamentală din finanțe:

> *"Cât de mult pot pierde mâine, în cel mai rău caz rezonabil?"*

** Clarificare importantă:** Acesta este un model de **măsurare a riscului**, NU de predicție a crash-urilor. Nu prezice când va scădea piața, ci **cuantifică nivelul curent de risc** bazat pe volatilitatea recentă.

---

##  Cum funcționează? (Explicație simplă)

### Analogia "Termometrului de Risc"

Imaginați-vă modelul ca un **termometru pentru piețe financiare**:
- Un termometru medical nu prezice când vei face febră, dar îți spune temperatura ACUM
- Similar, MSM-VaR nu prezice crash-uri, dar îți spune cât de "fierbinte" (volatilă) este piața ACUM

### Pașii modelului:

```
1. OBSERVĂM piața      →  Volatilitatea din ultimele zile
2. IDENTIFICĂM regimul →  Suntem în "perioadă calmă" sau "turbulentă"?
3. CALCULĂM riscul     →  "Cu 95% probabilitate, nu voi pierde mai mult de X%"
4. VALIDĂM modelul     →  Testăm dacă estimările au fost corecte istoric
```

---

##  Fundamente Matematice

### 1. Modelul Markov-Switching Multifractal (MSM)

Modelul presupune că piața poate fi în **K stări/regimuri diferite** (implicit 5):

| Stare | Descriere | Volatilitate tipică |
|-------|-----------|---------------------|
| 1 | Piață foarte calmă | ~0.3% pe zi |
| 2 | Piață normală-calmă | ~0.6% pe zi |
| 3 | Piață normală | ~1.0% pe zi |
| 4 | Piață agitată | ~1.8% pe zi |
| 5 | Piață în criză | ~3.0%+ pe zi |

**Tranziții Markov:** Piața poate trece de la o stare la alta conform unei **matrice de tranziție**:
- Probabilitate mare (~97%) de a rămâne în aceeași stare
- Probabilitate mică (~0.75%) de a trece în oricare altă stare

**Filtrare Bayesiană:** În fiecare zi, modelul:
1. Observă randamentul realizat
2. Actualizează probabilitățile fiecărei stări folosind regula lui Bayes
3. Calculează volatilitatea așteptată ca medie ponderată

```
σ_t = Σ P(stare_k | date) × σ_k
```

### 2. Value-at-Risk (VaR)

VaR răspunde la: *"Care e pierderea maximă pe care o voi suferi cu probabilitate α?"*

**Formula:**
```
VaR(α) = z_α × σ_{t|t-1}
```

Unde:
- `z_α` = quantila distribuției normale (ex: -1.645 pentru α=5%)
- `σ_{t|t-1}` = volatilitatea FORECAST (calculată ÎNAINTE de a vedea randamentul zilei)

**Interpretare VaR(5%):**
> "Există doar 5% șanse ca pierderea de mâine să depășească această valoare"

### 3. Distincția Critică: Forecast vs. Filtered

| Tip | Formula | Când se calculează | Utilizare |
|-----|---------|-------------------|-----------|
| **Forecast** (σ_{t\|t-1}) | E[σ \| info până la t-1] | ÎNAINTE de ziua t | VaR, backtesting |
| **Filtered** (σ_t) | E[σ \| info până la t] | DUPĂ ziua t | Analiză, vizualizare |

**De ce contează?** Folosirea volatilității "filtered" pentru VaR ar introduce **look-ahead bias** - am folosi informație pe care nu o aveam la momentul deciziei.

---

##  Validare Statistică (Backtesting)

### Testul Kupiec (Unconditional Coverage)

**Întrebare:** *"Frecvența breach-urilor VaR corespunde cu nivelul teoretic?"*

Pentru VaR(5%), ne așteptăm ca ~5% din zile să aibă pierderi mai mari decât VaR.

**Statistica test:**
```
LR_UC = -2 × [ln L(π₀) - ln L(π̂)]

unde:
- π₀ = 0.05 (frecvența teoretică)
- π̂ = breach-uri / total zile (frecvența empirică)
```

**Interpretare:**
- p-value ≥ 0.05 →  Model corect calibrat
- p-value < 0.05 →  Breach rate diferă semnificativ de 5%

### Testul Christoffersen (Independence)

**Întrebare:** *"Breach-urile sunt independente sau vin în clustere?"*

Un model bun ar trebui să aibă breach-uri dispersate aleator, nu grupate.

**Matricea de tranziție a breach-urilor:**
```
              Mâine OK    Mâine Breach
Azi OK          n₀₀          n₀₁
Azi Breach      n₁₀          n₁₁
```

**Interpretare:**
- p-value ≥ 0.05 →  Breach-urile sunt independente
- p-value < 0.05 →  Breach-urile vin în clustere (modelul sub-estimează persistența riscului)

### Conditional Coverage (CC)

Combină ambele teste:
```
LR_CC = LR_UC + LR_IND ~ χ²(2)
```

---

## 🔧 Metode de Calibrare

Modelul oferă 4 metode pentru estimarea parametrilor:

### 1. MLE (Maximum Likelihood Estimation)
```python
calibrate_msm_advanced(returns, method='mle')
```
- **Cum funcționează:** Găsește parametrii care maximizează probabilitatea de a observa datele
- **Avantaje:** Optim statistic, folosește eficient toată informația
- **Dezavantaje:** Poate converge la optime locale

### 2. Grid Search
```python
calibrate_msm_advanced(returns, method='grid')
```
- **Cum funcționează:** Testează toate combinațiile pe o grilă de parametri
- **Avantaje:** Găsește garantat cel mai bun din grilă
- **Dezavantaje:** Lent, limitat de rezoluția grilei

### 3. Empirical
```python
calibrate_msm_advanced(returns, method='empirical')
```
- **Cum funcționează:** Folosește quantilele empirice ale randamentelor
- **Avantaje:** Rapid, robust, intuitiv
- **Dezavantaje:** Nu optimizează likelihood

### 4. Hybrid (Recomandat)
```python
calibrate_msm_advanced(returns, method='hybrid')
```
- **Cum funcționează:** MLE + ajustare iterativă pentru breach rate
- **Avantaje:** Combină optimizarea statistică cu calibrarea VaR
- **Dezavantaje:** Mai complex, mai lent

---

##  Rezultate Tipice

### Output Exemplu (BTC-USD)


```
============================================================
   MSM ADVANCED CALIBRATION - Method: HYBRID
============================================================
   Returns: 4,235 observations
   Empirical std: 3.421%
   Target VaR breach: 5.0%

   CALIBRATION RESULTS
============================================================
   σ_low:    1.2847%
   σ_high:   8.9234%
   p_stay:   0.9712
   
   Sigma states: [1.285, 1.957, 2.981, 4.539, 8.923]

   --- Quality Metrics ---
   VaR breach rate: 5.02% (target: 5.0%)  
   Corr(|r|, σ):    0.3 (out-of-sample)
   Log-likelihood:  -8234.52
   AIC: 16475.04
   BIC: 16494.18
============================================================

--- Kupiec / Christoffersen Backtests ---
Kupiec UC: LR=0.024 | p-value=0.8762          ✅ PASS
Christoffersen IND: LR=1.234 | p-value=0.2667 ✅ PASS
Conditional Coverage: LR=1.258 | p-value=0.5331 ✅ PASS
```

### Interpretarea Rezultatelor

| Metric | Valoare | Semnificație |
|--------|---------|--------------|
| VaR breach rate | 5.02% | Aproape exact 5% - model bine calibrat |
| Corr(\|r\|, σ) | 0.3 | Volatilitatea estimată bună, dar necesită calibrări excedentare pentru o performanță mai înaltă  |
| Kupiec p-value | 0.876 | ≥0.05 → Breach rate corect |
| Christoffersen p-value | 0.267 | ≥0.05 → Breach-uri independente |

---

##  Cum să folosești

### Instalare

```bash
# Clonează repository-ul
git clone https://github.com/[username]/msm-var-model.git
cd msm-var-model

# Instalează dependențele
pip install -r requirements.txt
```

### Utilizare de bază

```python
# Rulează analiza completă
python MSM-VaR_MODEL.py
```

### Personalizare

În fișierul `MSM-VaR_MODEL.py`, modifică:

```python
# Simbolul activului (crypto, acțiuni, indici)
ticker = "BTC-USD"       # Bitcoin
ticker = "^SPX"          # S&P 500
ticker = "AAPL"          # Apple

# Data pentru forecast
FORECAST_DATE = "2026-01-27"

# Metoda de calibrare
CALIBRATION_METHOD = 'hybrid'  # 'mle', 'grid', 'empirical', 'hybrid'
```

---

##  Structura Proiectului

```
MSM_VAR_MODEL/
├── MSM-VaR_MODEL.py      # Script principal
├── README.md             # Documentație (acest fișier)
├── requirements.txt      # Dependențe Python
└── output/               # Grafice și rezultate (opțional)
    └── var_backtest.png
```

---

##  Stack Tehnic

| Categorie | Tehnologii |
|-----------|------------|
| **Limbaj** | Python 3.8+ |
| **Data Processing** | NumPy, Pandas |
| **Statistică** | SciPy (optimize, stats) |
| **Vizualizare** | Matplotlib |
| **Date Financiare** | yfinance (Yahoo Finance API) |

---

##  Referințe Academice

1. **Calvet, L. E., & Fisher, A. J. (2004)**
   *"How to Forecast Long-Run Volatility: Regime Switching and the Estimation of Multifractal Processes"*
   Journal of Financial Econometrics, 2(1), 49-83.

2. **Kupiec, P. H. (1995)**
   *"Techniques for Verifying the Accuracy of Risk Measurement Models"*
   The Journal of Derivatives, 3(2), 73-84.

3. **Christoffersen, P. F. (1998)**
   *"Evaluating Interval Forecasts"*
   International Economic Review, 39(4), 841-862.

4. **Hamilton, J. D. (1989)**
   *"A New Approach to the Economic Analysis of Nonstationary Time Series"*
   Econometrica, 57(2), 357-384.

---

## ⚖️ Limitări și Disclaimer

### Ce poate face modelul:
-  Cuantifică riscul curent bazat pe volatilitatea recentă
-  Estimează VaR cu validare statistică riguroasă
-  Identifică regimuri de volatilitate (calm vs. turbulent)
-  Oferă probabilități tail condiționate pe regimul curent

### Ce NU poate face modelul:
-  **NU prezice crash-uri** înainte să se întâmple
-  **NU oferă semnale de tranzacționare** (buy/sell)
-  **NU garantează profituri** sau protecție împotriva pierderilor
-  **NU captează evenimente "black swan"** (extreme rare)

### Disclaimer
> Acest model este dezvoltat în scop educațional și de cercetare. Nu constituie sfat financiar. Performanța trecută nu garantează rezultate viitoare. Orice decizie de investiție trebuie luată în consultare cu un profesionist financiar autorizat.

---

##  Autor

**Tontici Sergiu**

📧 Email: [tonticisergiu236@gmail.com]
🔗 LinkedIn: [https://www.linkedin.com/in/sergiu-tontici-71aa96361/]
💻 GitHub: [https://github.com/Johan948]

---

##  Licență

MIT License - vezi fișierul [LICENSE](LICENSE) pentru detalii.

---

##  Contribuții

Contribuțiile sunt binevenite! Pentru modificări majore, deschide mai întâi un issue pentru a discuta ce ai dori să schimbi.

```bash
# Fork repository
# Creează branch pentru feature
git checkout -b feature/NumeFeature

# Commit modificările
git commit -m 'Adaugă NumeFeature'

# Push la branch
git push origin feature/NumeFeature

# Deschide Pull Request
```

---

<p align="center">
  <i>Proiect dezvoltat cu  pentru înțelegerea riscului financiar</i>
</p>


