### AI NOTE: ITS NOT FULL YET! BAZ ALMA



# Calendar 
(GroundTruthCalendar)



![[01_calendar.drawio.svg]]


```python
calendar_last_180_days = dm.DailyGroundTruthCalendar(

df, TICKERS, WEIGHTS,

train_start="2024-06-03", 

train_end="2024-11-29", 

forecast_start="2024-12-02", 

forecast_end="2024-12-31",

)
```

* Ana Tasarlanma sebebi look ahead biasi engellemek ve "Ground Truthu", gun sonu gerceklesmis hisse satis fiyatini tutan tek merkezi obje olmak.
* Ayrica, elimizde sadece trading günleri kayitli oldugu icin tahmin yapilacakken "gün - 1" yerine "index -1" deki veriye bakilir. 
* Bunu saglamak icin uretetilirken trainin baslangic ve bitis, yapilacak forecastin baslangic ve bitis gunlerini alir.


### warmup_bar
train_start öncesi, Feature generation burnuplarinda ve EWMA bankdaki ilk covariance matrix  hesaplamasinda training perioddaki verilerin atilmamasi icin, ekstra önceki dönemden veri baglamak icin train_start öncesindeki veriler dahil edilebilir. bunlar sadece rolling fonksiyonlarin hesaplanamsinda kullanilir, kendi return degerleri vs. veri icerisinde olmaz.

Varsayilan = 0, eger training datanin ilk kismi 

## Butun tarih yonetiminde temel kural: own-day (sahip gün) etiketlemesi

Bank'larda (`ResidualBank`, `EwmaVarianceBank`) bir satırın tarihi **outcome günüdür**:

> `Date = X` satırı, X gününde **gerçekleşen** getiriyi ve X'in **bir önceki barının** feature'larından çıkan varyans tahminini taşır.

# Tarihler

| Terim                                                                         | Anlam                                                                                                                                                                                                                                                                                                                                                                                                               | nerede üretilir |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
|                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                     |                 |
| `TrainingFeature.feature_dates`                                               | Eğitimde hedefi bilinen feature satırlarının tarihleri, feature warm up lari (rolling pencerler vs hesaplanirrken) sonrasi survive eden  degerleri Yukaridaki grafikteki siyah alan.<br><br>Dikkat: ewma brun in ayrica usable_dates ile uygulanir.<br>                                                                                                                                                             | feature builder |
| `ForecastFeature.feature_dates`                                               | Feature matrisinin gerçekten kapsadığı tarihler. Feature Builder forecast degeri olustururken                                                                                                                                                                                                                                                                                                                       | feature builder |
| `calendar.forecast_dates`                                                     | Calendar olusturulurken verilen parametre ile olusturulmus Tahmin penceresindeki **bütün** barlar (`forecast_start … forecast_end`, tatil/hafta sonu hariç) (yukaridaki grafikte yesil)<br>                                                                                                                                                                                                                         |                 |
| calendar.forecast_pairs                                                       | `calendar.forecast_dates` de kayitli indexler ve bir sonraki degerlerini verir. Cunku forecast tarihinde +1 tarihi icin inceleme yapilir.<br>                                                                                                                                                                                                                                                                       |                 |
|                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                     |                 |
|                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                     |                 |

| Terim                                                                         | Anlam                                                                                                                                                                                                                                                                                                                                                                                                                                      |     |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --- |
| fonksiyon: calendar.train_dates_past_burn_in `candidate_dates, burn_in_bars`) | Takvimin basindaki tarihlerden  verilen burn_in_bars kadar degeri atar. Sonra takvim ile candidate_dates'i kiyaslar ve kesisenleri döndürür.<br><br>Ornek rakamlar**<br>:** takvim `[1, 2, 3, 4, 5, 6, 7, 8]`, `burn_in_bars = 3`, <br><br>candidate_dates ile verlilem `[2, 4, 5, 8]`.<br><br>- İlk 3 bar atılır, geçerli küme `[4, 5, 6, 7, 8]`.<br>- Aday 2 geçerli kümede yok, atılır.<br>- Sonuç: `[4, 5, 8]`.<br><br>Kullanim: EWMA  |     |
# Veri Noktalari

| Terim                                                         | Anlam                                                                                                                                                                                      | nerede üretilir |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| calendar.train_data                                           | verilen train_start ile train_end arasindaki data                                                                                                                                          |                 |
| calendar.forecast_data                                        | verilen forecast_start ile forecast end arasindaki data<br><br>(ForecastFeaturenin aksine training döneminin son out of sample verisini tasimaz! Sadece forecast doneminin datasini tasir) |                 |
| calendar.self.DANGEROUS_<br>merged_warmedup_<br>FORECAST_data | Train Data+ Forecast Data<br><br>DANGEROUS<br><br>-> ForecastFeature uretilirken Kullanilir<br>-> Ewma bank uretilirken kullanilir<br>                                                     |                 |
| calendar. train_data_with_warmup                              | warmup periyodunu da kapsayan veri                                                                                                                                                         |                 |

# Fonksiyonlar

| Fonksiyon |     |     |
| --------- | --- | --- |
|           |     |     |
|           |     |     |
|           |     |     |
|           |     |     |





|     |     |     |
| --- | --- | --- |
|     |     |     |
|     |     |     |
|     |     |     |
|     |     |     |

# TrainingFeature

# ForecastFeature



---



| `calendar.forecast_pairs()` in dondurdugu  `outcome_dates`    | `feature_dates[i]`'nin hemen **sonraki** barı. Tahminin ait olduğu gün, VaR'ın çıktığı gün |     |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --- |
| `calendar.usable_train_dates` in dondurduguu u `usable_dates` | `feature_dates` içinde EWMA burn-in'ini de geçenler                                        |     |
|                                                               |                                                                                            |     |
|                                                               |                                                                                            |     |
|                                                               |                                                                                            |     |
|                                                               |                                                                                            |     |

# Residual Bank
![[02_residual_bank.drawio.svg]]

$$
\text{standardized\_residual}_t =
\frac{\text{realized\_return}_t - \text{mu\_hat}}
{\sqrt{\text{predicted\_variance}_t}}
$$

* During filtered historical simulation, the quantiles of standartized residuals are taken as an proxy for the VaR. 
* To capture and save those residuals, the residual bank structure is implemented. Model egitimi sonrasinda model objesi icerisinde initial_residual_bank ile residuallere erisilebilir. Residualler sonrasinda istenilen confidence level (%1/%5 quantil ile) ile VaR degerine cevirilebilir.

* Realized return verisi direkt gercek veriden alinir.

```python
residual_bank = dm.ResidualBank(

mu_mode="constant", # "constant": mu_hat = default_mu_hat | "rolling": mean of the last rolling_mu_hat_window_size returns

default_mu_hat=0.0,

rolling_mu_hat_window_size=50, # only read when mu_mode="rolling"

)
```

- **`"constant"` (varsayılan):** her satırın `mu_hat`'i `default_mu_hat`. Hepsi tek seferde yazılır.
- **`"rolling"`:** `mu_hat_t` = t'den **önceki** son N barın ortalama getirisi (t'nin kendi getirisi dahil değil). Boş bank'ın ilk N barının geçmişi olmadığı için, `training_safe_bulk_insert` ilk N barı yavaş ve sıralı (her adımda `mu_hat`'i günceller), kalanını hızlı yazar. Sonuç aynı, sadece daha hızlı.


### Data Points

| Tarih      | actual_return | predicted_variance | predicted_sigma | mu_hat | standardized_residual |
| ---------- | ------------- | ------------------ | --------------- | ------ | --------------------- |
| 2024-07-17 | 0.004583      | 0.000035           | 0.005928        | 0.0    | **0.773**             |
| 2024-07-18 | −0.010688     | 0.000036           | 0.006029        | 0.0    | **−1.773**            |
| 2024-12-31 | 0.003848      | 0.000069           | 0.008313        | 0.0    | **0.463**             |

| Sütun                   | Anlam                                                                          |
| ----------------------- | ------------------------------------------------------------------------------ |
| `actual_return`         | O gün **gerçekleşen** portföy getirisi                                         |
| `predicted_variance`    | O gün için yapılan portföy varyans tahmini (bir önceki barın feature'larından) |
| `predicted_sigma`       | `√predicted_variance`                                                          |
| `mu_hat`                | O gün kullanılan ortalama getiri tahmini (varsayılan 0)                        |
| `standardized_residual` | `(actual_return − mu_hat) / predicted_sigma`                                   |

### Lifecycle

**Tek kullanımlıktir:** `trained = True` olan bank ikinci bir eğitime verilemez, `start_training` `ValueError` verir. Böylece bir eğitimin satırları bir sonrakine sızmaz.

Forecast/ Var hesaplamalari esnsasinda her bir forecast icin kopyalanir ve ayri tanimlar ile modelin forecast dönemi icin olusan residuallar kaydedilir

Iceri tarihler verilirken kronolojik sira ile verilmesi gerekir ve duplicate kabul edilmez

```
ResidualBank(mu_mode=..., ...)                       trained = False
      │
      ▼  EĞİTİM
start_training(start, frequency)                     takvime bağlanır, trained = True
training_safe_bulk_insert(...)                       96 eğitim satırı yazılır
      │
      ▼  FORECAST
.copy()                                     modelin bank'ı eğitim halinde kalır
general_bulk_insert(...)                             +21 forecast satırı → 117
      │
      ▼  OKUMA
multi_day_var(dates, var_confidence, window, min_periods)  → VaR
```

###### `_check_dates_follow_history(dates)`
Duplicate ve kronolojik sirayi kontrol eder


## Expected/Average Return'un rolling oldugu durumda lookahead biasi engellemek icin kurulmus yapilar.
###### `DANGEROUS_add_residual(date, actual_return, predicted_variance, mu_hat)`

**Tek bir satır** yazar: varyansı tabana çeker, `sigma = √varyans`, `standardized_residual = (getiri − mu_hat) / sigma` hesaplar, tabloya ekler. 

Dangereous: `mu_hat`'i **dışarıdan** aldığı için lookahead riski var  Sonuç sonlu değilse uyarıyla satırı atlar. Sadece `step_1` çağırıyor.

###### `step_1_add_residual_with_prev_stateful_mu(date, actual_return, predicted_variance)`
`DANGEROUS_add_residual`'ı, `mu_hat` olarak **bir önceki adımdan kalan** stateful değeri vererek çağırır. Yani bu barın getirisi `mu_hat`'e henüz karışmamış olur.

###### `step_2_change_stateful_mu(as_of)
`step_1`'den **sonra** çalışır, `mu_hat`'i son N barın (artık bu bar da dahil) ortalama getirisiyle günceller. Böylece bir sonraki `step_1` bu barın getirisini görür. `"constant"` modda hiçbir şey yapmaz.

###### `_bootstrap_sequential_prefix(dates, actual_returns, predicted_variances)``
Her tarih için sırayla `step_1` + `step_2`. Yavaş (tek tek pandas yazması) ama kısmi pencereyi tolere ediyor.  `rolling` mu hesaplanan modda ortalama geliri hesaplarken gerekli rolling_mu_hat_window_size kadar surede elle hesaplar, sonrasinda rolling(N).mean().shift(1) ile devam edilir

## Ana Residual Ekleme Fonksiyonlari
### `general_bulk_insert`
Vektörel ve hizli residual ekleme fonksiyonu.
-  `mu_hat`: `constant` ise `default_mu_hat`, `rolling` ise önceki N barın ortalaması (`rolling(N).mean().shift(1)`, geçmiş yetmezse `ValueError`).
- Satırları tabloya **ekler** (`concat`), üzerine yazmaz.
- Stateful `mu_hat`'i günceller en  son eklenen veriden sonra.
### `training_safe_bulk_insert(dates, actual_returns, predicted_variances)`
Bootstrap ya da general_bulk_inserti cagiran fonksiyon


---

# Residual Bank Uzerinden VaR Okuma

###### `windowed_read(window_size, as_of)`
Genel "son N satırı oku" (`as_of`'a kadar). `step_2` ve `empirical_var_for_target` kullanıyor.

## `multi_day_var(dates, var_confidence, window, min_periods)`


**Ana okuma fonksiyonu.** Her tarih için o günden önceki son N residual'ın `(1 − güven)` quantile'ını `shift(1)` ile alır, `VaR = −(mu_hat + sigma · q)`. Çıktı: `varForecast` ve `n_residuals_used`. Pencere `None` ise expanding. Yetersiz geçmiş varsa tüm çağrı `ValueError`. 
## VaR Prediction Window Size

| Ayar                | Anlam                                                              |
| ------------------- | ------------------------------------------------------------------ |
| `window = N` (sayı) | En fazla son **N** residual (rolling)                              |
| `window = None`     | O güne kadarki **hepsi** (expanding)                               |
| `min_periods`       | Bir günün VaR alması için gereken **en az** geçmiş residual sayısı |


* Window size belirtirsen (ROLLING), forecast gününden önceki window size kadar günün %5 %1 quantilelerine bakarak islemi gerceklestirir.
	* Son verilerin gücü daha yüksek olur.
	* **ORNEK KULLANIM**: Ust sinir gibi kullanilabilir, ornegin window_size = 180 ve min = 30 diyerek, asil benim intent ettigim sekilde kullanilabilir.


* Window size belirtmezsen(EXPANDING), quantili aldigi küme sayisi gittikce genisler. 
	* Son verilerin gücü o kadar gücü kaybolabilir.
	* **İlk forecast günleri tamamen eğitim residual'larına dayanıyor.** Model bu verilerle eğitildiği için bu residual'lar gerçek out-of-sample hatadan **küçük** çıkabilir (in-sample sapma), yani `q` biraz dar kalabilir.
	* gürültülü: `0.00578 → 0.01872`, üç katına çıkıyor.
	* **Kısa pencere (`N=20`) bu etkiyi hızla siliyor** (ay sonunda 20'nin 20'si forecast), ama 
	- **Uzun pencere/expanding istikrarlı ama yavaş tepki veriyor:** 96 eğitim residual'ı sonuna kadar kalıyor.
	
* Bir VaRin kac residual ile hesaplandigini görebilmek icin var_df icerisinde simdi `n_residuals_used` var.
  
* Min_Periods parametresi VaR hesaplanmak icin minimum ne kadar veri olmasi gerektigini gosteriyor, tek gün ile hesaplanmis bir VaR cok biasli olur.
	* Min periyot normalde expanding icin 30 gune tanimli, 30 gunden dusukse hesaplamiyor.
	* Rolling icin de rolling gun sayisina dayali

## `empirical_var_for_target`

```python

prev = calendar.previous_bars([target])[0]     # hedefin önceki barı, takvimden (hafta sonu/tatil atlanır)

bank.empirical_var_for_target(
    target_timestamp=target,                          # VaR'ını istediğin gün
    last_feature_timestamp=prev,                      # ondan hemen önceki bar, bank'ta olmalı
    predicted_variance=6.5e-05,                       # o gün için portföy varyans tahmini
    var_confidence=0.95,
    multi_day_var_calculation_window_size=60,         # int: son N residual | None: hepsi
    multi_day_var_min_periods=30,                     # None: rolling'de N, expanding'de 30
)
```

Tek bir gün icin, yeni bir variance tahmini **predicted_variance** ile, VaR hesaplar.





## Residualbank Konteksindeki Iki Windowu Ayirt Etme
| Parametre                               | Nerede Kullanılır?             | Ne Zaman Etkili Olur?                 | Değiştirilirse Ne Gerekir? |
| --------------------------------------- | ------------------------------ | ------------------------------------- | -------------------------- |
| `rolling_mu_hat_window_size`            | `ResidualBank(...)` kurulurken | Satır yazılırken(`mu_mode="rolling"`) | Yeniden eğitim gerekir     |
| `multi_day_var_calculation_window_size` | `var_from_residual_bank(...)`  | VaR okunurken                         | Hiçbir şey gerekmez        |
# EWMA Variance Bank

![[03_ewma_bank.drawio.svg|474]]


* DIKKAT: VAR HESAPLAMASI KONUSUNDA:
	* Bu formül getirilerin ortalamasının sıfır olduğu varsayımına dayanır!
	* 
* Long Formatta model her bir ticker icin ayri bir variance tahmini yapar, bunlari birlesitirp bir portfolyo variancei hesaplama ihtiyaci duyar. Normal covariance matirx

$$
 \Sigma= \begin{bmatrix} \sigma_1^2 & \sigma_{12} & \sigma_{13} & \sigma_{14}\\ \sigma_{12} & \sigma_2^2 & \sigma_{23} & \sigma_{24}\\ \sigma_{13} & \sigma_{23} & \sigma_3^2 & \sigma_{34}\\ \sigma_{14} & \sigma_{24} & \sigma_{34} & \sigma_4^2 \end{bmatrix}
$$


$${\sigma_p^2=w^T\Sigma w}$$

* Bu variance tahminlerinin **portfolyo** seviyesine ortak bir variance tahminine cevirilmesi icin direkt weightler ile ortalamasi alinmak yerine,  projedeki temel hedef olusan sok sonrasi varyans tahmini guncelleme hedefini saglamak ve son degerlere daha cok agirlik vermek icin EWMA variance kullanilir.
* Iki ticker arasi korelasyon hesaplamasi gercek returnlar uzerinden gerceklesir. 
* Diagonal entryler (variance) degerleri **ML Modellerinin variance tahmini** ile eklenir.

* EWMA'nin guncelleme formulu:

$$
 \Sigma_t^{EWMA} = \lambda\Sigma_{t-1}^{EWMA} + (1-\lambda)r_{t-1}r_{t-1}^{T}  
$$


```python
ewma_bank = dm.EwmaVarianceBank(

ewma_lam=0.94, # RiskMetrics decay: larger = forgets slower

ewma_min_periods=30, # the first N train bars seed the covariance (EWMA burn-in)

)
```


## Data Points

**`portfolio_variance`**: Her günün **portföy varyansının** listesi. Başta boş, `float` sayılar tutacak, adı `predicted_variance`. Bu, bank'ın asıl çıktısı: `ResidualBank`'a gidiyor.

**`cov_by_date`**: Hesaplanan her günün **kovaryans matrisi**, 4X4 array seklinde. 

**`sigma_diag`**: Diagonal veriler, modelin her hisse basina hesapladigi variance tahmini

**`_current_sigma`**: **Şu anki** kovaryans matrisi. Yarının kovaryansı bundan hesaplanacak. 

**`_prev_return`**: **Dünün** hisse getirileri (4 sayı). Yarın hesaplanırken kovaryansı güncellemek için lazım.


## Life Cycle

```
EwmaVarianceBank(ewma_lam, ewma_min_periods)             trained = False
      │
      ▼  EĞİTİM
start_training(tickers, weights)                          takvime bağlanır, trained = True
training_safe_bulk_insert(ilk 30 bar)                     başlangıç kovaryansı
general_bulk_insert(...)                                  96 eğitim satırı + portföy varyansları
      │
      ▼  FORECAST
.copy()  →  general_bulk_insert(...)                    
```





###### `training_safe_bulk_insert(warmup_returns)`
* TEMELDE:*
* warmup_returns : trainingein ilk xyz verisi
* ewma_min_periods kadar barin return degerleri uzerinden (varsayilan 30bar ) STANDART KOVARYANS matrixini olusturup` current_sigma` ya kaydeder
	* TEK BIR MATRIX olusturur, 30 ewma_min_periods barlik  baz alarak. 
* ewma_min_periods dan fazla periyot verilirse, 

###### `_portfolio_variance_from_cov(cov, sigma_diag, weights)`
* bir günun varyansindan (modelin hesapladigi) por

# Training

## Trained Model Abstract Class

### Constructor Variableleri
- `training_feature`: modelin eğitildiği `TrainingFeature` (frequency gibi bilgiler buradan okunur)
- `underlying_model`: gerçek model nesnesi (`XGBRegressor`, `RandomForestRegressor`, ...)
- `model_name`: `"xgboost"`, `"random_forest"` (VaR kaydında varsayılan anahtar olarak kullanılırdı)
- `target_scale`: hedefin ne kadar çarpılarak eğitildiği, tahminde geri bölmek için
- `feature_cols`: modelin gördüğü sütunların sırası (tahminde bu sırayla seçilir)
- `residual_bank_in_sample`, `ewma_variance_bank_in_sample`: eğitim dönemi bank'ları. **Başta `None`**, çünkü bank'ları doldurmak için önce modelin eğitilmiş olması gerekiyor 
	- !!!!!! (`long_format_train_seed` modeli skorlayıp bank'ları sonradan atıyor)


## Methods 

###### `predict_variance_single_wide(forecast_feature, feature_date) -> float`
* soyut, implement edilmesi gerekir 
- tek bir tarih için, o barın **ertesi barının** varyansını tahmin eder
- wide veya tek hisseli feature için (satır = bir gün)
	- temel cagirimda/ kullanim alaninda PORTFOLYO VARIANCESI döndürür. 
	- Long egitilmis bir modelde tek tickerin featureleri ile cagirilmasi durumunda (predict_variance_batch_long_by_tickerin) yaptigi gibi, TICKER VARIANCEsi döndürür.

###### `predict_variance_batch_wide(forecast_feature, feature_dates) -> pd.Series`
- aynı iş, bir sürü tarih için tek seferde (tek `.predict()` çağrısı)
- döner: tarih indeksli `predicted_variance` serisi, 


###### `predict_variance_batch_long_by_ticker(X_long, id_cols, feature_dates) -> dict[ticker → Series]`
- **soyut değil**, bütün alt sınıflar bunu hazır alıyor
- LONG formatta egitilmis bir model icin, her bir hissenin günlük variance tahmininin olusmasini saglar.

- long feature bloğunu (satır = bir (gün, hisse)) **hisseye göre** böler (`id_cols.groupby("ticker")`)
- her hisse için tarih indeksli bir "wide görünüm" kurar (`ForecastFeature`, `type="wide"`, o hissenin satırları) ve `predict_variance_batch_wide`'ı çağırır
- yani **long tahmini = hisse başına bir wide tahmini**


###  Training Residual Hesaplama Mantigi

* ### Su anda model Training Period icerisindeki bütün degerler ile tek seferde egitiliyor ve `long_format_calculate_training_standartized_residuals` ile tamamini verinin görmüs modelin tahmininin alinmasi ile training residual hesaplaniyor.


### long_format_calculate_training_standartized_residuals
```python
long_format_train_seed(
pooled_model: TrainedModel,
calendar: GroundTruthCalendar,
training_feature: TrainingFeature,
residual_bank: ResidualBank,
ewma_bank: EwmaVarianceBank,

) -> tuple[ResidualBank, EwmaVarianceBank]:
```

* Basit görevi: pooled (tüm tickerleri görerek egitilmis) long format modeli alarak `training_feature.feature_dates` icin model ile varianceleri hesaplamak, training periyodu icin residualleri hesaplamak, EWMA variance banki kurup varianceleri hesaplamak.
* Banklar bu kisimda trained olarak isaretlenir, boylelikle uzerlerine tekrardan train verisi yazilmasi engellenir.

* Detayli Akis:
* `predict_variance_batch_long_by_ticker` ile `training_feature.feature_dates` icin varianceler hesaplanir.
* usable_dates, ewma rekursyonunun yapilacagi, dolayisi ile training dönemi icin variancelerin/residulallarin hesaplanacagi dönemi isaret eder.
* training safe bulk insert ile, "ewma_bank_min_periods" kadar tick ile ewmanin ilk kovaryans matrixini hesaplar.
	* calendarda ewma_bank_min_periods a yetecek kadar warmup_bar var ise, usable_dates feature_dates ile baslar.
		* hesaplamasi train_dates_past_burn_in ile yapilir
	* yoksa burn in edilir, burn in ciktiktan sonraki periyot ile baslar
* ilk kovaryans matrixini hesapladiktan sonra, usable_dates/outcome_dates icin recursiv olarak general_bulk_insert ile ewma ile portfolio variance hesaplamarini yapar.
* bu portfolio variancelerini training_safe_bulk_insert icine koyar.

### calculate_multi_day_standart_residuals_from_wide_format_model, calculate_multi_day_standart_residuals_from_long_format_model 

1. Modelin eğitimden gelen bank'ının **kopyasını** alır (modelin kendi bank'ı bozulmaz).
2. `calendar.forecast_pairs`'ten `feature_dates` / `outcome_dates` çiftlerini alır. Tahmin feature gününden yapılır, sonuç ertesi günün (outcome) etiketiyle yazılır. İlk forecast günü, train'in son barının feature satırından fiyatlanır.
3. Model her feature günü için varyans tahmin eder.
4. O outcome günlerinin gerçekleşen portföy getirisini `calendar.forecast_data`'dan okur.
5. `bank.general_bulk_insert(...)` ile her gün için standardize residual hesaplayıp ekler: kabaca `(gerçek getiri − mu_hat) / sqrt(tahmin varyans)`.
6. Dolu bank'ı döndürür.





# Long Format Training Akisi




# Long Format Prediction Akisi 

# Wide Format Training Akisi

# Wide Format Prediction Akisi






---


* calendar.forecast_feature_dates icin butun featureleri tasir.
	* YANI CALENDARA VERILEN FORECAST TARIHLERININ YANINDA, Training Featurenin **SON GUNUNU DE, ILK FORECAST GUNUNU TAHMIN EDEBILMEK ICIN TASIR** 
* Warm-Up donemlerini ve trainin son gününu olusturabilmek icin, geriye yönelik accesi vardir ve veriye `calendar.self.DANGEROUS_merged_warmedup_FORECAST_data` uzerinden ulasir.

# Feature Generation


````python
training_feature_long, forecast_feature_long = fb.build_long_fetures(calendar, target_scale=1e4)
````

## Generation Esnasindaki Parameterler
| Parametre             | Etki                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `windows=(5,10,21)`   | Her pencere 3 feature üretir. **En büyük pencere kadar bar kaybedilir** (21 → 20 bar)            |
| `ewma_span=21`        | `ewma_vol` ve `market_ewma_vol`'ın hızı                                                          |
| `sq_return_lags=None` | `(1,2,3)` gibi verilirse gecikmeli kare getiri feature'ları eklenir, o kadar daha bar kaybedilir |
| `target_scale=1.0`    | Sadece hedefi çarpar                                                                             |
## Featureler
|Feature|Nasıl hesaplanıyor|Ne yakalıyor|
|---|---|---|
|`r`|o günün getirisi|son günün şoku|
|`abs_r`|`|r|
|`real_vol_w`|son `w` günün getirisinin std'si (w = 5, 10, 21)|yakın geçmişin volatilitesi|
|`mean_abs_w`|son `w` günün `|r|
|`skew_w`|son `w` günün çarpıklığı|düşüşlerin/yükselişlerin asimetrisi|
|`ewma_vol`|üstel ağırlıklı std (span = 21)|yakın günlere daha çok ağırlık veren volatilite|
|`r2_lag_k`|k gün önceki kare getiri (isteğe bağlı, `sq_return_lags`)|volatilite kümelenmesi (GARCH tarzı)|

## Featurelerdeki Rolling degerler nedeni ile Veri Kaybi
`rolling(21)` gibi rolling olarak olusturulan durmlarda ancak 21 gözlem varsa değer üretir, yani ilk **20 bar** feature'sız kalıp atılıyor. 

## Long Format

* Gun ve hisse basina bir satir
* Hedef: O hissenin ertesi gün karesi


## Wide Format
* gün başına bir satır
* Target : **Portföyün** ertesi gün karesi




## Target Scaling

* Ilk implementasyon esnasindaki target (squared percent return) cok kucuk bir dimensionda oldugu icin modellerin ogrenebilmesi adina scaloing gerekli oldu.
* Feature building esnasinda parametre ile verildikten sonra hangi scale ile olusturuldugu 
`TrainingFeature.target_scale` ile kaydoluyor.
* Model scale edilmis degerler ile ögreniyor.
* Prediction yapildiktan sonra `predict_variance` icerisinde yapilan dönüs scale edilmis sekildeo oluyor.

### Training Feature ve Forecast Feature icerisindeki Veri Noktalari



| `TrainingFeature`            | Anlam                                                                             |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `X_in_sample`, `y_in_sample` | Hedefi bilinen satırlar ve ölçeklenmiş hedefleri                                  |
| `X_out_sample_last_index`    | Son train günü, hedefi yok                                                        |
| `X_in_sample_id_cols`        | Satır başına `Date`, `ticker` (sadece long formatta)                              |
| `DANGEROUS_r_next_in_sample` | Ertesi gün gerçek getiri (**!!!ölçeksiz, karesiz!!!**), sadece değerlendirme için |

### Training Feature ve Forecast Feature icerisindeki tarih Noktalari

#### Calendar icerisindeki tarih

| Dönem    | Tarih Aralığı             | Bar Sayısı |
| -------- | ------------------------- | ---------- |
| Train    | `2024-06-03 → 2024-11-29` | 126 bar    |
| Forecast | `2024-12-02 → 2024-12-31` | 21 bar     

```python
calendar_last_180_days = dm.DailyGroundTruthCalendar(df, TICKERS, WEIGHTS,

train_start="2024-06-03", 
train_end="2024-11-29", 

forecast_start="2024-12-02", 
forecast_end="2024-12-31",

)
```

#### Training Feature Tarihleri

| Değişken        | Açıklama                                                                                       | Değer                                |
| --------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------ |
| `feature_dates` | `X_in_sample`'ın kapsadığı tarihler. Feature warmup'ı (20 bar) düşmüş, son hedefli bar'a kadar | 105 tarih: `2024-07-02 → 2024-11-27` |
| `start`, `end`  | `feature_dates`'in ilk ve son tarihi (`calendar.train_start`/`train_end` **değil**)            | `2024-07-02`, `2024-11-27`           |

#### Forecast Feature Tarihleri

| Değişken        | Açıklama                                                                   | Değer                               |
| --------------- | -------------------------------------------------------------------------- | ----------------------------------- |
| `feature_dates` | `train_end` + bütün forecast günleri (`= calendar.forecast_feature_dates`) | 22 tarih: `2024-11-29 → 2024-12-31` |
| `start`, `end`  | `feature_dates`'in ilk ve son tarihi (`calendar.forecast_start` **değil**) | `2024-11-29`, `2024-12-31`          |



