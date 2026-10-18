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
| calendar. train_data_with_warmup                              | warmup periyodunu da kapsayan veri<br>Warmup + Train Data, warmup periyodu varsa <br>TrainingFeature ve                                                                                    |                 |
| calendar.self.DANGEROUS_<br>merged_warmedup_<br>FORECAST_data | Warmup Data + Train Data+ Forecast Data<br><br>DANGEROUS<br><br>-> ForecastFeature uretilirken Kullanilir<br><br>                                                                          |                 |



# Fonksiyonlar

###### `var_from_residual_bank_X_period`, 
* Calendar icindeki train, forecast, veya train + forecast bütün periyot icin  VaR hesaplamasi yapar

###### `ewma_realized_variance_bank`
(self, ewma_lam: float = 0.94, ewma_min_periods: int = 30)
* MODEL VARIANCE TAHMINLERI ILE DEGIL, GERCEK RETURN DEGERLERININ RETURN UZERINDEN HESAPLANMIS KOVARYANS/VARYANS DEGERLERI HESAPLAMA YAPAR
  
* Detaylar
* EWMA bank olusturur, calendardaki warmup durumuna göre warmup verisi ile veya training verisi ile seed eder (ilk kovaryans matrixini olusturur)
* Sonra geri kalan dönem icin EWMA methodunu olusturur





---






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

###### `EWMABANK.general_bulk_insert()`
* Parametre olarak diagonal verileri alabilir ya da almadan kendi hesaplayarak ilerleyebilir 



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



| `TrainingFeature`            | Anlam                                                                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `X_in_sample`, `y_in_sample` | Hedefi bilinen satırlar ve ölçeklenmiş hedefleri                                                                        |
| `X_out_sample_last_index`    | Son train günü, hedefi yok                                                                                              |
| `X_in_sample_id_cols`        | Satır başına `Date`, `ticker` (sadece long formatta)                                                                    |
| `DANGEROUS_r_next_in_sample` | Ertesi gün gerçek getiri (**!!!ölçeksiz, karesiz!!!**), sadece değerlendirme için                                       |
| base_variance                | Eger base variancea ihtiyac duyan bir target varsa, varyans tahmini icin gerekli base variance.<br><br>y olusturulurken |
|                              |                                                                                                                         |

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
|                 |                                                                            |                                     |

---

# Featurebuilders, Targets ve Feature Blocks Mantigi

# Feature Builders


# Targets
* Modelin ile train edildigi ve sonra üretilen ciktinin nasil varianceye dönüstügünü tanimlayan bir class.
* Bunu icerisinde veri tutarak degil, **matematiksel formül tutarak yapiyor.**


## Veri Noktalari

* target.scale : targetin büyütüldügü scale
* target.base_lam :
* target.needs_base : base lam dolu ise, 


##### ` target.forward_make_y(self, r_next, base_variance=None) -> np.ndarray:`
* icerisinde bir sonraki günün r_nexti verildiginde bunu modelin ögrenecegi sayiya y ye cevirir.
	* Örnek: `SquaredReturn` → `r² × scale`
	* base variance mantiginde calisanlarda:
	* `RatioToBase` → `r² / base_variance`).
	* 

##### ` target.inverse_to_variance(self, prediction, base_variance=None) -> np.ndarray:`
* `forward`'ın tersi: iceri parametre ile verilen modelin tahminini gerçek varyans birimine geri çeviriyor.
	* `SquaredReturn` → prediction / scale`
	*  base variance mantiginde calisanlarda:
	* `RatioToBase` → `tahmin × base_variance`.


## Base Variance / Olcek Normalizasyonu Mantigi

* Olcek normalizasyonu iceren, **gerçek varyansın EWMA varyansının kaç katı olduğunu (m)** ögrenen bir model yazmak icin kullanilir.

Getirinin ortalaması $\approx 0$ ise, $r_{f+1}^2$ gerçek varyansın gürültülü bir gözlemidir:

$$  
r_{f+1} = \sigma_{f+1}\, z, \qquad z \sim (0,1)  
\;\Longrightarrow\;  
\mathbb{E}\!\left[r_{f+1}^2\right] = \sigma_{f+1}^2  
$$
Hedefi

$$y_f = \frac{r_{f+1}^2}{\hat{\sigma}^2_{f+1,\,\mathrm{base}}}$$

olarak tanımladığımızda, baz varyans $f$ anında bilindiği için beklentinin dışına çıkar:

$$  
\mathbb{E}\!\left[y_f \mid \mathcal{F}_f\right]  
= \frac{\mathbb{E}\!\left[r_{f+1}^2 \mid \mathcal{F}_f\right]}{\hat{\sigma}^2_{f+1,\,\mathrm{base}}}  
= \frac{\sigma_{f+1}^2}{\hat{\sigma}^2_{f+1,\,\mathrm{base}}}  
= m_f  
$$

Yani model, gerçek varyansın EWMA varyansının kaç katı olduğunu ($m_f$) öğrenir. $m_f = 1$ ise EWMA zaten doğrudur.



##### `emwa_base_variance(self, returns: pd.DataFrame, min_periods: int = 30) -> pd.DataFrame | None:`

* Ewma bank mantigini kopyalar.
* `min_periods=min_periods`: İlk 30 satır NaN kalır, yani ısınma (warmup) süresi. A
* iceri verilen returns icin,  **target mantigi ile** ile bir variance tahmin proxysi olusturur.
	* **BUNUN YUZUNDEN BASELI OLANLARDA KOPYALANIP** `(returns ** 2)` kismi targete göre uyarlanarak tekrar yazilmali.
* `base_lam` λ lambda agirligi ile bir EWMA mantigi ile güncellenmis GUNLUK EVMA varianceleri olusturur.
  




## Su an tanimli olan targetler
| Hedef           | Modelin öğrendiği `y`                   | Varyans (geri dönüşüm)             | Ek alanlar                                                                                                                                                                                          | Taban (EWMA) gerekir mi |
| --------------- | --------------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `SquaredReturn` | `r² × scale`                            | `tahmin / scale`                   | yok                                                                                                                                                                                                 | hayır                   |
| `RatioToBase`   | `min(r² / EWMA varyansı, clip) × scale` | `(tahmin / scale) × EWMA varyansı` | `base_lam=0.94`, `min_periods=30`, `clip=None`<br><br><br>clip= asiri volatil günlerde,<br>![[Pasted image 20261015230246.png]]<br>egitimin görecegi max sayiyi belirli bir degerde sinirlamak icin | **evet**                |
| `LogVariance`   | `log(r² + eps) × scale`                 | `exp(tahmin / scale)`              | `eps=1e-8`                                                                                                                                                                                          | hayır                   |
| `AbsReturn`     | `                                       | r                                  | × scale`                                                                                                                                                                                            | `(tahmin / scale)²`     |
|                 |                                         |                                    |                                                                                                                                                                                                     |                         |
|                 |                                         |                                    |                                                                                                                                                                                                     |                         |



# Feature Blocks / Feature Spec / Feature Builders

### Feature Blocks:


* FeatureBlock classi
	*  Tek bir **feature sütununun** matematiksel olarak tanimini icerir
	* Her blok bir `compute(returns, tickers)` metoduna sahip ve `(özellik_adı, değerler)` seklinde outputa sahip
	* Bütün bloklar aynı girdiyi alıyor: tarih × dönus tablosu (`returns`). 
		* ![[Pasted image 20261015231342.png]]
	* `long_only = True` oalrak isaretlenirse, ozellik sadece long icin mantikli vr wide icerisinde atlanir



### FeatureSpec:

##### 1. `blocks: tuple`

Tek alan: blokların sıralı listesi. Örneğin:

```python
FeatureSpec((Returns(), AbsReturn(), RollingStats((5, 21)), EwmaVol(21)))
```

##### 2. `default(...)`: hazır varsayılan liste
Ilk denemelerimden kalan feature seti, `[Returns(), AbsReturn(), RollingStats(tuple(windows)), EwmaVol(ewma_span)` ile olusturuluyor

##### 3. `withblocks(...)`: onceden hazirlanmis bir FeatureSpecin uzerine yeni bir ekleme yapma

##### 4. `compute(returns, tickers)`: asıl iş
Değerleri `kind`'a göre `series` veya `shared` sözlüğüne koyuyor, yanına `views`'i (wide/long) ekliyor.


# fb.build_features
* Iceri verilen Feature Tanimi, Target Tanimi, Calendar ve arzulanan tipe göre (long yada wide) Feature özelliklerini üretir fb.build_feauters 

```python
spec = FeatureSpec((
    Returns(),
    AbsReturn(),
    RollingVol((5, 21)),         # sadece volatilite, iki pencere
    EwmaVol(21),
    EwmaVariance(0.94),          # EWMA varyansı feature olarak
    TickerOneHot(),              # ticker kimliği (sadece long'da)
))

target = RatioToBase(clip=20)


training_feature, forecast_feature = fb.build_features(
    calendar, "long", spec,target,
)
```


### Akisa Dair Detay
* calendar.DANGEROUS_merged_warmedup_FORECAST_data'i, warmup + train + forecast birleşik tabloyu alir.
* her spec icin iceri return degerlerini ve ticker adlarini vererek ciktiyi alir.
* **r_next**i olusturur, returns.shift(-1) ile.
* 

# Tree Models

# Garch Models

* Garch(1,1) calistirir,
$$
σ²_{t+1} = ω + α · r_t² + β · σ²_t
$$

- `r_t²`: dünkü getirinin karesi (dün büyük hareket olduysa yarın da oynak olma eğilimi var, volatilite kümelenmesi).
- `σ²_t`: dünkü varyans tahmini (varyans yavaş değişir).
- `ω, α, β`: üç parametre, eğitim verisinden **maximum likelihood** ile bulunur.


* ## Temel baselinemiz, wide garch. sadece portfolyonun getirilerine bakarak tahmin yapar.



### helper methods

##### `_fit_params`
iceri veirlen return ve scale degeri ile garch fit eder, `result.params` döndürur

##### `_one_step_variance`
returns: pd.Series: tahmin yapilacak günlerin return degerleri
params: pd.Series : fit esnasinda buldugumuz arch parametreleri, train verisi ile hesalanmis

cikti: returns icindeki her gün icin variance tahmini

* .fix(params) ile bulunan parametreler ile sabit bir model olusturuyoruz
* fixed.forecast(horizon=1, start=0, reindex=False) sayesinde, `start=0` bu hesabın serinin başından itibaren **her gün için** üretilmesini sağlıyor. 

##### `check_chronological_order_and_combine_dataframes
#possiblebug
* verilen train feature ve forecast featurenin icerisindeki return degerlerini birlestirir
* **Tahmin yapilirken, GARCHIN train zamanindaki degerleri de görerek ilerlemesi gerekir. Aslinda expanding bir mantik** 


## Wide Training/ Prediction
* Training periyottaki **PORTFOLYO RETURNU** ile modeli train eder.
* Tahmin zamaninda da yine Portfolyo Returnlarini ve fit edilmis   _one_step_variance ile iceri verip tahminleri alir.

## Long  Training and Predicion
* Her bir ticker icin ayrica egitir.
	* Olusan parametreleri `TrainedGarchModel.params[ticker]` altinda saklar.
* Ewma Residual Bank ile varianceleri Tree Modelde oldugu gibi birlestirir ve portfolio varianceini hesaplar.



* Prediction esnasinda

---

# Backtest Suite



## Statistical Tests

| Test                                    | Neyi sorar                                                            | İstatistik                                                                | Çıktı alanı                          |
| --------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------ |
| **Binomial**                            | İhlal sayısı beklenenle uyuşuyor mu                                   | z = (X − Np)/√(Np(1−p)), \|z\| > 1.96 ise red (normal yaklaşım)           | `binomial_pass`                      |
| **Basel traffic light**                 | İhlal sayısı Basel'in yeşil/sarı/kırmızı bölgelerinden hangisinde     | P(Binom(N,p) ≤ X) < 0.95 yeşil (1), < 0.9999 sarı (2), diğeri kırmızı (3) | `traffic_light_zone`                 |
| **Kupiec POF**                          | İhlal _oranı_ p'ye eşit mi (unconditional coverage)                   | olabilirlik oranı, χ²(1)                                                  | `kupiec_pof_stat`, `kupiec_pof_pass` |
| **Kupiec TUFF**                         | İlk ihlale kadar geçen süre makul mü                                  | ilk ihlal gününe göre LR, χ²(1)                                           | `kupiec_tuff_pass`                   |
| **Haas TBF**                            | İhlaller arasındaki tüm bekleme süreleri makul mü (oran + aralık)     | LR_TBFI + LR_POF, χ²(X+1)                                                 | `haas_tbf_pass`                      |
| **Christoffersen independence**         | İhlaller kümeleniyor mu (dün ihlal → bugün ihlal olasılığı farklı mı) | 1. derece Markov zinciri LR, χ²(1)                                        | `christoffersen_ind_stat/_pass`      |
| **Christoffersen conditional coverage** | Oran **ve** bağımsızlık birlikte                                      | LR_POF + LR_IND, χ²(2)                                                    | `christoffersen_cc_stat/_pass`       |


###### `run_full_backtest_suite_single_window(forecast_df, ...)`
forecast_df: `varForecast` ve `loss` kolonlu, tarih indeksli tek bir `forecast_df` .
start_date,
end_date,
var_confidence: float : hesapladigimiz varin confidence seviyesi
test_confidence: float : Backtest'in ne kadar kesinlikle karar vereceği

* 

##### Exceedance/ iIhlal:
`loss > varForecast` olan gün (kesin büyük).

##### P = 1 - var_confidence
(VaR %95 ise p = 0.05). Doğru bir VaR'da ihlal olasılığı p olmalı. 

##### test confidence:
`alpha = 1 - test_confidence`,
**0.95 → 0.99**: Test daha zor red verir. Doğru bir modeli yanlışlıkla reddetme riski %5'ten %1'e düşer.




### 21 günlük bir periyotu test etmek ile 252 günlük bir periyodu test etmek arasindaki fark

21 günlük bir model icerisinde, %95 VaR da exceedancelerin olma ihtimali (binom dagilim ile, birbirinden bagimsiz günler)



### Model Based Tests

| Group            | Metric (result key)                                                       | Meaning                                                                                     |
| ---------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Score            | `mean_tick_loss`                                                          | headline, lower is better                                                                   |
|                  | `tick_loss_exceedance_part`                                               | mean of `a*(L-V)+`: cost of being too LOW (tail days)                                       |
|                  | `tick_loss_overshoot_part`                                                | mean of `(1-a)*(V-L)+`: cost of being too HIGH (conservatism)                               |
| Conservativeness | `mean_var`, `median_var`                                                  | raw VaR level                                                                               |
|                  | `var_to_realized_quantile_ratio`                                          | `mean_var / empirical a-quantile of L in the window` (>1 = conservative)                    |
|                  | `mean_slack_non_exceedance`                                               | mean of `V - L` on non-exceedance days (unused buffer)                                      |
| Tail severity    | `mean_exceedance_size`, `max_exceedance_size`                             | how bad the misses are (`L - V` on exceedance days)                                         |
| Volatility       | `var_std`, `var_cv` (std/mean)                                            | scale-free spread of the VaR                                                                |
|                  | `mean_abs_daily_change`, `relative_mean_abs_daily_change` (/ mean VaR)    | day-to-day jumpiness                                                                        |
|                  | `var_p95_p5_ratio`, `var_max_min_ratio`                                   | range                                                                                       |
| Context          | `exceedance_rate`, `expected_exceedance_rate`, `n_days`, `var_confidence` | conservatism must be read together with coverage (VaR = 100% is "conservative" and useless) |


# Baseline = Garch Model



# Walk Forward Experiment

```python

setting = wf.WalkForwardSetting(
    train_bars=756,        # 3 yıl train
    forecast_bars=252,     # her model 252 gün tahmin eder, sonra pencere 252 bar ilerler
    expanding=False,       # False: kayan pencere (hep 756 bar); True: baştan büyüyen
    warmup_bars=30,        # train başından önceki ısınma barları (sadece rolling/EWMA ısınması)
)
calendars = setting.make_calendars(df, TICKERS, WEIGHTS)     # her pencere için bir takvim


recipe = wf.ModelRecipe(
    'long',                                   # 'long' | 'wide'
    model_family='xgboost',                   # 'xgboost' | 'garch' | 'ewma' (modelsiz)
    target=SquaredReturn(scale=10000),        # ne tahmin edilir (RatioToBase, LogVariance, AbsReturn da var)
    features=FeatureSpec.default(),           # hangi feature'lar
    model_kwargs=dict(n_estimators=100, max_depth=3),   # xgboost'a gider (garch'ta arch_model'e)
    mu_mode='constant', default_mu_hat=0.0,   # residual bank ayarı (eğitimde residual'lara gömülür)
    ewma_lam=0.94, ewma_min_periods=30,       # EWMA korelasyon bank'ı (sadece long)
)

fits = wf.train_walk_forward(calendars, recipe)    # her pencere için bir WalkForwardSingleCalendarFit


var_df = wf.create_combined_residuals_bank_and_read_var_df(
    fits,                                          # birleşik residual bank'ı içeride kurar ve VaR'ı ondan okur
    var_confidence=0.95,                           # alt %5 quantile
    multi_day_var_calculation_window_size=252,     # int: son N residual; None: eldeki tüm residual'lar (expanding)
    multi_day_var_min_periods=30,                  # bir günün VaR alması için gereken en az önceki residual
)
# var_df sütunları: varForecast, n_residuals_used, loss, window_id

bank = wf.build_combined_residual_bank_WITHOUT_CALC(fits)    # aynı birleşik bank, VaR hesaplamadan (grafikler için)





```


* Elimizdeki veri icin birden fazla calendar olusturarak, elimizdeki veriyi bircok pencereye cevirip ayni modeli farkli zaman dilimleri icin egitip, total olarak bütün periyotlar icerisinde ne kadar basarili oldugumuzu incelemek icin bir sistem.

* ##### forecast edilen günler atlanmadan, kendileri  bir sonraki periyodun tahminine eklenir
* ##### Tek bir experiment sonunda tek bir VaR Dataframei olusturulur.


- 

## Class: WalkForwardSetting
```
train_bars: int
forecast_bars: int = 21
expanding: bool = False
warmup_bars: int = 30
residual_sharing: Literal["none", "carry_over"] = "none"
```

* train_bars = kac günlük/ barlik veri ile tahmin edecegiz?
* forecast_bars = kac günü tahmin edecegiz
* warmup_bars = normal calendarda oldugu gibi, train verisi icerisinde direkt olmayan ama train verisi icerisindeki gecmise yonelik verileri egitmek icin kac gun geriye bakabileceginin ayari

* **expanding** = train dönemi gittikce uzamali mi mi (önerilmez) yoksa sadece bu train bars kadarki periyot icerisinde mi kalmali



### `def WalkForwardSetting.make_calendars(df, TICKERS, WEIGHTS)
* Ayarlar tanimlandiktan sonra onlara göre verilen dataframe uzerinden **calendarlari** olusturup döndürür.
* 

# Model Ayarlama, Featureleri Verme ve Train Etme
### Class: ModelRecipe
* **`model_family`:** `'xgboost'`, `'garch'` (arch paketi, sadece getirileri okur) veya `'ewma'` (modelsiz saf EWMA, "hiçbir şey yapmama" tabanı).
- **`model_format`:** `'long'` veya `'wide'`.
- **`features` ve `target`:** hangi feature blokları, ne tahmin edilecek (`core/feature_blocks.py`, `core/targets.py`).
- **Bank ayarları:** `mu_mode`, `default_mu_hat`, `ewma_lam`, `ewma_min_periods`. Bunlar eğitim sırasında residual'lara gömüldüğü için burada.
-
```python

recipe_garch = wf.ModelRecipe('wide', model_family='garch',
                              model_kwargs=dict(mean='Zero', vol='GARCH', p=1, q=1, dist='normal'))
recipe_ewma  = wf.ModelRecipe('wide', model_family='ewma')                      # saf EWMA tabanı
recipe_ratio = wf.ModelRecipe('long', target=RatioToBase(clip=20),
                              features=FeatureSpec.default().with_blocks(TickerOneHot()))
```
###  `train_walk_forward(calendars, recipe)`
Olusturulan calendarlar ile modelleri üretir backtesting icin.

### WalkForwardSingleCalendarFit

* Her bir egitilmis calendar döneminin egitilmis modelini, residual bankini, ve  ewma bankini tasir.
* window_id: alir egitim surecindeki sirasina göre.
* 

# Var ve Residual Bank Olusturma


#### `build_combined_residual_bank_WITHOUT_CALC`
* fitlerin residual banklarina erisir ve sadece forecast kisimlarinin alip birlestirir
* ilk fitin training lossunu da iceri verilen tüm tarih icin residual olusturabilmek adina en basa ekler.
* 

#### `create_combined_residuals_bank_and_read_var_df`
* 



