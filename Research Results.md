
# Arastirma Sorusu

**"XGBoost tabanlı varyans tahmini + standardize residual quantile (filtered historical simulation), basit baseline'lardan daha iyi kalibre edilmiş bir VaR üretiyor mu?"**


![[Pasted image 20261005232028.png]]

---

# Sonuç: 23 tickerla bir kez eğitilmiş modeller, rastgele sabit ağırlıklı portföylerde

Kod: `train_23_experiment.py` (23 tickerlık modeller), `train_random_weights_experiment.py` (portföyler ve uygulama), `analyze_random_weights.py` (bu tablolar ve beş kayıtlı deney). Experiments sayfasında: "Random portfolio rw1 ... rw5".

## Kurulum
- **Veri:** 23 hisse (AAPL, MSFT, NVDA, AVGO, GOOGL, META, NFLX, AMZN, HD, MCD, PG, KO, WMT, XOM, CVX, JPM, GS, BAC, V, JNJ, UNH, LLY, MRK), Yahoo günlük fiyat, Alpaca SIP 5 dakikalık bar → günlük realized variance (RV). Walk-forward: eğitim 756 gün, tahmin 21 gün, 2016-03-01'den, test günleri 2019-03-04 → 2026-06-30 (1842 gün, 88 pencere).
- **Modeller:** Beş XGBoost, **23 tickerla bir kez** eğitildi, taşınabilir feature'larla (`MarketEwmaVol` yok, böylece başka bir portföye uygulanabilir): `xgb_long` (kare getiri hedefi), `rv_har` ve `drv_har` (RV hedefleri + HAR feature'ları), `rv_macro` ve `drv_macro` (+ VIX, S&P 500 getirisi, faiz farkları). Uygulamada model **yeniden eğitilmiyor**, sadece yeni portföyün bankaları dolduruluyor (`ModelRecipe.pretrained_from`).
- **Portföyler:** 23 hisse üzerinde Dirichlet'ten çekilmiş sabit ağırlıklar (tohumlu, tekrarlanabilir), her biri tüm dönem boyunca sabit. Her portföy için GARCH, EWMA ve historical VaR o portföyün getirisiyle hesaplandı.

| Portföy | Dirichlet alfa | Etkin ticker sayısı | En büyük ağırlık | Köşegen payı* |
|---|---|---|---|---|
| eq (1/23, referans) | - | 23.0 | 0.04 | 0.19 |
| rw1 | 1.0 | 11.6 | 0.17 (META) | 0.31 |
| rw2 | 1.0 | 10.1 | 0.23 (BAC) | 0.34 |
| rw3 | 0.3 | 6.8 | 0.24 (MCD) | 0.42 |
| rw4 | 0.3 | 4.5 | 0.40 (LLY) | 0.65 |
| rw5 | 0.1 | 1.3 | 0.86 (MCD) | 0.90 |

*Portföy varyansının ne kadarı ticker varyanslarından (`Σ wᵢ² σᵢ²`) geliyor, kalanı çapraz terimlerden (`σᵢσⱼρᵢⱼ`).

## Ana tablo: GARCH'a göre fark
Mean tick loss'un **aynı portföyün GARCH'ına** göre farkı, % (negatif = XGBoost daha iyi). Sağdaki sütun, 5 rastgele portföyün kaçında GARCH'tan iyi olduğunu sayıyor (eq hariç).

**%95, tüm dönem**

| Model | eq | rw1 | rw2 | rw3 | rw4 | rw5 | Ortalama | GARCH'tan iyi |
|---|---|---|---|---|---|---|---|---|
| `drv_har` | -1.7 | -3.3 | -2.3 | -2.0 | -2.7 | -2.0 | **-2.5** | 5/5 |
| `rv_macro` | -0.1 | -2.0 | -1.1 | -0.2 | -3.1 | -2.6 | -1.8 | 5/5 |
| `drv_macro` | -1.1 | -2.5 | -1.4 | -0.9 | -2.1 | -1.6 | -1.7 | 5/5 |
| `rv_har` | 0.0 | -2.2 | -1.2 | -1.2 | -2.4 | -1.8 | -1.7 | 5/5 |
| `xgb_long` | +4.3 | +1.5 | +2.3 | +2.9 | +0.1 | +3.1 | +2.0 | 0/5 |
| `ewma` | +6.8 | +5.1 | +5.6 | +6.9 | +1.2 | +3.6 | +4.5 | 0/5 |
| historical | +19.6 | +17.8 | +17.0 | +18.9 | +6.4 | +10.3 | +14.1 | 0/5 |

**%99, tüm dönem**

| Model | eq | rw1 | rw2 | rw3 | rw4 | rw5 | Ortalama | GARCH'tan iyi |
|---|---|---|---|---|---|---|---|---|
| `drv_har` | -6.0 | -9.3 | -5.0 | -4.5 | -3.5 | -5.1 | **-5.5** | 5/5 |
| `rv_har` | -6.8 | -9.9 | -6.3 | -4.3 | -6.1 | -0.3 | -5.4 | 5/5 |
| `drv_macro` | -3.5 | -7.8 | -2.7 | -2.8 | -3.5 | -5.7 | -4.5 | 5/5 |
| `rv_macro` | -5.0 | -8.6 | -4.8 | -2.3 | -6.3 | +0.7 | -4.2 | 4/5 |
| `xgb_long` | -2.0 | -4.3 | -1.9 | +3.5 | -3.8 | +10.3 | +0.8 | 3/5 |
| `ewma` | +0.4 | +1.4 | +2.3 | +4.9 | -0.7 | +8.2 | +3.2 | 1/5 |
| historical | +35.1 | +31.1 | +35.5 | +37.5 | +15.2 | +29.3 | +29.7 | 0/5 |

**%95, kriz penceresi (2020-02-19 → 2020-06-30, 93 gün, tanımlayıcı):** ortalama fark `rv_macro` -1.7, `drv_har` +0.8, `drv_macro` +2.5, `rv_har` +4.2, `xgb_long` +17.7, `ewma` +26.9, historical +64.2. RV modellerinin GARCH'a karşı işareti portföye göre değişiyor (örn. `rv_macro` rw4'te -12.5, rw1'de +4.1). **%95, kriz sonrası:** RV modelleri ortalama %1.9-2.9 önde, rastgele portföylerin hepsinde; `xgb_long` de 5/5 önde (-1.0).

## İstatistiksel anlamlılık
- Her portföyde Diebold-Mariano (HAC) testi, 7 karşılaştırma için Holm düzeltmesi.
- **%95'te hiçbir tekil fark düzeltmeden sonra anlamlı değil** (en iyi ham p = 0.027, düzeltilmiş 0.17).
- **%99'da yalnızca rw1** net: `drv_har` -%9.3 (düzeltilmiş p = 0.002), `rv_har` -%9.9 (0.015), `rv_macro` -%8.6 (0.025), `drv_macro` -%7.8 (0.033). Diğer portföylerde anlamlı değil.
- historical, %99'da rw1, rw3 ve rw5'te Kupiec testinden kalıyor.

## Çıkarımlar
1. **RV hedefli modeller GARCH'ı beş portföyün beşinde de geçiyor ama farklar küçük:** %95'te %1-3, %99'da %3-10. Yön tutarlı, anlamlılık zayıf. **Beş portföy aynı 23 hisseden ve aynı piyasa faktöründen çıktığı için bağımsız örnek sayılmaz**; "5/5" göründüğü kadar güçlü bir kanıt değil.
2. **EWMA ve historical hiçbir portföyde GARCH'ı geçmiyor** (%95'te sırasıyla ortalama +4.5 ve +14.1).
3. **Kare getiri hedefli `xgb_long`, 23 tickerla eğitilse de %95'te GARCH'ı geçemiyor** (0/5); RV hedefleri ve HAR feature'ları fark yaratıyor.
4. **Köşegen payı hipotezi desteklenmedi.** Önce şu açıklamayı denemiştim: 23 tickerlık eşit ağırlıklı portföyde portföy varyansının büyük kısmı çapraz terimlerden geldiği için (köşegen payı 0.19) XGBoost'un ticker varyans tahmini az işe yarıyor, GARCH öne geçiyor. O doğruysa portföy konsantre oldukça (köşegen payı 0.19 → 0.90) XGBoost'un avantajı artmalıydı. Artmıyor: `drv_har` için %95 farkı -1.7, -3.3, -2.3, -2.0, -2.7, -2.0, yani düz. Kriz penceresinde en konsantre portföyde (rw5, köşegen 0.90) `drv_har` GARCH'tan %6.1 kötü. Verideki farkın nedeni bilinmiyor.
5. **Eşit ağırlıklı 23'lüde GARCH'ı geçememe,** çeşitlendirilmiş portföyün piyasaya çok paralel olması ve farkların küçülmesiyle uyumlu (RV modeli GARCH'ın yalnızca %1.7 altında), ama bu da bir açıklama değil bir gözlem.

## Sınırlar
- Tek dönem (2019-03 → 2026-06), tek evren (23 ABD büyük hissesi), beş portföy.
- Kriz penceresi 93 gün, %95'te beklenen aşım yaklaşık 5, yani yalnızca tanımlayıcı.
- RV, seans içi varyans (kapanıştan kapanışa varyansın yaklaşık %50-85'i), modeller bu ölçekte eğitiliyor; VaR'da artık kuantili bunu telafi ediyor ama QLIKE gibi seviye skorlarında RV hedefli modeller adaletsiz cezalanıyor.
- Daha güçlü bir kanıt için çok daha fazla portföy (yirmi gibi), farklı dönemler ve farklı hisse evrenleri gerekir.

