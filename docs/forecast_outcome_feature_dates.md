# Tarih kümeleri: `forecast_dates`, `feature_dates`, `outcome_dates`

Bu doküman VaR hattında geçen tarih kümelerini ve aralarındaki farkı anlatır. Örnek tarihler gerçek bir çalıştırmadan alınmıştır (4 ticker, eşit ağırlık, `train 2024-06-03 → 2024-11-29`, `forecast 2024-12-02 → 2024-12-31`, long format, XGBoost).

## Tek cümlelik özet

Bir bardaki **feature**'lar, **bir sonraki barın** varyansını tahmin eder. Tahminin yapıldığı bar `feature_date`, tahminin ait olduğu (ve gerçek getirisinin yazıldığı) bar `outcome_date`'tir. `forecast_dates` ise tahmin döneminin bütün barlarıdır, ikisinin de kaynağı odur.

## Terimler

| Terim                             | Anlam                                                                                              | Nerede üretilir                                      |
| --------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `calendar.forecast_dates`         | Tahmin penceresindeki **bütün** barlar (`forecast_start … forecast_end`, tatil/hafta sonu hariç)   | [data_management.py:137](../data_management.py#L137) |
| `feature_dates` (forecast tarafı) | VaR için skorlanan bar: **o barın feature'ı** bir sonraki barı tahmin eder                         | `calendar.forecast_pairs`                            |
| `outcome_dates` (forecast tarafı) | `feature_dates[i]`'nin hemen **sonraki** barı. Tahminin ait olduğu gün, VaR'ın çıktığı gün         | `calendar.forecast_pairs`                            |
| `calendar.forecast_feature_dates` | `train_end` + bütün forecast barları. `ForecastFeature.feature_dates` bununla aynıdır              | [data_management.py](../data_management.py)          |
| `ForecastFeature.feature_dates`   | Feature matrisinin gerçekten kapsadığı tarihler (= `calendar.forecast_feature_dates`)              | feature builder                                      |
| `TrainingFeature.feature_dates`   | Eğitimde hedefi bilinen feature satırlarının tarihleri                                             | feature builder                                      |
| `usable_dates`                    | `feature_dates` içinde EWMA burn-in'ini de geçenler (eğitim seed'i)                                | `calendar.usable_train_dates`                        |
| seed `outcome_dates`              | `usable_dates`'in her birinin sonraki barı (eğitim seed'i)                                         | `calendar.next_bars`                                 |
| `X_out_sample_last_index`         | Train penceresinin **son** barının feature satırı, hedefi henüz yok. Outcome'u `forecast_dates[0]` | feature builder                                      |

## Kuralı: own-day (sahip gün) etiketlemesi

Bank'larda (`ResidualBank`, `EwmaVarianceBank`) bir satırın tarihi **outcome günüdür**:

> `Date = X` satırı, X gününde **gerçekleşen** getiriyi ve X'in **bir önceki barının** feature'larından çıkan varyans tahminini taşır.

Bu yüzden model tahmini üretildikten sonra `feature_dates` indeksinden `outcome_dates` indeksine **yeniden etiketlenir** ([data_management.py:1126-1127](../data_management.py#L1126-L1127)).

## Zaman çizelgesi (gerçek tarihlerle)

```
TRAIN penceresi: 2024-06-03 ................................ 2024-11-29   (126 bar)

  TrainingFeature.feature_dates : 2024-07-02 ............ 2024-11-27        (105)
        └─ başı: rolling warmup (21 bar) düşer
        └─ sonu: 11-29'un hedefi yok → 11-27 son hedefli bar (11-28 tatil)
  usable_dates                  : 2024-07-16 ............ 2024-11-27         (96)
        └─ başı: EWMA burn-in (ewma_min_periods - 1 = 29 bar) düşer
  seed outcome_dates            : 2024-07-17 ............ 2024-11-29         (96)
  X_out_sample_last_index       : 2024-11-29  (tek bar, hedefi yok)

  ResidualBank (eğitim sonrası) : 2024-07-17 ............ 2024-11-29         (96)

FORECAST penceresi: 2024-12-02 ................................ 2024-12-31   (21 bar)

  calendar.forecast_dates       : 12-02 12-03 ... 12-30 12-31                 (21)
  ForecastFeature.feature_dates : 11-29 12-02 12-03 ... 12-30 12-31           (22)  ← train_end + hepsi
  forecast_pairs → feature_dates: 11-29 12-02 12-03 ... 12-30                 (21)  ← train_end + [:-1]
  forecast_pairs → outcome_dates:       12-02 12-03 ... 12-30 12-31           (21)  ← forecast_dates
  var_df.index                  :       12-02 12-03 ... 12-30 12-31           (21)

  ResidualBank (forecast sonrası): 2024-07-17 ........... 2024-12-31         (117 = 96 + 21)
```

## `calendar.forecast_pairs`

`forecast_pairs`, takvim oluşturulurken bir kez hesaplanan bir özelliktir (metot değil), `(feature_dates, outcome_dates)` ikilisi olarak okunur: `feature_dates, outcome_dates = calendar.forecast_pairs`. Tanımı [data_management.py:141-150](../data_management.py#L141-L150):

```python
self.train_data_last_index = self.train_data.index[-1:]
self.forecast_feature_dates = self.train_data_last_index.append(self.forecast_dates)
self.forecast_pairs = (self.forecast_feature_dates[:-1], self.forecast_dates)
```

`forecast_feature_dates` hem feature builder'lar (hangi feature satırlarının hesaplanacağı) hem de `forecast_pairs` (hangi çiftlerin kurulacağı) tarafından kullanılır, böylece "train'in son barı forecast aşamasına katılır" kuralı tek yerde yaşar.

- `outcome_dates` = **bütün** forecast barları
- `feature_dates` = `forecast_feature_dates[:-1]`: **train'in son barı** + forecast barlarının sondan bir öncesine kadar olanı
- Eşleşme **bar konumuyla** yapılır: `feature_dates[i]`, `outcome_dates[i]`'nin hemen önceki barıdır.

```
('2024-11-29','2024-12-02')   ← train_end → ilk forecast günü
('2024-12-02','2024-12-03')  ('2024-12-03','2024-12-04')  ...
('2024-12-06','2024-12-09')   ← Cuma → Pazartesi
('2024-12-27','2024-12-30')  ('2024-12-30','2024-12-31')
```

Hafta sonu ve tatil boşlukları konum kullanıldığı için kendiliğinden atlanır (`12-25` verisi olmadığı için 12-24 → 12-26 de böyledir). `date - 1 gün` gibi bir çıkarma kullanılsaydı Pazartesi'nin "önceki barı" Pazar olurdu, ki o bar yok.

`calendar.next_bars(dates)` aynı mantığı ters yönde uygular: `train_end`'in sonraki barı, `forecast_data`'daki ilk bardır.

## `ForecastFeature.feature_dates` ile `forecast_pairs[0]` neden farklı?

| | Kapsam | Uzunluk (örnek) |
|---|---|---|
| `ForecastFeature.feature_dates` | `train_end` + feature satırı olan **bütün** forecast barları | 22 |
| `forecast_pairs[0]` (feature_dates) | Bunların **son bar hariç** hali. Son barın tahmini forecast penceresinin dışındaki bir güne ait olur | 21 |

Son forecast barının (`12-31`) feature satırı vardır ama kullanılmaz, çünkü onun outcome'u (`2025-01-02`) forecast penceresinde değildir.

## İlk forecast günü (`forecast_dates[0]`)

İlk forecast gününü (`12-02`) tahmin eden feature satırı train'in son barıdır (`11-29`, hedefi henüz yok). Feature builder `calendar.forecast_feature_dates` ile bu satırı `ForecastFeature`'a da koyar (rolling pencereler merged train+forecast verisinden hesaplandığı için satır zaten hazırdır), ve `forecast_pairs` onu ilk `feature_date` olarak verir. Böylece ilk gün diğer günlerle aynı yoldan geçer ve eğitim seed'inde ayrıca bir "pending" tahmin gerekmez.

Sonuç: **`n` forecast barı → `n` VaR değeri** (örnekte 21 bar → 21 VaR).

## Hızlı başvuru

| Soru | Cevap |
|---|---|
| Bir `var_df` satırının tarihi ne? | `outcome_date` (VaR'ın ait olduğu gün) |
| O satırın tahmini hangi günün feature'ından? | `outcome_date`'in bar olarak bir öncekinden |
| `n` forecast barından kaç VaR çıkar? | `n` |
| ResidualBank'taki satır sayısı? | seed outcome sayısı + forecast bar sayısı |
| Bank'ta tarih neyi etiketler? | Gerçekleşen getirinin yaşandığı **outcome günü** |
| Neden bar konumu kullanılıyor, `date - 1 gün` değil? | Hafta sonu / tatil boşluklarını atlamak için |

## Bilinen tutarsızlık

`ResidualBank.empirical_var_for_target` önceki barı `target_timestamp - frequency` ile bulur ([data_management.py:682](../data_management.py#L682)). Bu, konum kullanan `forecast_pairs` / `next_bars()` ile aynı değildir. Hedef bir Pazartesi ise `target - 24h` Pazar olur, residual geçmişinde olmadığı için `KeyError` verir. (Çalıştırılarak doğrulandı: `2024-12-10` (Salı) çalışır, `2024-12-09` (Pazartesi) `KeyError` verir.)
