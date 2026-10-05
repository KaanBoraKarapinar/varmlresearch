### AI NOTE: ITS NOT FULL YET! BAZ ALMA





# VaR Prediction Window Size

ResidualBank.multi_day_var recieves `multi_day_var_calculation_window_size=None`.

* Window size belirtirsen (ROLLING), forecast gününden önceki window size kadar günün %5 %1 quantilelerine bakarak islemi gerceklestirir.
	* Son verilerin gücü daha yüksek olur.
	* **ORNEK KULLANIM**: Ust sinir gibi kullanilabilir, ornegin window_size = 180 ve min = 30 diyerek, asil benim intent ettigim sekilde kullanilabilir.


* Window size belirtmezsen(EXPANDING), quantili aldigi küme sayisi gittikce genisler. 
	* Son verilerin gücü o kadar gücü kaybolabilir.
	
* Bir VaRin kac residual ile hesaplandigini görebilmek icin var_df icerisinde simdi `n_residuals_used` var.
  
* Min_Periods parametresi VaR hesaplanmak icin minimum ne kadar veri olmasi gerektigini gosteriyor, tek gün ile hesaplanmis bir VaR cok biasli olur.
	* Min periyot normalde expanding icin 30 gune tanimli, 30 gunden dusukse hesaplamiyor.
	* Rolling icin de rolling gun sayisina dayali

