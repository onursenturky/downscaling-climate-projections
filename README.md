# İstanbul İçin Derin Öğrenme İle Günlük Yağış Ölçek Küçültme

Mevcut **Ocak–Mart 1981** ERA5 ve ERA5-Land dosyalarıyla çalışan bir yöntem deneyi.
ERA5 atmosfer alanlarından aynı günün ERA5-Land yağışını kestirir. Gelecek iklim
projeksiyonu üretmez; GCM/SSP aşaması veri gelene kadar beklemededir.

Ana dosya: [Statistical_Downscaling_of_Climate_Projections_with_Deep_Learning.ipynb](Statistical_Downscaling_of_Climate_Projections_with_Deep_Learning.ipynb).
Veri, eğitim ve kayıt işlemleri [istanbul_downscaling.py](istanbul_downscaling.py) içindedir.

## Çalıştırma

Mevcut `.venv` ortamı Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, xarray 2024.7.0,
netCDF4 1.7.2 ve PyTorch 2.8.0 ile doğrulandı. Yeni ortam için Python 3.11+ önerilir.
`requirements.txt` uyumlu sürüm aralıkları tanımlar; tam sürüm kilidi değildir.

```bash
python -m pip install -r requirements.txt
```

IDE'de proje klasörünü açın, `.venv` kernel'ini seçin ve notebook'ta **Restart Kernel → Run All** çalıştırın.
Hücreler veri indirmez ve paket kurmaz. Haritalar Cartopy veya çevrimiçi kıyı verisi gerektirmez.

Notebook'un tüm Python hücrelerini yeni süreçte yürütüp metin, tablo ve grafik çıktılarını kaydetmek için:

```bash
.venv/bin/python scripts/run_notebook.py --write
```

Bu küçük çalıştırıcı bu projenin düz Python hücrelerine özeldir; genel bir Jupyter çalıştırıcısı değildir.
`--write` verilmezse hücreleri çalıştırır ve deney dosyalarını üretir, notebook çıktılarını değiştirmez.

## Yerel veri

Aşağıdaki altı dosya gerekir:

```text
IST_domain/raw/era5_monthly/ERA5_1981_01.nc
IST_domain/raw/era5_monthly/ERA5_1981_02.nc
IST_domain/raw/era5_monthly/ERA5_1981_03.nc
IST_domain/raw/era5_land_monthly/ERA5_Land_1981_01.nc
IST_domain/raw/era5_land_monthly/ERA5_Land_1981_02.nc
IST_domain/raw/era5_land_monthly/ERA5_Land_1981_03.nc
```

Girdi: 850/700/500 hPa seviyelerinde `u`, `v`, `q`, `t`, `z`; 00/06/12/18 UTC.
Hedef: saatlik ERA5-Land `tp` ve `t2m`. Bölge: 40,5–42°K, 27,5–30,5°D.
Girdi 7×13, hedef 16×31 hücredir; 210 geçerli hedef hücresi kullanılır.

ERA5-Land 00 UTC yağış birikimi **önceki günün** toplamıdır. Aylar birleştirilerek
31 Ocak ve 28 Şubat korunur. 31 Mart için 1 Nisan 00 UTC kaydı eksik olduğundan
**89 tam gün** elde edilir; dışarıda kalan gün denetim dosyasında açıkça belirtilir.

[ECMWF birikim açıklaması](https://confluence.ecmwf.int/spaces/CKB/pages/462888259/ERA5%2Bfamily%2Bpost-processed%2Bdaily%2Bstatistics%2Bdocumentation).

## Deney ve değerlendirme

| Bölüm | Tarihler | Gün |
|---|---|---:|
| Eğitim | 1–31 Ocak 1981 | 31 |
| Doğrulama | 1–28 Şubat 1981 | 28 |
| Bağımsız test | 1–30 Mart 1981 | 30 |

Normalizasyon, kara maskesi ve referans tahmin ortalamaları yalnız eğitimden hesaplanır.
Model LeakyReLU ve pozitif Softplus çıktısı kullanan bir DeepESD uyarlamasıdır;
eğitim hücre ortalamaları çevresinde başlatılır. En fazla 200 epoch, 25 epoch sabır,
16 batch büyüklüğü, 0,001 öğrenme hızı ve seed 0 kullanılır. Şubat doğrulamasıyla
en iyi ağırlıklar seçilir. Mart sonuçları seçimden sonra hesaplanır.

DeepESD; sıfır yağış, eğitim genel ortalaması ve eğitim hücre ortalamasıyla aynı
gün ve hücrelerde karşılaştırılır. RMSE tüm gün×hücre kare hatalarının ortalamasının
kareköküdür. Islak hücre-gün eşiği 1 mm/gündür. SDII tüm geçerli ıslak hücre-günler
üzerinden hesaplanır; hiç ıslak hücre-gün yoksa NaN'dır. Kısa dönemden yıllık RX1day
veya iklimsel P98 sonucu çıkarılmaz.

ERA5-Land yağışı bağımsız yüksek çözünürlüklü gözlem değildir; ERA5 zorlamasına
dayanır. Üç aylık sonuçlar mevsimsel genelleme, bağımsız gözlemsel doğruluk veya
gelecek iklim becerisi göstermez. [ERA5-Land veri makalesi](https://essd.copernicus.org/articles/13/4349/2021/).

## Çıktılar

- `IST_domain/prepared/istanbul_1981_q1/`: günlük predictor ve hedef NetCDF dosyaları.
- `models/deepesd_istanbul_1981_q1.pt`: ağırlıklar, mimari, normalizasyon, maske, koordinatlar, değişken sırası/birimleri, veri özetleri ve deney ayarları.
- `results/istanbul_1981_q1/test_metrics.csv`: bağımsız test ve referans tahmin karşılaştırması.
- `results/istanbul_1981_q1/test_predictions.nc`: Mart referansı, DeepESD ve eğitim hücre ortalaması tahminleri.
- Aynı sonuç klasöründe: `training_history.csv`, `data_audit.json`, `run_summary.json` ve PNG haritaları/grafikleri.

Bir sonraki çalıştırma aynı deney adının üretilmiş dosyalarını günceller. Yeni deneyleri
saklamak için notebook'taki `RUN_NAME` değerini değiştirin.
Önceki `models/deepesd_pr_istanbul.pt` korunur. Düzenleme öncesi notebook yerel olarak
`backups/istanbul_before_cleanup_20260910.ipynb` altında saklanmıştır.
Ham veriler, yedekler ve üretilen dosyalar Git'e dahil edilmez.

CodeCarbon isteğe bağlıdır: paketi ayrıca kurup `TRACK_EMISSIONS = True` seçilebilir.
`OfflineEmissionsTracker` Türkiye için çevrimdışı tahmin üretir. Varsayılan çalıştırma
enerji takibi başlatmaz; eğitim süresini kaydeder.

## Kontroller

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Kontroller tarih kaymasını, aynı boyutta farklı ızgarayı, eksik ham saatleri,
ay sonu yağış birikimini, dönem çakışmasını, değişken sırasını, kara maskesini,
birimleri, normalizasyonu, referans tahminleri ve model kayıt/yükleme eşitliğini kapsar.

## Köken ve lisans

Özgün çalışma Jose González-Abad tarafından Climate Change AI için hazırlanmıştır:
[Statistical Downscaling of Climate Projections with Deep Learning](https://github.com/climatechange-ai-tutorials/downscaling-climate-projections).
Özgün Yeni Zelanda/CORDEX-ML-Bench anlatısı ve bu İstanbul/ERA5 deneyi farklı veri
kaynakları kullanır. Özgün U-Net örneği mevcut 7×13 girdi ve 210 maskeli çıktıyla
uyumlu olmadığından çalışan akışa dahil edilmemiştir.

Kod [MIT lisansı](LICENSE) altındadır. ERA5 ve ERA5-Land kaynaklarını ayrıca belirtin:
[ERA5 basınç seviyeleri](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels),
[ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land).

González-Abad, J. (2026). *Statistical Downscaling of Climate Projections with Deep Learning* [Tutorial].
Climate Change AI. https://doi.org/10.5281/zenodo.21446887
