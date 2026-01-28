# UGV Navigation System - Modular Architecture v2.0

Surrogate-Assisted Receding-Horizon Planning Under Field-of-View Constraints.

## Özellikler

### 🎯 Temel İyileştirmeler

1. **Adaptive FoV (Alan Görüşü)**
   - Tıkanıklık durumunda FoV otomatik genişler (25 → 50 cells)
   - Başarılı navigasyonda geri daralır
   - Exponential expansion stratejisi

2. **Multi-Strategy Recovery System**
   - `expand_fov`: FoV genişletip yeniden planlama
   - `backtrack`: Geçmiş yolda geri gitme
   - `random_escape`: Rastgele kaçış
   - `global_replan`: Global A* ile yeniden planlama
   - `wall_follow`: Duvar takibi

3. **Global Memory**
   - Ziyaret edilen hücreler takip edilir
   - Başarısız hücreler işaretlenir ve cezalandırılır
   - Döngü tespiti ve önleme

4. **Modular SOLID Architecture**
   - Her modül tek sorumluluk
   - Kolay test ve genişletme
   - Dependency injection

## Proje Yapısı

```
ugv_nav/
├── config/          # Konfigürasyon yönetimi
│   └── settings.py  # Tüm parametreler
├── terrain/         # Arazi modelleme
│   ├── types.py     # TerrainType enum
│   └── generator.py # Harita üreteci
├── environment/     # Çevre temsili
│   ├── world.py     # Global Environment
│   └── local_view.py # FoV-kısıtlı LocalEnvironment
├── energy/          # Enerji modeli
│   └── model.py     # Fizik tabanlı enerji hesaplama
├── planning/        # Planlama algoritmaları
│   ├── astar.py     # A* planner
│   └── receding_horizon.py # Ana kontrol döngüsü
├── recovery/        # Kurtarma sistemi
│   ├── adaptive_fov.py # Adaptif FoV
│   └── strategies.py   # Kurtarma stratejileri
├── optimization/    # GA ve Surrogate
│   ├── ga/          # Genetik algoritma
│   └── surrogate/   # Surrogate modeller
├── metrics/         # Metrik ve sınıflandırma
├── visualization/   # Canlı izleme
│   └── monitor.py   # Debug görselleştirme
├── pipeline/        # Deney yönetimi
│   └── runner.py    # ExperimentRunner
├── main.py          # CLI giriş noktası
└── test_system.py   # Test scripti
```

## Kullanım

### Hızlı Test
```bash
cd ugv_nav
python test_system.py
```

### Tek Senaryo Çalıştırma
```bash
python main.py test --seed 42 --verbose
```

### Debug Modu (Canlı Görselleştirme)
```bash
python main.py debug --seed 42
```

### Tam Deney Suite
```bash
python main.py suite --num_scenarios 30 --output results/
```

### Google Colab
```python
from ugv_nav import Config, ExperimentRunner

config = Config()
runner = ExperimentRunner(config)
results = runner.run_suite(num_scenarios=30)
```

## Konfigürasyon

```python
from ugv_nav import Config

config = Config()

# FoV ayarları
config.fov.base_radius_cells = 25
config.fov.max_radius_cells = 50

# Recovery ayarları
config.recovery.enabled = True
config.recovery.strategies = ('expand_fov', 'backtrack', 'global_replan')

# Unknown terrain modeli
config.unknown.mode = 'adaptive'  # 'optimistic', 'balanced', 'pessimistic'
```

## Metotlar

| Metot | Açıklama |
|-------|----------|
| `full_map_energy` | Tam harita A* (enerji optimum) - Baseline |
| `full_map_time` | Tam harita A* (zaman optimum) |
| `fov_energy` | FoV-kısıtlı A* + Recovery |
| `fov_time` | FoV-kısıtlı A* (zaman modu) |
| `fov_ga` | FoV + GA iyileştirme |
| `fov_ga_surrogate` | FoV + GA + Surrogate |

## Test Sonuçları

Seed=42 ile tek senaryo testi:
- **full_map_energy**: ✓ success (baseline)
- **fov_energy**: ✓ success (67 replan, 0 recovery)

## Gereksinimler

```
numpy
scipy
matplotlib
scikit-learn (surrogate için opsiyonel)
```

## Lisans

MIT License

## İletişim

Sorular için: [berkeogurlu@gmail.com]
