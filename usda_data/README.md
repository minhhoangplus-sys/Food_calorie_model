# USDA Nutrition & Physical Density Database (`usda_data/`)

Single Source of Truth for **Step 3: Nutrition & Density Database Curation** covering the 28 frozen VFN food categories (`yolo_id` `0`–`27`).

## Deliverables

- [`nutrition_density_table.json`](nutrition_density_table.json): Structured JSON lookup table for the Step 5 3D calorie engine (`predict.py`), Step 6 evaluation, and ICT handover. Contains only core fields needed for lookup, calorie/volume math, portion presets, and USDA/FAO provenance.
- [`nutrition_table.csv`](nutrition_table.csv): Flat 28-row reference table for quick inspection and academic reporting.
- [`build_nutrition_density_db.py`](build_nutrition_density_db.py): Reproducible generator script that cross-checks class IDs and names against `data_pipeline/class_mapping.csv` and `yolo_dataset/data.yaml` before writing both deliverables.

## Official Data Sources & Download Links

1. **USDA FoodData Central — SR Legacy (April 2018):**
   - Portal: https://fdc.nal.usda.gov/
   - Direct Dataset ZIP (JSON, 7,793 foods): https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_json_2018-04.zip
2. **FAO/INFOODS Density Database (Version 2.0):**
   - Portal: https://www.fao.org/infoods/infoods/tables-and-databases/faoinfoods-databases/en/
   - Direct Excel File (666 entries): https://www.fao.org/fileadmin/templates/food_composition/documents/density_DB_v2_0_final-1__1_.xlsx

## Data Provenance & Methodology (For Academic Reporting)

### 1. Nutritional Values & Reference Servings (USDA FDC)
- **`fdc_id`, `usda_description`, `kcal_per_100g`:** Extracted verbatim and verified 28/28 against `FoodData_Central_sr_legacy_food_json_2018-04.json`.
- **`standard_serving_g` & `preparation_notes`:** Derived from official USDA `foodPortions` for each `fdc_id` under the dominant ready-to-eat preparation observed in `vfn_1_0` (e.g., roasted/grilled meats without sauces, boiled rice/pasta, standard fast-food items). Where USDA lists a single unit or 0.5-cup unit, realistic meal-plate multiples are used (e.g., `bacon`: 3 slices × 8.1 g = `24.3 g`; `pancake`: 2 × 77 g = `154.0 g`; `chicken_tender`: 4 strips × 30 g = `120.0 g`; `baked_beans`: 0.5 cup = `127.0 g`; `broccoli`: 1 cup = `156.0 g`; `avocado`: 0.5 fruit ≈ `100.0 g`; `pasta`: `140.0 g` midpoint of 124–151 g/cup).

### 2. Computed Portion Presets (Step 3 Formulas)
- `small_serving_g` = `0.75 × standard_serving_g`
- `large_serving_g` = `1.50 × standard_serving_g`
- `small_kcal`, `standard_kcal`, `large_kcal` = `(grams × kcal_per_100g) / 100`

### 3. Physical Density $\rho$ ($\text{g/cm}^3$) Breakdown
Because FAO/INFOODS v2.0 mainly catalogs raw commodities, grains, and basic ingredients rather than prepared American fast-food/plated dishes, physical densities are grouped into 4 transparent tiers (also recorded per item in `density_source`):

1. **Direct FAO/INFOODS v2.0 entries (7 classes):**
   - `carrot` (`0.54` — *Carrot, raw, chopped*, KEN)
   - `chips` (`0.09` — *Potato chips*, RC)
   - `fried_rice` (`0.70` — *Rice, boiled with fat*, KEN)
   - `green_beans` (`0.53` — *French green beans*, KEN)
   - `pasta` (`0.59` — *Pasta/ noodles, boiled*, KEN)
   - `shrimp` (`0.77` — *Prawns, whole, boiled, without shell*, RC)
   - `yeast_bread` (`0.29` — *Bread, white, sliced, prepacked*, FNDDS 4.1)
2. **USDA Cup-to-Volume conversions ($\text{gramWeight} / 236.59\text{ cm}^3$, 4 classes):**
   - `avocado` (`0.97` — 230 g/cup pureed)
   - `baked_beans` (`1.07` — 254 g/cup)
   - `cabbage` (`0.38` — 89 g/cup chopped raw)
   - `mashed_potatoes` (`0.89` — 210 g/cup)
3. **Project Implementation Plan Benchmarks (3 classes):**
   - `white_rice` (`0.85` — plated cooked white rice benchmark; FAO loose boiled = `0.73`)
   - `chicken_breast` (`1.05` — solid cooked poultry tissue benchmark)
   - `broccoli` (`0.60` — cooked florets bulk benchmark; USDA chopped cup = `0.66`)
4. **Food-Engineering / Geometry Estimates (14 classes):**
   - Solid cooked meat/tuber/egg tissue (~0.98–1.05 $\text{g/cm}^3$): `steak` (`1.05`), `baked_potato` (`1.05`), `pork_chop` (`1.03`), `chicken_thigh` (`1.02`), `fried_egg` (`0.98`), `meat_loaf` (`0.98`), `bacon` (`0.85`), `chicken_tender` (`0.82` with porous breading).
   - Porous / piled / composite assemblies (~0.35–0.60 $\text{g/cm}^3$): `pizza` (`0.60`), `hash_brown` (`0.50`), `pancake` (`0.48`), `cheese_burger` (`0.45`), `french_fries` (`0.45`), `waffle` (`0.35`).

## Step 5 Calorie Math

```text
Mass (g) = Volume (cm³) × density_g_per_cm3
Item kcal = (Mass (g) × kcal_per_100g) / 100
```

Automatic portion fallback compares `Mass (g) / standard_serving_g`:
- `< 0.80` → `Small`
- `0.80 – 1.20` → `Standard`
- `> 1.20` → `Large`

## Regenerate

```bash
python usda_data/build_nutrition_density_db.py
```
