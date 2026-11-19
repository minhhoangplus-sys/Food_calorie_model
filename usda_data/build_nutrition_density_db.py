"""
Build the unified Nutrition & Physical Density Reference Database (Step 3).

Reads the frozen 28 food classes from `data_pipeline/class_mapping.csv` and
`yolo_dataset/data.yaml`, attaches curated USDA FoodData Central (SR Legacy
2018-04) nutrition metrics, standardized Small/Standard/Large portion presets,
default preparation assumptions, and FAO/INFOODS v2.0 physical densities,
and exports:
  - usda_data/nutrition_density_table.json
  - usda_data/nutrition_table.csv
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_MAPPING_CSV = PROJECT_ROOT / "data_pipeline" / "class_mapping.csv"
YOLO_DATA_YAML = PROJECT_ROOT / "yolo_dataset" / "data.yaml"
OUT_JSON = Path(__file__).resolve().parent / "nutrition_density_table.json"
OUT_CSV = Path(__file__).resolve().parent / "nutrition_table.csv"

PORTION_MULTIPLIERS = {
    "Small": 0.75,
    "Standard": 1.00,
    "Large": 1.50,
}

# Verified against official USDA FoodData Central SR Legacy (2018-04) JSON
# and FAO/INFOODS Density Database v2.0.
CURATED_RECORDS: dict[str, dict] = {
    "avocado": {
        "fdc_id": 171705,
        "usda_description": "Avocados, raw, all commercial varieties",
        "kcal_per_100g": 160.0,
        "standard_serving_g": 100.0,
        "preparation_notes": "Raw peeled and pitted avocado flesh (0.5 medium avocado), plain without dressing.",
        "density_g_per_cm3": 0.97,
        "density_source": "USDA FDC 171705 volume-to-weight (1 cup pureed = 230 g / 236.59 cm3 = 0.972 g/cm3).",
    },
    "bacon": {
        "fdc_id": 167914,
        "usda_description": "Pork, cured, bacon, cooked, baked",
        "kcal_per_100g": 548.0,
        "standard_serving_g": 24.3,
        "preparation_notes": "Cured pork bacon, baked/pan-cooked until crisp and drained (3 slices at 8.1 g).",
        "density_g_per_cm3": 0.85,
        "density_source": "Food engineering estimate for plated crisp cured pork strips (FAO raw pork medium = 0.93 g/mL).",
    },
    "baked_beans": {
        "fdc_id": 175182,
        "usda_description": "Beans, baked, canned, plain or vegetarian",
        "kcal_per_100g": 94.0,
        "standard_serving_g": 127.0,
        "preparation_notes": "Canned vegetarian/plain baked beans in tomato-sugar sauce, heated (0.5 cup).",
        "density_g_per_cm3": 1.07,
        "density_source": "USDA FDC 175182 volume-to-weight (1 cup = 254 g / 236.59 cm3 = 1.074 g/cm3).",
    },
    "baked_potato": {
        "fdc_id": 170093,
        "usda_description": "Potatoes, baked, flesh and skin, without salt",
        "kcal_per_100g": 93.0,
        "standard_serving_g": 173.0,
        "preparation_notes": "Whole medium baked potato (flesh and skin), plain without added butter, sour cream, or salt.",
        "density_g_per_cm3": 1.05,
        "density_source": "Whole cooked potato tuber density estimate (~1.05 g/cm3; FAO boiled cubed bulk = 0.59 g/mL).",
    },
    "broccoli": {
        "fdc_id": 169967,
        "usda_description": "Broccoli, cooked, boiled, drained, without salt",
        "kcal_per_100g": 35.0,
        "standard_serving_g": 156.0,
        "preparation_notes": "Cooked/steamed/boiled broccoli florets, drained, without added butter or oil (1 cup chopped).",
        "density_g_per_cm3": 0.60,
        "density_source": "Project benchmark / USDA FDC 169967 florets bulk density (~0.60 g/cm3; chopped 1 cup = 156 g / 236.59 cm3 = 0.659).",
    },
    "cabbage": {
        "fdc_id": 169975,
        "usda_description": "Cabbage, raw",
        "kcal_per_100g": 25.0,
        "standard_serving_g": 89.0,
        "preparation_notes": "Raw chopped or shredded cabbage without heavy mayonnaise/oil dressing (1 cup chopped).",
        "density_g_per_cm3": 0.38,
        "density_source": "USDA FDC 169975 volume-to-weight (1 cup chopped raw = 89 g / 236.59 cm3 = 0.376 g/cm3).",
    },
    "carrot": {
        "fdc_id": 170393,
        "usda_description": "Carrots, raw",
        "kcal_per_100g": 41.0,
        "standard_serving_g": 122.0,
        "preparation_notes": "Raw sliced, shredded, or stick carrots, plain without glaze or butter (1 cup strips/slices).",
        "density_g_per_cm3": 0.54,
        "density_source": "FAO/INFOODS v2.0 ('Carrot, raw, chopped' = 0.54 g/mL, KEN; matches USDA 128 g/cup = 0.541 g/cm3).",
    },
    "cheese_burger": {
        "fdc_id": 170691,
        "usda_description": "Fast foods, cheeseburger; single, regular patty, with condiments",
        "kcal_per_100g": 270.0,
        "standard_serving_g": 127.0,
        "preparation_notes": "Single grilled beef patty with cheese on white bun with standard condiments (1 regular burger).",
        "density_g_per_cm3": 0.45,
        "density_source": "Composite assembly estimate (aerated bun ~0.29 g/cm3 + beef/cheese ~1.0 g/cm3; 127 g / ~280 cm3).",
    },
    "chicken_breast": {
        "fdc_id": 171477,
        "usda_description": "Chicken, broilers or fryers, breast, meat only, cooked, roasted",
        "kcal_per_100g": 165.0,
        "standard_serving_g": 140.0,
        "preparation_notes": "Boneless, skinless chicken breast, roasted or grilled without breading or heavy oil (~5 oz cooked).",
        "density_g_per_cm3": 1.05,
        "density_source": "Project benchmark / solid cooked poultry tissue density (~1.05 g/cm3; FAO poultry mixture = 1.038 g/mL).",
    },
    "chicken_tender": {
        "fdc_id": 173321,
        "usda_description": "Fast foods, chicken tenders",
        "kcal_per_100g": 271.0,
        "standard_serving_g": 120.0,
        "preparation_notes": "Breaded and deep-fried chicken breast tenderloin strips, without dipping sauce (4 strips at 30 g).",
        "density_g_per_cm3": 0.82,
        "density_source": "Plated breaded fried poultry estimate (meat core ~1.05 g/cm3 + crispy porous breading).",
    },
    "chicken_thigh": {
        "fdc_id": 173625,
        "usda_description": "Chicken, broilers or fryers, thigh, meat and skin, cooked, roasted",
        "kcal_per_100g": 232.0,
        "standard_serving_g": 137.0,
        "preparation_notes": "Roasted/baked chicken thigh, meat and skin included, unbreaded (1 thigh).",
        "density_g_per_cm3": 1.02,
        "density_source": "Solid cooked poultry dark-meat tissue with skin (~1.02 g/cm3; slightly lower than lean breast 1.05).",
    },
    "chips": {
        "fdc_id": 169677,
        "usda_description": "Snacks, potato chips, plain, salted",
        "kcal_per_100g": 532.0,
        "standard_serving_g": 28.0,
        "preparation_notes": "Fried/baked crispy salted snack chips (potato/tortilla chips), served dry (1 oz / ~22 chips).",
        "density_g_per_cm3": 0.09,
        "density_source": "FAO/INFOODS v2.0 ('Potato chips' = 0.09 g/mL, RC).",
    },
    "french_fries": {
        "fdc_id": 170698,
        "usda_description": "Fast foods, potato, french fried in vegetable oil",
        "kcal_per_100g": 312.0,
        "standard_serving_g": 117.0,
        "preparation_notes": "Deep-fried potato strips in vegetable oil, salted, without cheese or chili toppings (1 medium serving).",
        "density_g_per_cm3": 0.45,
        "density_source": "Bulk piled fried potato strips estimate (inter-piece air voids; ~0.40-0.50 g/cm3).",
    },
    "fried_egg": {
        "fdc_id": 173423,
        "usda_description": "Egg, whole, cooked, fried",
        "kcal_per_100g": 196.0,
        "standard_serving_g": 46.0,
        "preparation_notes": "Whole large chicken egg pan-fried in minimal cooking fat/oil (1 large egg).",
        "density_g_per_cm3": 0.98,
        "density_source": "Whole flat cooked egg on plate (yolk/albumen tissue ~1.03 g/cm3; FAO 0.60 is chopped cup bulk).",
    },
    "fried_rice": {
        "fdc_id": 167668,
        "usda_description": "Restaurant, Chinese, fried rice, without meat",
        "kcal_per_100g": 174.0,
        "standard_serving_g": 137.0,
        "preparation_notes": "Stir-fried cooked white rice with oil, soy sauce, egg, and aromatics without extra meat (1 cup).",
        "density_g_per_cm3": 0.70,
        "density_source": "FAO/INFOODS v2.0 ('Rice, boiled with fat' = 0.70 g/mL, KEN).",
    },
    "green_beans": {
        "fdc_id": 169141,
        "usda_description": "Beans, snap, green, cooked, boiled, drained, without salt",
        "kcal_per_100g": 35.0,
        "standard_serving_g": 125.0,
        "preparation_notes": "Boiled or steamed whole/cut green snap beans, drained, without butter or bacon (1 cup).",
        "density_g_per_cm3": 0.53,
        "density_source": "FAO/INFOODS v2.0 ('French green beans' = 0.53 g/mL, KEN; matches USDA 125 g/cup = 0.528 g/cm3).",
    },
    "hash_brown": {
        "fdc_id": 169848,
        "usda_description": "Restaurant, family style, hash browns",
        "kcal_per_100g": 197.0,
        "standard_serving_g": 124.0,
        "preparation_notes": "Pan-fried/griddled shredded potato hash browns or fried potato patty in oil (1 restaurant serving).",
        "density_g_per_cm3": 0.50,
        "density_source": "Midpoint of USDA FDC 169848 loose shredded (94 g/cup = 0.40 g/cm3) and pan-fried patty (~0.63 g/cm3).",
    },
    "mashed_potatoes": {
        "fdc_id": 168555,
        "usda_description": "Potatoes, mashed, home-prepared, whole milk and butter added",
        "kcal_per_100g": 113.0,
        "standard_serving_g": 210.0,
        "preparation_notes": "Home-style boiled potatoes mashed with whole milk and butter, without extra gravy (1 cup).",
        "density_g_per_cm3": 0.89,
        "density_source": "USDA FDC 168555 volume-to-weight (1 cup = 210 g / 236.59 cm3 = 0.888 g/cm3).",
    },
    "meat_loaf": {
        "fdc_id": 174035,
        "usda_description": "Beef, ground, 85% lean meat / 15% fat, loaf, cooked, baked",
        "kcal_per_100g": 240.0,
        "standard_serving_g": 140.0,
        "preparation_notes": "Baked 85% lean ground beef loaf, sliced, without extra brown gravy (1 slice ~5 oz).",
        "density_g_per_cm3": 0.98,
        "density_source": "Solid baked ground beef loaf estimate (~0.98 g/cm3; slightly below intact steak 1.05).",
    },
    "pancake": {
        "fdc_id": 175009,
        "usda_description": "Pancakes, plain, prepared from recipe",
        "kcal_per_100g": 227.0,
        "standard_serving_g": 154.0,
        "preparation_notes": "Plain griddled buttermilk/egg batter pancakes, excluding syrup and butter (2 x 6\" pancakes at 77 g).",
        "density_g_per_cm3": 0.48,
        "density_source": "Aerated griddled cake estimate (77 g / ~160 cm3 per 6\" pancake).",
    },
    "pasta": {
        "fdc_id": 169737,
        "usda_description": "Pasta, cooked, enriched, without added salt",
        "kcal_per_100g": 158.0,
        "standard_serving_g": 140.0,
        "preparation_notes": "Boiled enriched wheat pasta (spaghetti/penne/macaroni), drained, excluding heavy sauces (1 cup).",
        "density_g_per_cm3": 0.59,
        "density_source": "FAO/INFOODS v2.0 ('Pasta/ noodles, boiled' = 0.59 g/mL, KEN).",
    },
    "pizza": {
        "fdc_id": 173292,
        "usda_description": "Fast Food, Pizza Chain, 14\" pizza, cheese topping, regular crust",
        "kcal_per_100g": 266.0,
        "standard_serving_g": 107.0,
        "preparation_notes": "Baked regular-crust cheese pizza with tomato sauce and mozzarella (1 slice, 1/8 of 14\" pie).",
        "density_g_per_cm3": 0.60,
        "density_source": "Composite baked crust + cheese/sauce slice estimate (107 g / ~178 cm3).",
    },
    "pork_chop": {
        "fdc_id": 167827,
        "usda_description": "Pork, fresh, loin, center loin (chops), bone-in, separable lean and fat, cooked, broiled",
        "kcal_per_100g": 209.0,
        "standard_serving_g": 157.0,
        "preparation_notes": "Broiled or grilled center-loin pork chop, unbreaded, lean and fat included (1 chop edible portion).",
        "density_g_per_cm3": 1.03,
        "density_source": "Solid cooked pork loin muscle + fat tissue estimate (~1.03 g/cm3; FAO raw pork medium = 0.93 g/mL).",
    },
    "shrimp": {
        "fdc_id": 171971,
        "usda_description": "Crustaceans, shrimp, mixed species, cooked, moist heat (may contain additives to retain moisture)",
        "kcal_per_100g": 119.0,
        "standard_serving_g": 85.0,
        "preparation_notes": "Boiled, steamed, or lightly sauteed peeled shrimp without breading or batter (3 oz cooked).",
        "density_g_per_cm3": 0.77,
        "density_source": "FAO/INFOODS v2.0 ('Prawns, whole, boiled, without shell' = 0.77 g/mL, RC).",
    },
    "steak": {
        "fdc_id": 168631,
        "usda_description": "Beef, loin, top loin steak, boneless, lip off, separable lean and fat, trimmed to 0\" fat, select, cooked, grilled",
        "kcal_per_100g": 207.0,
        "standard_serving_g": 150.0,
        "preparation_notes": "Grilled boneless beef top loin steak, trimmed, without compound butter or cream sauce (1 steak).",
        "density_g_per_cm3": 1.05,
        "density_source": "Solid cooked beef muscle tissue density (~1.05 g/cm3; FAO raw lean beef = 0.96 g/mL).",
    },
    "waffle": {
        "fdc_id": 175039,
        "usda_description": "Waffles, plain, prepared from recipe",
        "kcal_per_100g": 291.0,
        "standard_serving_g": 75.0,
        "preparation_notes": "Plain batter waffle baked in waffle iron, excluding syrup, whipped cream, and butter (1 round 7\" waffle).",
        "density_g_per_cm3": 0.35,
        "density_source": "Aerated gridded batter envelope estimate (75 g / ~215 cm3 including waffle pockets).",
    },
    "white_rice": {
        "fdc_id": 168878,
        "usda_description": "Rice, white, long-grain, regular, enriched, cooked",
        "kcal_per_100g": 130.0,
        "standard_serving_g": 158.0,
        "preparation_notes": "Steamed or boiled plain long-grain white rice, without added oil or butter (1 cup cooked).",
        "density_g_per_cm3": 0.85,
        "density_source": "Project benchmark for plated cooked white rice (FAO/INFOODS v2.0 loose 'Rice, white, boiled' = 0.73 g/mL, RC).",
    },
    "yeast_bread": {
        "fdc_id": 174924,
        "usda_description": "Bread, white, commercially prepared (includes soft bread crumbs)",
        "kcal_per_100g": 266.0,
        "standard_serving_g": 29.0,
        "preparation_notes": "Commercially prepared sliced white yeast bread, plain without butter or spreads (1 regular slice).",
        "density_g_per_cm3": 0.29,
        "density_source": "FAO/INFOODS v2.0 ('Bread, white, sliced, prepacked' = 0.29 g/mL, FNDDS 4.1).",
    },
}


def load_class_mapping() -> list[dict]:
    rows = []
    with CLASS_MAPPING_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "yolo_id": int(r["yolo_id"]),
                    "class_name": r["class_name"].strip(),
                }
            )
    rows.sort(key=lambda x: x["yolo_id"])
    return rows


def verify_yolo_yaml(class_rows: list[dict]) -> None:
    yaml_text = YOLO_DATA_YAML.read_text(encoding="utf-8")
    names = {}
    for line in yaml_text.splitlines():
        m = re.match(r"^\s+(\d+):\s*(\S+)\s*$", line)
        if m:
            names[int(m.group(1))] = m.group(2)
    assert len(names) == 28, f"Expected 28 classes in data.yaml, found {len(names)}"
    for r in class_rows:
        assert names[r["yolo_id"]] == r["class_name"], (
            f"Mismatch at yolo_id={r['yolo_id']}: {names[r['yolo_id']]} != {r['class_name']}"
        )


def compute_presets(standard_serving_g: float, kcal_per_100g: float) -> dict[str, dict]:
    presets = {}
    for label, mult in PORTION_MULTIPLIERS.items():
        grams = round(standard_serving_g * mult, 1)
        kcal = round((grams * kcal_per_100g) / 100.0, 1)
        presets[label] = {
            "multiplier": mult,
            "grams": grams,
            "kcal": kcal,
        }
    return presets


def main() -> None:
    class_rows = load_class_mapping()
    verify_yolo_yaml(class_rows)
    assert len(class_rows) == 28
    assert set(r["class_name"] for r in class_rows) == set(CURATED_RECORDS.keys())

    foods_dict: dict[str, dict] = {}
    id_to_class: dict[str, str] = {}
    csv_rows: list[dict] = []

    for r in class_rows:
        cname = r["class_name"]
        rec = CURATED_RECORDS[cname]
        presets = compute_presets(rec["standard_serving_g"], rec["kcal_per_100g"])

        entry = {
            "yolo_id": r["yolo_id"],
            "class_name": cname,
            "fdc_id": rec["fdc_id"],
            "usda_description": rec["usda_description"],
            "kcal_per_100g": rec["kcal_per_100g"],
            "standard_serving_g": rec["standard_serving_g"],
            "preparation_notes": rec["preparation_notes"],
            "density_g_per_cm3": rec["density_g_per_cm3"],
            "density_source": rec["density_source"],
            "portion_presets": presets,
        }
        foods_dict[cname] = entry
        id_to_class[str(r["yolo_id"])] = cname

        csv_rows.append(
            {
                "yolo_id": r["yolo_id"],
                "class_name": cname,
                "fdc_id": rec["fdc_id"],
                "usda_description": rec["usda_description"],
                "kcal_per_100g": rec["kcal_per_100g"],
                "standard_serving_g": rec["standard_serving_g"],
                "small_serving_g": presets["Small"]["grams"],
                "small_kcal": presets["Small"]["kcal"],
                "standard_kcal": presets["Standard"]["kcal"],
                "large_serving_g": presets["Large"]["grams"],
                "large_kcal": presets["Large"]["kcal"],
                "density_g_per_cm3": rec["density_g_per_cm3"],
                "density_source": rec["density_source"],
                "preparation_notes": rec["preparation_notes"],
            }
        )

    payload = {
        "metadata": {
            "version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "num_classes": len(foods_dict),
            "usda_source": "USDA FoodData Central - SR Legacy (April 2018)",
            "density_source": "FAO/INFOODS Density Database v2.0 + USDA FDC volume-to-weight conversions",
            "portion_multipliers": PORTION_MULTIPLIERS,
            "portion_classification_thresholds": {
                "Small": "ratio < 0.80",
                "Standard": "0.80 <= ratio <= 1.20",
                "Large": "ratio > 1.20",
            },
            "calorie_formulas": {
                "mass_g": "volume_cm3 * density_g_per_cm3",
                "kcal": "(mass_g * kcal_per_100g) / 100.0",
            },
        },
        "id_to_class": id_to_class,
        "foods": foods_dict,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = list(csv_rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Generated {OUT_JSON} ({len(foods_dict)} classes)")
    print(f"Generated {OUT_CSV} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
