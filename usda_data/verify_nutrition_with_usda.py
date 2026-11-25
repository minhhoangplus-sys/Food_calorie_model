"""
Đối chiếu nutrition_table.csv với USDA FoodData Central (API chính thức).

Với mỗi dòng trong CSV, script gọi API theo fdc_id và kiểm tra:
  1. Mô tả trên USDA có khớp cột usda_description không
  2. Năng lượng (kcal / 100 g) trên USDA có khớp cột kcal_per_100g không (sai số <= 1 kcal)
  3. standard_serving_g có trùng với một khẩu phần (household portion) nào USDA liệt kê không

Cách chạy (chỉ dùng thư viện chuẩn, cần Internet):
    python verify_nutrition_with_usda.py --csv nutrition_table.csv --api-key DEMO_KEY

Khuyên dùng khoá riêng (miễn phí, đăng ký tại https://fdc.nal.usda.gov/api-key-signup):
DEMO_KEY bị giới hạn khoảng 30 lượt gọi mỗi giờ, đủ cho 28 lớp nhưng chạy lại nhiều lần sẽ bị chặn.

Kết quả in ra màn hình và lưu vào usda_verification_report.csv
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={key}"


def fetch(fdc_id, key, retries=3):
    url = API.format(fdc_id=fdc_id, key=key)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("  (bị giới hạn lượt gọi, chờ 60 giây...)")
                time.sleep(60)
                continue
            if e.code == 404:
                return None
            raise
        except urllib.error.URLError:
            time.sleep(3)
    return "ERROR"


def get_kcal(food):
    """Lấy năng lượng kcal / 100 g. Ưu tiên nutrient id 1008, nếu thiếu thì thử 2047/2048 (Atwater)."""
    found = {}
    for fn in food.get("foodNutrients", []):
        nut = fn.get("nutrient", {})
        if nut.get("unitName", "").lower() == "kcal" and fn.get("amount") is not None:
            found[nut.get("id")] = fn["amount"]
    for nid in (1008, 2047, 2048):
        if nid in found:
            return found[nid]
    return None


def get_portions(food):
    out = []
    for p in food.get("foodPortions", []):
        g = p.get("gramWeight")
        if g is None:
            continue
        label = p.get("portionDescription") or p.get("modifier") or ""
        amt = p.get("amount")
        unit = (p.get("measureUnit") or {}).get("name", "")
        out.append((float(g), f"{amt if amt is not None else ''} {unit} {label}".strip()))
    return out


def norm(s):
    return " ".join(s.lower().replace('"', "").replace(",", " ").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--api-key", default="DEMO_KEY")
    ap.add_argument("--out", default="usda_verification_report.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    report = []
    n_bad = 0

    for r in rows:
        name, fdc = r["class_name"], r["fdc_id"]
        print(f"[{r['yolo_id']:>2}] {name:16s} fdc_id={fdc} ... ", end="", flush=True)
        food = fetch(fdc, args.api_key)
        if food in (None, "ERROR"):
            print("KHÔNG LẤY ĐƯỢC" if food == "ERROR" else "KHÔNG TỒN TẠI (fdc_id sai?)")
            report.append(dict(class_name=name, fdc_id=fdc, status="NOT_FOUND",
                               note="fdc_id không tồn tại hoặc không gọi được API"))
            n_bad += 1
            continue

        usda_desc = food.get("description", "")
        usda_kcal = get_kcal(food)
        csv_kcal = float(r["kcal_per_100g"])
        csv_serv = float(r["standard_serving_g"])
        portions = get_portions(food)

        desc_ok = norm(usda_desc) == norm(r["usda_description"])
        kcal_ok = usda_kcal is not None and abs(usda_kcal - csv_kcal) <= 1.0
        serv_match = [lab for g, lab in portions if abs(g - csv_serv) <= 0.6]
        # chấp nhận bội số 0.5 / 2 / 3 của một khẩu phần (vd 3 lát bacon, 2 cái pancake)
        serv_multiple = [lab for g, lab in portions
                         if any(abs(g * m - csv_serv) <= 0.6 for m in (0.5, 2, 3, 4))]

        problems = []
        if not desc_ok:
            problems.append(f"MÔ TẢ KHÁC: USDA='{usda_desc}'")
        if not kcal_ok:
            problems.append(f"KCAL KHÁC: USDA={usda_kcal} vs CSV={csv_kcal}")
        if not serv_match and not serv_multiple:
            problems.append(f"KHẨU PHẦN {csv_serv} g KHÔNG có trong danh sách USDA")

        status = "OK" if not problems else "CHECK"
        n_bad += status != "OK"
        print(status if not problems else "; ".join(problems))
        if status == "CHECK" and portions:
            print("      Khẩu phần USDA có: " + " | ".join(f"{g:g} g = {lab}" for g, lab in portions[:8]))

        report.append(dict(
            class_name=name, fdc_id=fdc, status=status,
            usda_description=usda_desc, csv_description=r["usda_description"],
            usda_kcal=usda_kcal, csv_kcal=csv_kcal,
            csv_standard_serving_g=csv_serv,
            serving_matches=";".join(serv_match) or ";".join(serv_multiple),
            usda_portions=" | ".join(f"{g:g} g = {lab}" for g, lab in portions),
            note="; ".join(problems),
        ))
        time.sleep(1.0)  # nhẹ tay với API

    fields = ["class_name", "fdc_id", "status", "usda_description", "csv_description", "usda_kcal",
              "csv_kcal", "csv_standard_serving_g", "serving_matches", "usda_portions", "note"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in report:
            w.writerow({k: row.get(k, "") for k in fields})

    print(f"\nTổng: {len(rows)} lớp | OK: {len(rows) - n_bad} | cần xem lại: {n_bad}")
    print(f"Đã lưu: {args.out}")
    if n_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
