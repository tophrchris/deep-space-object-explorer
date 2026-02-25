from __future__ import annotations

# Transitional bridge during Explorer split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

def compute_night_rating_details(
    hourly_weather_rows: list[dict[str, Any]],
    *,
    temperature_unit: str,
) -> dict[str, Any] | None:
    _refresh_legacy_globals()
    if not hourly_weather_rows:
        return None

    def _finite(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return float(numeric)

    def _average(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values)) / float(len(values))

    def _score_precip_accum_mm(accum_mm: float) -> float:
        if accum_mm <= 0.0:
            return 1.0
        if accum_mm <= 0.1:
            return 0.80
        if accum_mm <= 0.5:
            return 0.45
        if accum_mm <= 1.5:
            return 0.15
        return 0.0

    def _score_precip_probability(probability_pct: float) -> float:
        prob = max(0.0, min(100.0, float(probability_pct)))
        if prob <= 10.0:
            return 1.0
        if prob <= 20.0:
            return 0.90
        if prob <= 40.0:
            return 0.70
        if prob <= 60.0:
            return 0.45
        if prob <= 80.0:
            return 0.20
        return 0.05

    def _score_cloud_cover(cloud_cover_pct: float) -> float:
        if cloud_cover_pct <= 5.0:
            return 1.0
        if cloud_cover_pct <= 15.0:
            return 0.92
        if cloud_cover_pct <= 30.0:
            return 0.75
        if cloud_cover_pct <= 50.0:
            return 0.50
        if cloud_cover_pct <= 70.0:
            return 0.25
        return 0.05

    def _score_visibility_meters(distance_meters: float) -> float:
        miles = max(0.0, float(distance_meters)) * 0.000621371
        if miles > 6.0:
            return 1.0
        if miles >= 4.0:
            return 0.75
        if miles >= 2.0:
            return 0.40
        return 0.10

    def _score_wind_mph(wind_mph: float) -> float:
        if wind_mph <= 8.0:
            return 1.0
        if wind_mph <= 12.0:
            return 0.85
        if wind_mph <= 18.0:
            return 0.65
        if wind_mph <= 25.0:
            return 0.40
        return 0.20

    def _score_relative_humidity(rh_pct: float) -> float:
        if rh_pct <= 65.0:
            return 1.0
        if rh_pct <= 75.0:
            return 0.90
        if rh_pct <= 85.0:
            return 0.75
        if rh_pct <= 92.0:
            return 0.50
        return 0.30

    def _score_dew_spread_c(spread_c: float) -> float:
        spread_f = float(spread_c) * 9.0 / 5.0
        if spread_f >= 5.0:
            return 1.0
        if spread_f >= 3.0:
            return 0.80
        if spread_f >= 2.0:
            return 0.60
        if spread_f >= 1.0:
            return 0.40
        return 0.20

    precip_scores: list[float] = []
    precip_probability_scores: list[float] = []
    cloud_scores: list[float] = []
    visibility_scores: list[float] = []
    wind_scores: list[float] = []
    dew_scores: list[float] = []
    precip_accum_mm_values: list[float] = []
    wind_mph_values: list[float] = []

    for row in hourly_weather_rows:
        rain_mm = _finite(row.get("rain")) or 0.0
        showers_mm = _finite(row.get("showers")) or 0.0
        snowfall_cm = _finite(row.get("snowfall")) or 0.0
        precip_accum_mm = max(0.0, rain_mm + showers_mm + (snowfall_cm * 10.0))
        precip_accum_mm_values.append(precip_accum_mm)
        precip_scores.append(_score_precip_accum_mm(precip_accum_mm))

        precip_probability_pct = _finite(row.get("precipitation_probability"))
        if precip_probability_pct is not None:
            precip_probability_scores.append(_score_precip_probability(precip_probability_pct))

        cloud_cover = _finite(row.get("cloud_cover"))
        if cloud_cover is not None:
            cloud_scores.append(_score_cloud_cover(cloud_cover))

        visibility_meters = _finite(row.get("visibility"))
        if visibility_meters is not None:
            visibility_scores.append(_score_visibility_meters(visibility_meters))

        gust_kmh = _finite(row.get("wind_gusts_10m"))
        if gust_kmh is None:
            gust_kmh = _finite(row.get("wind_speed_10m"))
        if gust_kmh is not None:
            wind_mph = gust_kmh * 0.621371
            wind_mph_values.append(wind_mph)
            wind_scores.append(_score_wind_mph(wind_mph))

        dew_component_scores: list[float] = []
        humidity_pct = _finite(row.get("relative_humidity_2m"))
        if humidity_pct is not None:
            dew_component_scores.append(_score_relative_humidity(humidity_pct))
        temp_c = _finite(row.get("temperature_2m"))
        dew_c = _finite(row.get("dew_point_2m"))
        if temp_c is not None and dew_c is not None:
            spread_c = abs(temp_c - dew_c)
            dew_component_scores.append(_score_dew_spread_c(spread_c))
        if dew_component_scores:
            dew_scores.append(min(dew_component_scores))

    factor_scores: dict[str, float | None] = {
        "precipitation": _average(precip_scores),
        "precip_probability": _average(precip_probability_scores),
        "cloud_coverage": _average(cloud_scores),
        "visibility": _average(visibility_scores),
        "wind": _average(wind_scores),
        "dew_risk": _average(dew_scores),
    }
    weighted_total = 0.0
    available_weight = 0.0
    for factor_name, factor_weight in NIGHT_RATING_FACTOR_WEIGHTS.items():
        factor_score = factor_scores.get(factor_name)
        if factor_score is None:
            continue
        weighted_total += float(factor_weight) * float(factor_score)
        available_weight += float(factor_weight)

    if available_weight <= 0.0:
        return None

    normalized_score = weighted_total / available_weight
    raw_rating = int(np.clip(round(normalized_score * 5.0), 1, 5))
    rating = raw_rating
    rating_caps: list[dict[str, Any]] = []

    if precip_accum_mm_values:
        serious_precip_hours = sum(1 for value in precip_accum_mm_values if value >= 0.1)
        heavy_precip_hours = sum(1 for value in precip_accum_mm_values if value >= 0.5)
        total_precip_mm = float(sum(precip_accum_mm_values))
        precip_cap: int | None = None
        precip_reason = ""
        if heavy_precip_hours > 0 or total_precip_mm >= 2.0:
            precip_cap = 2
            precip_reason = "Heavy precipitation accumulation risk."
        elif serious_precip_hours > 0 or total_precip_mm >= 0.5:
            precip_cap = 3
            precip_reason = "Serious precipitation accumulation risk."
        elif total_precip_mm > 0.0:
            precip_cap = 4
            precip_reason = "Light precipitation accumulation risk."

        if precip_cap is not None and rating > precip_cap:
            rating = precip_cap
            rating_caps.append({"max_rating": precip_cap, "reason": precip_reason})

    cloud_score = factor_scores.get("cloud_coverage")
    visibility_score = factor_scores.get("visibility")
    if (
        wind_mph_values
        and cloud_score is not None
        and visibility_score is not None
        and cloud_score >= 0.80
        and visibility_score >= 0.80
    ):
        max_wind_mph = max(wind_mph_values)
        if max_wind_mph >= 20.0 and rating > 4:
            rating = 4
            rating_caps.append(
                {
                    "max_rating": 4,
                    "reason": "Strong wind prevents a 5/5 despite clear sky and visibility.",
                }
            )

    dew_score = factor_scores.get("dew_risk")
    if dew_score is not None and dew_score < 0.75 and rating > 4:
        rating = 4
        rating_caps.append(
            {
                "max_rating": 4,
                "reason": "Elevated dew risk prevents a 5/5 night.",
            }
        )

    emoji = NIGHT_RATING_EMOJIS.get(rating, "⭐️")
    factor_rows: list[dict[str, Any]] = []
    factor_definitions = (
        ("precipitation", "Precip accumulation risk", precip_scores),
        ("precip_probability", "Precip probability risk", precip_probability_scores),
        ("cloud_coverage", "Cloud cover quality", cloud_scores),
        ("visibility", "Visibility quality", visibility_scores),
        ("wind", "Wind stability", wind_scores),
        ("dew_risk", "Dew risk (humidity/spread)", dew_scores),
    )
    for factor_key, factor_label, factor_values in factor_definitions:
        factor_weight = float(NIGHT_RATING_FACTOR_WEIGHTS.get(factor_key, 0.0))
        factor_score = factor_scores.get(factor_key)
        data_hours = len(factor_values)
        pass_hours = sum(1 for value in factor_values if float(value) >= 0.70)
        factor_rows.append(
            {
                "key": factor_key,
                "label": factor_label,
                "weight": factor_weight,
                "score": factor_score,
                "data_hours": data_hours,
                "pass_hours": pass_hours,
                "weighted_contribution": (
                    factor_weight * float(factor_score) if factor_score is not None else None
                ),
            }
        )

    return {
        "rating": rating,
        "raw_rating": raw_rating,
        "emoji": emoji,
        "normalized_score": normalized_score,
        "available_weight": available_weight,
        "factors": factor_rows,
        "caps": rating_caps,
    }



def compute_night_rating(
    hourly_weather_rows: list[dict[str, Any]],
    *,
    temperature_unit: str,
) -> tuple[int, str] | None:
    _refresh_legacy_globals()
    details = compute_night_rating_details(
        hourly_weather_rows,
        temperature_unit=temperature_unit,
    )
    if details is None:
        return None
    return int(details["rating"]), str(details["emoji"])


def format_night_rating_tooltip(rating_details: dict[str, Any] | None) -> str:
    _refresh_legacy_globals()
    if not isinstance(rating_details, dict):
        return ""

    try:
        rating = int(rating_details.get("rating", 0))
    except (TypeError, ValueError):
        rating = 0
    try:
        raw_rating = int(rating_details.get("raw_rating", rating))
    except (TypeError, ValueError):
        raw_rating = rating
    emoji = str(rating_details.get("emoji", "")).strip()
    normalized_score = rating_details.get("normalized_score")
    available_weight = rating_details.get("available_weight")
    try:
        weighted_pct_text = f"{float(normalized_score) * 100.0:.0f}%"
    except (TypeError, ValueError):
        weighted_pct_text = "-"
    try:
        available_weight_pct_text = f"{float(available_weight) * 100.0:.0f}%"
    except (TypeError, ValueError):
        available_weight_pct_text = "-"

    lines: list[str] = [
        f"Night rating: {rating}/5 {emoji}".strip(),
        f"Weighted score: {weighted_pct_text} (weights in use: {available_weight_pct_text})",
    ]
    if raw_rating != rating:
        lines.append(f"Raw rating before caps: {raw_rating}/5")
    caps = rating_details.get("caps", [])
    if isinstance(caps, list) and caps:
        lines.append("Caps applied:")
        for cap in caps:
            if not isinstance(cap, dict):
                continue
            reason = str(cap.get("reason", "")).strip()
            try:
                max_rating = int(cap.get("max_rating", 0))
            except (TypeError, ValueError):
                max_rating = 0
            if max_rating > 0 and reason:
                lines.append(f"- max {max_rating}/5: {reason}")
            elif reason:
                lines.append(f"- {reason}")

    lines.append("Factors:")

    raw_factors = rating_details.get("factors", [])
    if isinstance(raw_factors, list):
        for factor in raw_factors:
            if not isinstance(factor, dict):
                continue
            label = str(factor.get("label", "")).strip()
            if not label:
                continue
            try:
                weight_pct = float(factor.get("weight", 0.0)) * 100.0
            except (TypeError, ValueError):
                weight_pct = 0.0
            score = factor.get("score")
            if score is None:
                lines.append(f"- {label} (w {weight_pct:.0f}%): no data")
                continue
            try:
                score_pct = float(score) * 100.0
            except (TypeError, ValueError):
                lines.append(f"- {label} (w {weight_pct:.0f}%): no data")
                continue
            pass_hours = int(factor.get("pass_hours", 0))
            data_hours = int(factor.get("data_hours", 0))
            lines.append(
                f"- {label} (w {weight_pct:.0f}%): {score_pct:.0f}% ({pass_hours}/{data_hours} hrs)"
            )

    return "\n".join(lines)

