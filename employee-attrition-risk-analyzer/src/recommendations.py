def recommendations_from_factors(factors):
    recommendations = []

    text = " ".join(f[0].lower() for f in factors)

    if "overtime" in text:
        recommendations.append("Review workload, overtime frequency, and staffing.")
    if "satisfaction" in text:
        recommendations.append("Consider a structured employee check-in.")
    if "promotion" in text:
        recommendations.append("Review career development and progression opportunities.")
    if "work-life" in text:
        recommendations.append("Review work-life balance and workload expectations.")
    if "distance" in text:
        recommendations.append("Discuss practical commuting or work-location constraints where appropriate.")
    if "income" in text:
        recommendations.append("Consider compensation benchmarking where appropriate.")

    if not recommendations:
        recommendations.append("No strong rule-based review signals were identified; use normal HR review processes.")

    return recommendations
