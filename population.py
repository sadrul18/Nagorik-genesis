"""
Population generation module for নাগরিক-GENESIS (NAGORIK-GENESIS).
Generate synthetic Bangladeshi citizen populations with controllable distributions.
"""
import numpy as np
from typing import List, Optional
from data_models import (
    Citizen, CITY_ZONES, INCOME_LEVELS, POLITICAL_VIEWS,
    GENDERS, EDUCATION_LEVELS, DIVISIONS, RELIGIONS
)


# Profession mappings by income level — Bangladesh context
PROFESSIONS_BY_INCOME = {
    "low": [
        "Rickshaw Puller", "Day Laborer", "Garment Worker", "Street Vendor",
        "Domestic Worker", "Fisherman", "Farmer", "Construction Worker",
        "Tea Stall Worker", "Auto-Rickshaw Driver", "Security Guard", "Cleaner"
    ],
    "middle": [
        "School Teacher", "Government Officer", "Bank Officer", "Small Shop Owner",
        "NGO Worker", "Police Officer", "Pharmacist", "Nurse",
        "IT Professional", "Accountant", "Journalist", "Army Personnel"
    ],
    "high": [
        "Garment Factory Owner", "Real Estate Developer", "Senior Doctor",
        "Senior Government Official", "IT Company Owner", "Importer/Exporter",
        "University Professor", "Corporate Executive", "Lawyer", "Business Tycoon"
    ]
}

# Division population weights (approximate share of Bangladesh population)
DIVISION_WEIGHTS = [0.30, 0.18, 0.12, 0.10, 0.06, 0.07, 0.10, 0.07]
# Dhaka 30%, Chittagong 18%, Rajshahi 12%, Khulna 10%, Barisal 6%, Sylhet 7%, Rangpur 10%, Mymensingh 7%

# Religion distribution (BD demographics: ~91% Muslim, ~8% Hindu, ~1% other)
RELIGION_WEIGHTS = [0.910, 0.080, 0.004, 0.002, 0.004]

# Gender distribution (BD: ~50.5% male, ~49.2% female, ~0.3% hijra)
GENDER_WEIGHTS = [0.505, 0.492, 0.003]


def generate_population(
    size: int,
    seed: Optional[int] = None,
    low_share: float = 0.55,
    middle_share: float = 0.35,
    high_share: float = 0.10
) -> List[Citizen]:
    """
    Generate a synthetic population of Bangladeshi citizens.

    Args:
        size: Number of citizens to generate (up to 50,000).
        seed: Random seed for reproducibility.
        low_share: Proportion of low-income citizens (default 0.55).
        middle_share: Proportion of middle-income citizens (default 0.35).
        high_share: Proportion of high-income citizens (default 0.10).

    Returns:
        List of Citizen objects with randomized but realistic Bangladeshi attributes.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Validate shares sum to approximately 1.0
    total_share = low_share + middle_share + high_share
    if not (0.99 <= total_share <= 1.01):
        raise ValueError(f"Income shares must sum to 1.0, got {total_share}")

    # Calculate counts for each income level
    low_count = int(size * low_share)
    middle_count = int(size * middle_share)
    high_count = size - low_count - middle_count  # Ensure exact total

    # Create income level assignments
    income_levels = (
        ["low"] * low_count +
        ["middle"] * middle_count +
        ["high"] * high_count
    )
    rng.shuffle(income_levels)

    # Precompute age weights (youth bulge: BD median age ~27)
    age_range = np.arange(15, 71)
    age_weights = np.array([3.0 if a < 30 else 2.0 if a < 45 else 1.0 for a in age_range])
    age_weights = age_weights / age_weights.sum()

    citizens = []

    for i in range(size):
        income_level = income_levels[i]

        # Age: 15 to 70, weighted toward younger (BD youth bulge)
        age = int(rng.choice(age_range, p=age_weights))

        # Gender (BD demographics)
        gender = str(rng.choice(GENDERS, p=GENDER_WEIGHTS))

        # City zone
        city_zone = str(rng.choice(CITY_ZONES))

        # Division (population-weighted)
        division = str(rng.choice(DIVISIONS, p=DIVISION_WEIGHTS))

        # Religion (BD demographics)
        religion = str(rng.choice(RELIGIONS, p=RELIGION_WEIGHTS))

        # Remittance family (~15% of BD families receive remittance)
        is_remittance_family = bool(rng.random() < 0.15)

        # Education (correlated with income) — Bangladesh system
        if income_level == "low":
            education = str(rng.choice(
                ["No Formal Education", "Madrasa Education", "SSC"],
                p=[0.40, 0.25, 0.35]
            ))
        elif income_level == "middle":
            education = str(rng.choice(
                ["SSC", "HSC", "Honours/Bachelor's", "Madrasa Education"],
                p=[0.20, 0.35, 0.35, 0.10]
            ))
        else:  # high
            education = str(rng.choice(
                ["Honours/Bachelor's", "Masters", "PhD"],
                p=[0.30, 0.50, 0.20]
            ))

        # Profession based on income level
        profession = str(rng.choice(PROFESSIONS_BY_INCOME[income_level]))

        # Family size: zone-dependent (BD average ~4.2, larger in rural)
        if city_zone == "graam":
            family_size = int(rng.choice(
                [2, 3, 4, 5, 6, 7, 8],
                p=[0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]
            ))
        elif city_zone == "bosti":
            family_size = int(rng.choice(
                [2, 3, 4, 5, 6, 7, 8],
                p=[0.05, 0.10, 0.25, 0.25, 0.20, 0.10, 0.05]
            ))
        else:
            family_size = int(rng.choice(
                [2, 3, 4, 5, 6, 7],
                p=[0.10, 0.20, 0.30, 0.25, 0.10, 0.05]
            ))

        # Political view
        political_view = str(rng.choice(POLITICAL_VIEWS))

        # Personality traits
        risk_tolerance = float(rng.uniform(0.0, 1.0))
        openness_to_change = float(rng.uniform(0.0, 1.0))

        # Base happiness: 0.3 to 0.8
        base_happiness = float(rng.uniform(0.3, 0.8))

        # Base income depends on income level (monthly BDT)
        if income_level == "low":
            base_income = float(rng.uniform(8000, 15000))      # ৳8,000–৳15,000/month
        elif income_level == "middle":
            base_income = float(rng.uniform(20000, 60000))     # ৳20,000–৳60,000/month
        else:  # high
            base_income = float(rng.uniform(80000, 500000))    # ৳80,000–৳5,00,000/month

        citizen = Citizen(
            id=i,
            age=age,
            gender=gender,
            city_zone=city_zone,
            income_level=income_level,
            education=education,
            profession=profession,
            family_size=family_size,
            political_view=political_view,
            risk_tolerance=risk_tolerance,
            openness_to_change=openness_to_change,
            base_happiness=base_happiness,
            base_income=base_income,
            division=division,
            religion=religion,
            is_remittance_family=is_remittance_family
        )

        citizens.append(citizen)

    return citizens
