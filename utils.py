"""
Utility functions for নাগরিক-GENESIS (NAGORIK-GENESIS).
Helper functions for data conversion, formatting, and feature engineering — Bangladesh localized.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from data_models import (
    Citizen, CitizenState,
    CITY_ZONES, INCOME_LEVELS, POLITICAL_VIEWS, DOMAINS,
    DIVISIONS, RELIGIONS
)


def citizen_to_dict(citizen: Citizen) -> Dict[str, Any]:
    """Convert a Citizen object to a dictionary."""
    return {
        "id": citizen.id,
        "age": citizen.age,
        "gender": citizen.gender,
        "city_zone": citizen.city_zone,
        "income_level": citizen.income_level,
        "education": citizen.education,
        "profession": citizen.profession,
        "family_size": citizen.family_size,
        "political_view": citizen.political_view,
        "risk_tolerance": citizen.risk_tolerance,
        "openness_to_change": citizen.openness_to_change,
        "base_happiness": citizen.base_happiness,
        "base_income": citizen.base_income,
        "division": citizen.division,
        "religion": citizen.religion,
        "is_remittance_family": citizen.is_remittance_family
    }


def state_to_dict(state: CitizenState) -> Dict[str, Any]:
    """Convert a CitizenState object to a dictionary."""
    return {
        "citizen_id": state.citizen_id,
        "step": state.step,
        "happiness": state.happiness,
        "policy_support": state.policy_support,
        "income": state.income,
        "diary_entry": state.diary_entry,
        "llm_updated": state.llm_updated
    }


def citizens_to_dataframe(citizens: List[Citizen]) -> pd.DataFrame:
    """Convert list of citizens to a pandas DataFrame."""
    return pd.DataFrame([citizen_to_dict(c) for c in citizens])


def states_to_dataframe(states: List[CitizenState]) -> pd.DataFrame:
    """Convert list of citizen states to a pandas DataFrame."""
    return pd.DataFrame([state_to_dict(s) for s in states])


def format_support(support: float) -> str:
    """
    Format policy support value as percentage.

    Args:
        support: Support value from -1 to 1.

    Returns:
        Formatted string like "+45%" or "-23%".
    """
    percentage = support * 100
    sign = "+" if percentage >= 0 else ""
    return f"{sign}{percentage:.0f}%"


def format_income(income: float) -> str:
    """Format income value with Bangladeshi Taka symbol."""
    return f"৳{income:,.0f}"


def encode_categorical_one_hot(value: str, categories: List[str]) -> List[float]:
    """
    One-hot encode a categorical value.

    Args:
        value: The categorical value to encode.
        categories: List of all possible categories.

    Returns:
        One-hot encoded list.
    """
    encoding = [0.0] * len(categories)
    if value in categories:
        idx = categories.index(value)
        encoding[idx] = 1.0
    return encoding


def build_feature_vector(
    citizen: Citizen,
    prev_state: CitizenState,
    policy_domain: str
) -> np.ndarray:
    """
    Build a feature vector for ML model input — 40 dimensions for Bangladesh.

    Combines citizen static attributes, previous state, and policy domain
    into a single feature vector for the neural network.

    Feature breakdown (40 dimensions):
        Age (1) + Income Level (3) + City Zone (5) + Political View (4) +
        Risk Tolerance (1) + Openness to Change (1) + Family Size (1) +
        Prev Happiness (1) + Prev Support (1) + Prev Income (1) +
        Policy Domain (7) + Division (8) + Religion (5) + Remittance (1)

    Args:
        citizen: Citizen object with demographic and personality data.
        prev_state: Previous CitizenState with happiness, support, income.
        policy_domain: Policy domain.

    Returns:
        NumPy array of 40 features.
    """
    features = []

    # Citizen static features
    features.append(citizen.age / 100.0)  # Normalize age (1)
    features.extend(encode_categorical_one_hot(citizen.income_level, INCOME_LEVELS))  # (3)
    features.extend(encode_categorical_one_hot(citizen.city_zone, CITY_ZONES))  # (5)
    features.extend(encode_categorical_one_hot(citizen.political_view, POLITICAL_VIEWS))  # (4)
    features.append(citizen.risk_tolerance)  # (1)
    features.append(citizen.openness_to_change)  # (1)
    features.append(citizen.family_size / 10.0)  # Normalize family size (1)

    # Previous state features
    features.append(prev_state.happiness)  # (1)
    features.append(prev_state.policy_support)  # (1)
    features.append(np.log1p(prev_state.income) / 15.0)  # Log-scaled income, adjusted for BDT (1)

    # Policy domain features
    features.extend(encode_categorical_one_hot(policy_domain, DOMAINS))  # (7)

    # Bangladesh-specific features
    features.extend(encode_categorical_one_hot(citizen.division, DIVISIONS))  # (8)
    features.extend(encode_categorical_one_hot(citizen.religion, RELIGIONS))  # (5)
    features.append(1.0 if citizen.is_remittance_family else 0.0)  # (1)

    return np.array(features, dtype=np.float32)


def get_feature_dimension() -> int:
    """
    Get the total dimension of feature vectors.

    Returns:
        Integer dimension of feature space (40 for Bangladesh version).
    """
    # Age (1) + income_level (3) + city_zone (5) + political_view (4) +
    # risk_tolerance (1) + openness_to_change (1) + family_size (1) +
    # prev_happiness (1) + prev_support (1) + prev_income (1) +
    # policy_domain (7) + division (8) + religion (5) + remittance (1)
    return 1 + 3 + 5 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 7 + 8 + 5 + 1  # = 40


def compute_deltas(
    prev_state: CitizenState,
    new_happiness: float,
    new_support: float,
    new_income: float
) -> np.ndarray:
    """
    Compute deltas (changes) in citizen state.

    Args:
        prev_state: Previous state.
        new_happiness: New happiness value.
        new_support: New support value.
        new_income: New income value.

    Returns:
        NumPy array [delta_happiness, delta_support, delta_income].
    """
    delta_h = new_happiness - prev_state.happiness
    delta_s = new_support - prev_state.policy_support
    delta_i = new_income - prev_state.income

    return np.array([delta_h, delta_s, delta_i], dtype=np.float32)


def apply_deltas(
    prev_state: CitizenState,
    deltas: np.ndarray
) -> tuple[float, float, float]:
    """
    Apply deltas to previous state and clamp values.

    Args:
        prev_state: Previous CitizenState.
        deltas: NumPy array [delta_happiness, delta_support, delta_income].

    Returns:
        Tuple of (new_happiness, new_support, new_income) with clamped values.
    """
    new_happiness = np.clip(prev_state.happiness + deltas[0], 0.0, 1.0)
    new_support = np.clip(prev_state.policy_support + deltas[1], -1.0, 1.0)
    new_income = max(0.0, prev_state.income + deltas[2])

    return float(new_happiness), float(new_support), float(new_income)


def get_policy_presets() -> List[Dict[str, str]]:
    """
    Get predefined Bangladesh policy presets for quick demos.

    Returns:
        List of policy dicts with title, description, and domain.
    """
    return [
        {
            "title": "জ্বালানি ভর্তুকি প্রত্যাহার (Fuel Subsidy Removal)",
            "description": (
                "সরকার ডিজেল ও কেরোসিনের ভর্তুকি প্রত্যাহার করছে। জ্বালানির দাম ৩০% বৃদ্ধি পাবে, "
                "যা পরিবহন ও খাদ্যমূল্যে প্রভাব ফেলবে। "
                "Government removes diesel and kerosene subsidies. Fuel prices rise 30%, "
                "impacting transport and food costs across Bangladesh."
            ),
            "domain": "Economy"
        },
        {
            "title": "গার্মেন্ট শ্রমিকদের ন্যূনতম মজুরি বৃদ্ধি (RMG Minimum Wage Hike)",
            "description": (
                "গার্মেন্ট শ্রমিকদের ন্যূনতম মাসিক মজুরি ৳12,500 থেকে ৳18,000-এ উন্নীত করা হচ্ছে। "
                "Minimum monthly wage for RMG workers raised from ৳12,500 to ৳18,000."
            ),
            "domain": "Economy"
        },
        {
            "title": "ডিজিটাল বাংলাদেশ বৃত্তি (Digital Bangladesh Scholarship)",
            "description": (
                "গ্রামীণ এলাকার মেধাবী শিক্ষার্থীদের জন্য বার্ষিক ৳50,000 বৃত্তি এবং ল্যাপটপ প্রদান। "
                "Annual ৳50,000 scholarship + laptop for meritorious rural students to pursue STEM education."
            ),
            "domain": "Education"
        },
        {
            "title": "বন্যা-পরবর্তী সহায়তা প্যাকেজ (Post-Flood Relief Package)",
            "description": (
                "বন্যা-আক্রান্ত পরিবারগুলোকে ৳25,000 নগদ সহায়তা, বিনামূল্যে খাদ্য ও অস্থায়ী আশ্রয় প্রদান। "
                "৳25,000 cash aid, free food rations, and temporary shelters for flood-affected families "
                "in northern Bangladesh."
            ),
            "domain": "Climate & Disaster"
        },
        {
            "title": "মেট্রোরেল সম্প্রসারণ (Metro Rail Expansion)",
            "description": (
                "ঢাকা মেট্রোরেল সম্প্রসারণ: উত্তরা থেকে কমলাপুর পর্যন্ত। "
                "দৈনিক ৫ লক্ষ যাত্রী ধারণক্ষমতা, যা যানজট কমাবে। "
                "Dhaka Metro Rail expansion from Uttara to Kamalapur. "
                "Capacity for 500K daily riders, reducing traffic congestion."
            ),
            "domain": "Infrastructure"
        },
        {
            "title": "বস্তি উচ্ছেদ ও পুনর্বাসন (Slum Eviction & Resettlement)",
            "description": (
                "ঢাকার করাইল বস্তি উচ্ছেদ করে বাসিন্দাদের শহরের বাইরে পুনর্বাসনের পরিকল্পনা। "
                "Plan to evict Korail slum in Dhaka and resettle residents to the outskirts with basic housing."
            ),
            "domain": "Social"
        },
        {
            "title": "সকলের জন্য স্বাস্থ্যসেবা কার্ড (Universal Health Card)",
            "description": (
                "প্রতিটি পরিবারকে একটি স্বাস্থ্যসেবা কার্ড দেওয়া হবে যাতে বছরে ৳50,000 পর্যন্ত "
                "চিকিৎসা বিনামূল্যে পাওয়া যাবে। "
                "A health card for every family covering up to ৳50,000/year in medical expenses "
                "at government hospitals."
            ),
            "domain": "Healthcare"
        },
        {
            "title": "ফ্রিল্যান্সিং ট্যাক্স প্রণোদনা (Freelancing Tax Incentive)",
            "description": (
                "IT ফ্রিল্যান্সারদের আয়ের উপর ৫ বছর কর অব্যাহতি এবং সরকারি ডিজিটাল হাব স্থাপন। "
                "5-year tax exemption on freelancing income + government digital hubs in every district."
            ),
            "domain": "Digital & Technology"
        }
    ]
