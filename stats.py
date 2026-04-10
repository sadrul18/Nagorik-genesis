"""
Statistics computation module for নাগরিক-GENESIS.
Aggregate and analyze simulation step data.
Adds by_division and by_religion groupings for Bangladesh context.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from data_models import Citizen, CitizenState, StepStats


def compute_step_stats(
    citizens: List[Citizen],
    states_for_step: List[CitizenState]
) -> StepStats:
    """
    Compute aggregated statistics for a single simulation step.

    Args:
        citizens: List of all citizens in the population.
        states_for_step: List of CitizenState objects for this step.

    Returns:
        StepStats object with aggregated metrics including by_division, by_religion.
    """
    if not states_for_step:
        return StepStats(
            step=0,
            avg_happiness=0.0,
            avg_support=0.0,
            avg_income=0.0,
            by_income=[],
            by_zone=[],
            by_division=[],
            by_religion=[],
            inequality_gap_happiness=0.0,
            inequality_gap_support=0.0
        )

    step = states_for_step[0].step
    citizen_map = {c.id: c for c in citizens}

    # Overall averages
    avg_happiness = np.mean([s.happiness for s in states_for_step])
    avg_support = np.mean([s.policy_support for s in states_for_step])
    avg_income = np.mean([s.income for s in states_for_step])

    # Group by income level
    by_income_dict: Dict[str, Dict[str, list]] = {}
    for state in states_for_step:
        citizen = citizen_map.get(state.citizen_id)
        if citizen:
            level = citizen.income_level
            if level not in by_income_dict:
                by_income_dict[level] = {"happiness": [], "support": []}
            by_income_dict[level]["happiness"].append(state.happiness)
            by_income_dict[level]["support"].append(state.policy_support)

    by_income = []
    for level in ["low", "middle", "high"]:
        if level in by_income_dict:
            data = by_income_dict[level]
            by_income.append({
                "income_level": level,
                "avg_happiness": float(np.mean(data["happiness"])),
                "avg_support": float(np.mean(data["support"]))
            })

    # Group by city zone
    by_zone_dict: Dict[str, Dict[str, list]] = {}
    for state in states_for_step:
        citizen = citizen_map.get(state.citizen_id)
        if citizen:
            zone = citizen.city_zone
            if zone not in by_zone_dict:
                by_zone_dict[zone] = {"happiness": [], "support": []}
            by_zone_dict[zone]["happiness"].append(state.happiness)
            by_zone_dict[zone]["support"].append(state.policy_support)

    by_zone = []
    for zone, data in by_zone_dict.items():
        by_zone.append({
            "city_zone": zone,
            "avg_happiness": float(np.mean(data["happiness"])),
            "avg_support": float(np.mean(data["support"]))
        })

    # Group by division (Bangladesh-specific)
    by_division_dict: Dict[str, Dict[str, list]] = {}
    for state in states_for_step:
        citizen = citizen_map.get(state.citizen_id)
        if citizen:
            division = citizen.division
            if division not in by_division_dict:
                by_division_dict[division] = {"happiness": [], "support": []}
            by_division_dict[division]["happiness"].append(state.happiness)
            by_division_dict[division]["support"].append(state.policy_support)

    by_division = []
    for division, data in by_division_dict.items():
        by_division.append({
            "division": division,
            "avg_happiness": float(np.mean(data["happiness"])),
            "avg_support": float(np.mean(data["support"]))
        })

    # Group by religion (Bangladesh-specific)
    by_religion_dict: Dict[str, Dict[str, list]] = {}
    for state in states_for_step:
        citizen = citizen_map.get(state.citizen_id)
        if citizen:
            religion = citizen.religion
            if religion not in by_religion_dict:
                by_religion_dict[religion] = {"happiness": [], "support": []}
            by_religion_dict[religion]["happiness"].append(state.happiness)
            by_religion_dict[religion]["support"].append(state.policy_support)

    by_religion = []
    for religion, data in by_religion_dict.items():
        by_religion.append({
            "religion": religion,
            "avg_happiness": float(np.mean(data["happiness"])),
            "avg_support": float(np.mean(data["support"]))
        })

    # Compute inequality gaps
    high_stats = next((x for x in by_income if x["income_level"] == "high"), None)
    low_stats = next((x for x in by_income if x["income_level"] == "low"), None)

    inequality_gap_happiness = 0.0
    inequality_gap_support = 0.0

    if high_stats and low_stats:
        inequality_gap_happiness = high_stats["avg_happiness"] - low_stats["avg_happiness"]
        inequality_gap_support = high_stats["avg_support"] - low_stats["avg_support"]

    return StepStats(
        step=step,
        avg_happiness=float(avg_happiness),
        avg_support=float(avg_support),
        avg_income=float(avg_income),
        by_income=by_income,
        by_zone=by_zone,
        by_division=by_division,
        by_religion=by_religion,
        inequality_gap_happiness=float(inequality_gap_happiness),
        inequality_gap_support=float(inequality_gap_support)
    )


def build_stats_dataframe(
    all_states: List[CitizenState],
    citizens: List[Citizen]
) -> pd.DataFrame:
    """
    Build a pandas DataFrame with all states for analysis.

    Args:
        all_states: List of all CitizenState objects across all steps.
        citizens: List of all citizens.

    Returns:
        DataFrame with Bangladesh-specific columns including division, religion, is_remittance_family.
    """
    citizen_map = {c.id: c for c in citizens}

    rows = []
    for state in all_states:
        citizen = citizen_map.get(state.citizen_id)
        if citizen:
            rows.append({
                "step": state.step,
                "citizen_id": state.citizen_id,
                "happiness": state.happiness,
                "policy_support": state.policy_support,
                "income": state.income,
                "income_level": citizen.income_level,
                "city_zone": citizen.city_zone,
                "age": citizen.age,
                "profession": citizen.profession,
                "political_view": citizen.political_view,
                "division": citizen.division,
                "religion": citizen.religion,
                "is_remittance_family": citizen.is_remittance_family,
                "llm_updated": state.llm_updated,
                "has_diary": state.diary_entry is not None
            })

    return pd.DataFrame(rows)


def compute_time_series_stats(
    all_states: List[CitizenState],
    citizens: List[Citizen],
    max_step: int
) -> List[StepStats]:
    """
    Compute statistics for all steps in the simulation.

    Args:
        all_states: List of all CitizenState objects.
        citizens: List of all citizens.
        max_step: Maximum step number.

    Returns:
        List of StepStats objects, one per step.
    """
    states_by_step: Dict[int, List[CitizenState]] = {}
    for state in all_states:
        if state.step not in states_by_step:
            states_by_step[state.step] = []
        states_by_step[state.step].append(state)

    time_series = []
    for step in range(max_step + 1):
        states = states_by_step.get(step, [])
        stats = compute_step_stats(citizens, states)
        time_series.append(stats)

    return time_series
