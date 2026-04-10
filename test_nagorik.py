"""
Test suite for NAGORIK-GENESIS.
Tests population generation, feature vectors, rule-based engine,
stats computation, data models, and end-to-end simulation (rule-based only).

Run: python test_nagorik.py
"""
import sys
import os
import numpy as np
import traceback
from pathlib import Path

# Ensure we're in the right directory
sys.path.insert(0, str(Path(__file__).parent))

from data_models import (
    Citizen, CitizenState, PolicyInput, SimulationConfig, StepStats, ExpertSummary,
    DOMAINS, CITY_ZONES, INCOME_LEVELS, POLITICAL_VIEWS,
    GENDERS, EDUCATION_LEVELS, DIVISIONS, RELIGIONS
)
from population import (
    generate_population, PROFESSIONS_BY_INCOME,
    DIVISION_WEIGHTS, RELIGION_WEIGHTS, GENDER_WEIGHTS
)
from utils import (
    citizen_to_dict, state_to_dict, citizens_to_dataframe, states_to_dataframe,
    format_income, format_support, encode_categorical_one_hot,
    build_feature_vector, get_feature_dimension, compute_deltas, apply_deltas,
    get_policy_presets
)
from simulation import rule_based_update
from stats import compute_step_stats, build_stats_dataframe, compute_time_series_stats
from ml_data import MLDataset
from nn_model import CitizenReactionModel


PASSED = 0
FAILED = 0
ERRORS = []


def run_test(name, func):
    """Run a single test and track results."""
    global PASSED, FAILED, ERRORS
    try:
        func()
        PASSED += 1
        print(f"  ✅ {name}")
    except Exception as e:
        FAILED += 1
        error_msg = f"  ❌ {name}: {e}"
        ERRORS.append(error_msg)
        print(error_msg)
        traceback.print_exc()


# ===================================================================
# 1. DATA MODELS TESTS
# ===================================================================
def test_constants_count():
    """Verify all BD constants have correct counts."""
    assert len(DOMAINS) == 7, f"Expected 7 domains, got {len(DOMAINS)}"
    assert len(CITY_ZONES) == 5, f"Expected 5 zones, got {len(CITY_ZONES)}"
    assert len(INCOME_LEVELS) == 3, f"Expected 3 income levels, got {len(INCOME_LEVELS)}"
    assert len(POLITICAL_VIEWS) == 4, f"Expected 4 political views, got {len(POLITICAL_VIEWS)}"
    assert len(GENDERS) == 3, f"Expected 3 genders (incl Hijra), got {len(GENDERS)}"
    assert len(EDUCATION_LEVELS) == 7, f"Expected 7 education levels, got {len(EDUCATION_LEVELS)}"
    assert len(DIVISIONS) == 8, f"Expected 8 divisions, got {len(DIVISIONS)}"
    assert len(RELIGIONS) == 5, f"Expected 5 religions, got {len(RELIGIONS)}"


def test_constants_values():
    """Verify specific BD constants are present."""
    assert "Hijra" in GENDERS
    assert "bosti" in CITY_ZONES
    assert "civil_society_aligned" in POLITICAL_VIEWS
    assert "Madrasa Education" in EDUCATION_LEVELS
    assert "Climate & Disaster" in DOMAINS
    assert "Digital & Technology" in DOMAINS
    assert "Mymensingh" in DIVISIONS
    assert "Buddhist" in RELIGIONS


def test_citizen_dataclass_fields():
    """Verify Citizen has all BD-specific fields."""
    c = Citizen(
        id=0, age=25, gender="Male", city_zone="graam", income_level="low",
        education="SSC", profession="Farmer", family_size=5,
        political_view="neutral", risk_tolerance=0.5, openness_to_change=0.5,
        base_happiness=0.5, base_income=12000.0,
        division="Rangpur", religion="Muslim", is_remittance_family=True
    )
    assert hasattr(c, "division")
    assert hasattr(c, "religion")
    assert hasattr(c, "is_remittance_family")
    assert c.division == "Rangpur"
    assert c.religion == "Muslim"
    assert c.is_remittance_family is True


def test_step_stats_fields():
    """Verify StepStats has by_division and by_religion."""
    s = StepStats(step=0, avg_happiness=0.5, avg_support=0.0, avg_income=20000.0)
    assert hasattr(s, "by_division")
    assert hasattr(s, "by_religion")
    assert isinstance(s.by_division, list)
    assert isinstance(s.by_religion, list)


def test_expert_summary_fields():
    """Verify ExpertSummary has 4 BD expert views."""
    es = ExpertSummary(
        step=1, economist_view="e", activist_view="a",
        garment_industry_view="g", rural_leader_view="r"
    )
    assert hasattr(es, "garment_industry_view")
    assert hasattr(es, "rural_leader_view")


# ===================================================================
# 2. POPULATION TESTS
# ===================================================================
def test_population_size():
    """Generate population and verify correct size."""
    pop = generate_population(100, seed=42)
    assert len(pop) == 100, f"Expected 100 citizens, got {len(pop)}"


def test_population_income_distribution():
    """Verify income distribution matches 55/35/10 BD defaults."""
    pop = generate_population(1000, seed=42)
    counts = {"low": 0, "middle": 0, "high": 0}
    for c in pop:
        counts[c.income_level] += 1

    assert 540 <= counts["low"] <= 560, f"Low income count {counts['low']} not near 550"
    assert 340 <= counts["middle"] <= 360, f"Middle income count {counts['middle']} not near 350"
    assert 90 <= counts["high"] <= 110, f"High income count {counts['high']} not near 100"


def test_population_custom_income_shares():
    """Verify custom income shares work."""
    pop = generate_population(1000, seed=42, low_share=0.70, middle_share=0.20, high_share=0.10)
    counts = {"low": 0, "middle": 0, "high": 0}
    for c in pop:
        counts[c.income_level] += 1

    assert 690 <= counts["low"] <= 710


def test_population_age_range():
    """Verify ages are 15-70 (BD working age)."""
    pop = generate_population(500, seed=42)
    ages = [c.age for c in pop]
    assert min(ages) >= 15, f"Min age {min(ages)} < 15"
    assert max(ages) <= 70, f"Max age {max(ages)} > 70"


def test_population_youth_bulge():
    """Verify youth-weighted age distribution (BD median ~27)."""
    pop = generate_population(5000, seed=42)
    ages = [c.age for c in pop]
    under_30 = sum(1 for a in ages if a < 30)
    pct_under_30 = under_30 / len(ages)
    # With weights 3.0 for <30, ~45-55% should be under 30
    assert 0.40 <= pct_under_30 <= 0.60, f"Under-30 pct {pct_under_30:.2%} outside expected range"


def test_population_gender_distribution():
    """Verify gender distribution includes Hijra."""
    pop = generate_population(5000, seed=42)
    genders = [c.gender for c in pop]
    assert "Hijra" in genders, "Hijra gender missing from population"
    hijra_pct = genders.count("Hijra") / len(genders)
    assert 0.001 <= hijra_pct <= 0.010, f"Hijra pct {hijra_pct:.4f} outside expected range"


def test_population_divisions():
    """Verify all 8 divisions are represented."""
    pop = generate_population(5000, seed=42)
    divisions_found = set(c.division for c in pop)
    for div in DIVISIONS:
        assert div in divisions_found, f"Division {div} not found in population"


def test_population_division_weights():
    """Verify Dhaka has highest share (~30%)."""
    pop = generate_population(5000, seed=42)
    dhaka_count = sum(1 for c in pop if c.division == "Dhaka")
    dhaka_pct = dhaka_count / len(pop)
    assert 0.25 <= dhaka_pct <= 0.35, f"Dhaka pct {dhaka_pct:.2%} outside expected 25-35%"


def test_population_religions():
    """Verify all religions present and Muslim majority."""
    pop = generate_population(5000, seed=42)
    religions_found = set(c.religion for c in pop)
    for rel in RELIGIONS:
        if rel != "Other":  # Other may have very low probability
            assert rel in religions_found, f"Religion {rel} not found"
    muslim_pct = sum(1 for c in pop if c.religion == "Muslim") / len(pop)
    assert 0.88 <= muslim_pct <= 0.94, f"Muslim pct {muslim_pct:.2%} outside expected 88-94%"


def test_population_remittance():
    """Verify ~15% remittance families."""
    pop = generate_population(5000, seed=42)
    remit_pct = sum(1 for c in pop if c.is_remittance_family) / len(pop)
    assert 0.12 <= remit_pct <= 0.18, f"Remittance pct {remit_pct:.2%} outside expected 12-18%"


def test_population_income_ranges_bdt():
    """Verify income ranges are in BDT (not USD)."""
    pop = generate_population(1000, seed=42)
    for c in pop:
        if c.income_level == "low":
            assert 8000 <= c.base_income <= 15000, f"Low income {c.base_income} outside BDT 8K-15K"
        elif c.income_level == "middle":
            assert 20000 <= c.base_income <= 60000, f"Middle income {c.base_income} outside BDT 20K-60K"
        elif c.income_level == "high":
            assert 80000 <= c.base_income <= 500000, f"High income {c.base_income} outside BDT 80K-500K"


def test_population_education_income_correlation():
    """Verify education is correlated with income level."""
    pop = generate_population(5000, seed=42)
    low_edus = [c.education for c in pop if c.income_level == "low"]
    high_edus = [c.education for c in pop if c.income_level == "high"]

    # Low income should have no PhD or Masters
    assert "PhD" not in low_edus, "PhD found in low income"
    assert "Masters" not in low_edus, "Masters found in low income"
    # High income should have no 'No Formal Education'
    assert "No Formal Education" not in high_edus, "No Formal Education found in high income"


def test_population_professions():
    """Verify professions match income levels."""
    pop = generate_population(1000, seed=42)
    for c in pop:
        assert c.profession in PROFESSIONS_BY_INCOME[c.income_level], \
            f"Profession '{c.profession}' invalid for {c.income_level} income"


def test_population_family_size_zone_dependent():
    """Verify family size varies by zone."""
    pop = generate_population(10000, seed=42)
    graam_families = [c.family_size for c in pop if c.city_zone == "graam"]
    shohor_families = [c.family_size for c in pop if c.city_zone == "shohor_kendro"]

    if graam_families and shohor_families:
        avg_graam = np.mean(graam_families)
        avg_shohor = np.mean(shohor_families)
        # Rural families should be larger on average
        assert avg_graam > avg_shohor, f"Rural avg ({avg_graam:.1f}) not > urban avg ({avg_shohor:.1f})"


def test_population_reproducibility():
    """Verify same seed produces same population."""
    pop1 = generate_population(100, seed=123)
    pop2 = generate_population(100, seed=123)
    for c1, c2 in zip(pop1, pop2):
        assert c1.age == c2.age
        assert c1.gender == c2.gender
        assert c1.income_level == c2.income_level
        assert c1.division == c2.division
        assert c1.base_income == c2.base_income


# ===================================================================
# 3. UTILS / FEATURE VECTOR TESTS
# ===================================================================
def test_format_income_bdt():
    """Verify BDT prefix and comma formatting."""
    assert format_income(12000) == "BDT 12,000"
    assert format_income(150000) == "BDT 1,50,000" or format_income(150000) == "BDT 150,000"
    assert format_income(0) == "BDT 0"


def test_format_support():
    """Verify support percentage formatting."""
    assert format_support(0.5) == "+50%"
    assert format_support(-0.3) == "-30%"
    assert format_support(0.0) == "+0%"


def test_one_hot_encoding():
    """Verify one-hot encoding correctness."""
    result = encode_categorical_one_hot("middle", INCOME_LEVELS)
    assert result == [0.0, 1.0, 0.0], f"Got {result}"

    result = encode_categorical_one_hot("bosti", CITY_ZONES)
    assert result == [0.0, 0.0, 0.0, 0.0, 1.0], f"Got {result}"

    result = encode_categorical_one_hot("Dhaka", DIVISIONS)
    expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert result == expected, f"Got {result}"


def test_one_hot_unknown():
    """Verify unknown values produce all-zero encoding."""
    result = encode_categorical_one_hot("nonexistent", INCOME_LEVELS)
    assert result == [0.0, 0.0, 0.0]


def test_feature_dimension():
    """Verify get_feature_dimension returns 40."""
    assert get_feature_dimension() == 40


def test_build_feature_vector_shape():
    """Verify feature vector is 40-dimensional."""
    citizen = Citizen(
        id=0, age=30, gender="Female", city_zone="bosti", income_level="low",
        education="SSC", profession="Garment Worker", family_size=5,
        political_view="opposition_supporter", risk_tolerance=0.7,
        openness_to_change=0.3, base_happiness=0.4, base_income=12000.0,
        division="Dhaka", religion="Muslim", is_remittance_family=False
    )
    prev_state = CitizenState(citizen_id=0, step=0, happiness=0.4, policy_support=0.0, income=12000.0)
    X = build_feature_vector(citizen, prev_state, "Economy")

    assert X.shape == (40,), f"Expected (40,), got {X.shape}"
    assert X.dtype == np.float32


def test_build_feature_vector_values():
    """Verify specific feature values are correct."""
    citizen = Citizen(
        id=0, age=50, gender="Male", city_zone="shohor_kendro", income_level="high",
        education="Masters", profession="Corporate Executive", family_size=3,
        political_view="government_supporter", risk_tolerance=0.8,
        openness_to_change=0.6, base_happiness=0.7, base_income=200000.0,
        division="Chittagong", religion="Hindu", is_remittance_family=True
    )
    prev_state = CitizenState(citizen_id=0, step=1, happiness=0.65, policy_support=0.3, income=210000.0)
    X = build_feature_vector(citizen, prev_state, "Healthcare")

    # Age: 50/100 = 0.5
    assert abs(X[0] - 0.5) < 0.01, f"Age feature {X[0]} != 0.5"
    # Income level high = [0,0,1]
    assert X[1:4].tolist() == [0.0, 0.0, 1.0], f"Income encoding {X[1:4].tolist()}"
    # City zone shohor_kendro = [1,0,0,0,0]
    assert X[4:9].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    # Remittance = 1.0 (last feature)
    assert X[39] == 1.0, f"Remittance feature {X[39]}"


def test_build_feature_vector_all_domains():
    """Verify feature vector works for all 7 domains."""
    citizen = Citizen(
        id=0, age=25, gender="Female", city_zone="graam", income_level="low",
        education="SSC", profession="Farmer", family_size=4,
        political_view="neutral", risk_tolerance=0.5, openness_to_change=0.5,
        base_happiness=0.5, base_income=10000.0,
        division="Rajshahi", religion="Muslim", is_remittance_family=False
    )
    prev_state = CitizenState(citizen_id=0, step=0, happiness=0.5, policy_support=0.0, income=10000.0)

    for domain in DOMAINS:
        X = build_feature_vector(citizen, prev_state, domain)
        assert X.shape == (40,), f"Domain {domain}: shape {X.shape}"
        # Domain encoding starts at index 19 (after age(1)+income(3)+zone(5)+political(4)+risk(1)+openness(1)+family(1)+happiness(1)+support(1)+income_log(1)=19)
        domain_start = 1 + 3 + 5 + 4 + 1 + 1 + 1 + 1 + 1 + 1  # = 19
        domain_encoding = X[domain_start:domain_start + 7]
        assert sum(domain_encoding) == 1.0, f"Domain {domain}: encoding sum {sum(domain_encoding)}"


def test_compute_deltas():
    """Verify delta computation."""
    prev = CitizenState(citizen_id=0, step=0, happiness=0.5, policy_support=0.0, income=15000.0)
    deltas = compute_deltas(prev, 0.7, 0.3, 16000.0)
    assert abs(deltas[0] - 0.2) < 0.01
    assert abs(deltas[1] - 0.3) < 0.01
    assert abs(deltas[2] - 1000.0) < 0.01


def test_apply_deltas_clamping():
    """Verify deltas are clamped correctly."""
    prev = CitizenState(citizen_id=0, step=0, happiness=0.9, policy_support=0.8, income=5000.0)
    deltas = np.array([0.3, 0.5, -10000.0])  # Would exceed bounds
    h, s, i = apply_deltas(prev, deltas)

    assert h == 1.0, f"Happiness not clamped at 1.0: {h}"
    assert s == 1.0, f"Support not clamped at 1.0: {s}"
    assert i == 0.0, f"Income not clamped at 0.0: {i}"


def test_policy_presets_count():
    """Verify exactly 8 BD policy presets."""
    presets = get_policy_presets()
    assert len(presets) == 8, f"Expected 8 presets, got {len(presets)}"


def test_policy_presets_domains():
    """Verify presets cover multiple domains."""
    presets = get_policy_presets()
    domains_used = set(p["domain"] for p in presets)
    assert len(domains_used) >= 5, f"Presets only cover {len(domains_used)} domains"
    for p in presets:
        assert p["domain"] in DOMAINS, f"Preset domain '{p['domain']}' not in DOMAINS"


def test_policy_presets_content():
    """Verify presets contain English policy descriptions."""
    presets = get_policy_presets()
    for p in presets:
        assert len(p["description"]) > 20, \
            f"Preset '{p['title']}' has insufficient description"
        assert len(p["title"]) > 5, \
            f"Preset '{p['title']}' has insufficient title"


def test_citizen_to_dict():
    """Verify citizen_to_dict includes BD fields."""
    c = Citizen(
        id=0, age=30, gender="Male", city_zone="graam", income_level="low",
        education="SSC", profession="Farmer", family_size=5,
        political_view="neutral", risk_tolerance=0.5, openness_to_change=0.5,
        base_happiness=0.5, base_income=12000.0,
        division="Rangpur", religion="Muslim", is_remittance_family=True
    )
    d = citizen_to_dict(c)
    assert "division" in d
    assert "religion" in d
    assert "is_remittance_family" in d
    assert d["division"] == "Rangpur"


# ===================================================================
# 4. RULE-BASED ENGINE TESTS
# ===================================================================
def test_rule_based_all_domains():
    """Verify rule_based_update works for all 7 domains."""
    rng = np.random.default_rng(42)
    citizen = Citizen(
        id=0, age=30, gender="Female", city_zone="bosti", income_level="low",
        education="SSC", profession="Garment Worker", family_size=5,
        political_view="neutral", risk_tolerance=0.5, openness_to_change=0.5,
        base_happiness=0.5, base_income=12000.0,
        division="Dhaka", religion="Muslim", is_remittance_family=False
    )
    prev_state = CitizenState(citizen_id=0, step=0, happiness=0.5, policy_support=0.0, income=12000.0)

    for domain in DOMAINS:
        h, s, i = rule_based_update(citizen, prev_state, domain, rng)
        assert 0.0 <= h <= 1.0, f"Domain {domain}: happiness {h} out of range"
        assert -1.0 <= s <= 1.0, f"Domain {domain}: support {s} out of range"
        assert i >= 0.0, f"Domain {domain}: income {i} < 0"


def test_rule_based_climate_low_income_devastation():
    """Verify Climate & Disaster hits low income hardest (negative skew)."""
    rng = np.random.default_rng(42)
    citizen_low = Citizen(
        id=0, age=35, gender="Male", city_zone="graam", income_level="low",
        education="No Formal Education", profession="Farmer", family_size=6,
        political_view="neutral", risk_tolerance=0.3, openness_to_change=0.3,
        base_happiness=0.5, base_income=10000.0,
        division="Rangpur", religion="Muslim", is_remittance_family=False
    )
    citizen_high = Citizen(
        id=1, age=45, gender="Male", city_zone="shohor_kendro", income_level="high",
        education="Masters", profession="Corporate Executive", family_size=3,
        political_view="government_supporter", risk_tolerance=0.7, openness_to_change=0.6,
        base_happiness=0.7, base_income=200000.0,
        division="Dhaka", religion="Muslim", is_remittance_family=False
    )
    prev_low = CitizenState(citizen_id=0, step=0, happiness=0.5, policy_support=0.0, income=10000.0)
    prev_high = CitizenState(citizen_id=1, step=0, happiness=0.7, policy_support=0.0, income=200000.0)

    # Run multiple samples to check expected value
    low_deltas = []
    high_deltas = []
    for _ in range(1000):
        _, _, new_i_low = rule_based_update(citizen_low, prev_low, "Climate & Disaster", rng)
        _, _, new_i_high = rule_based_update(citizen_high, prev_high, "Climate & Disaster", rng)
        low_deltas.append(new_i_low - 10000)
        high_deltas.append(new_i_high - 200000)

    avg_low_delta = np.mean(low_deltas)
    avg_high_delta = np.mean(high_deltas)
    # Low income average delta should be much more negative
    assert avg_low_delta < avg_high_delta, \
        f"Climate impact: low avg delta {avg_low_delta:.0f} not < high avg delta {avg_high_delta:.0f}"


def test_rule_based_income_never_negative():
    """Verify income never goes below 0."""
    rng = np.random.default_rng(42)
    citizen = Citizen(
        id=0, age=30, gender="Male", city_zone="bosti", income_level="low",
        education="No Formal Education", profession="Day Laborer", family_size=6,
        political_view="neutral", risk_tolerance=0.5, openness_to_change=0.5,
        base_happiness=0.3, base_income=8000.0,
        division="Barisal", religion="Muslim", is_remittance_family=False
    )
    prev_state = CitizenState(citizen_id=0, step=0, happiness=0.3, policy_support=-0.5, income=1000.0)

    for _ in range(1000):
        h, s, i = rule_based_update(citizen, prev_state, "Climate & Disaster", rng)
        assert i >= 0.0, f"Income went negative: {i}"


def test_rule_based_social_redistributive():
    """Verify Social domain: positive for low income, negative for high income."""
    rng = np.random.default_rng(42)
    low_citizen = Citizen(
        id=0, age=30, gender="Female", city_zone="bosti", income_level="low",
        education="No Formal Education", profession="Domestic Worker", family_size=5,
        political_view="neutral", risk_tolerance=0.5, openness_to_change=0.5,
        base_happiness=0.5, base_income=10000.0,
        division="Dhaka", religion="Muslim", is_remittance_family=False
    )
    high_citizen = Citizen(
        id=1, age=50, gender="Male", city_zone="shohor_kendro", income_level="high",
        education="Masters", profession="Real Estate Developer", family_size=3,
        political_view="government_supporter", risk_tolerance=0.7, openness_to_change=0.4,
        base_happiness=0.7, base_income=300000.0,
        division="Dhaka", religion="Muslim", is_remittance_family=False
    )
    prev_low = CitizenState(citizen_id=0, step=0, happiness=0.5, policy_support=0.0, income=10000.0)
    prev_high = CitizenState(citizen_id=1, step=0, happiness=0.7, policy_support=0.0, income=300000.0)

    low_deltas, high_deltas = [], []
    for _ in range(1000):
        _, _, new_i_low = rule_based_update(low_citizen, prev_low, "Social", rng)
        _, _, new_i_high = rule_based_update(high_citizen, prev_high, "Social", rng)
        low_deltas.append(new_i_low - 10000)
        high_deltas.append(new_i_high - 300000)

    # Social: Low has U(0, 3000), High has U(-10000, 0)
    assert np.mean(low_deltas) > 0, f"Social low delta avg {np.mean(low_deltas):.0f} not positive"
    assert np.mean(high_deltas) < 0, f"Social high delta avg {np.mean(high_deltas):.0f} not negative"


# ===================================================================
# 5. STATS TESTS
# ===================================================================
def test_compute_step_stats():
    """Verify step stats computation with division and religion."""
    citizens = generate_population(100, seed=42)
    states = [
        CitizenState(
            citizen_id=c.id, step=1,
            happiness=np.random.uniform(0.3, 0.8),
            policy_support=np.random.uniform(-0.5, 0.5),
            income=c.base_income + np.random.uniform(-2000, 2000)
        )
        for c in citizens
    ]

    stats = compute_step_stats(citizens, states)
    assert stats.step == 1
    assert 0.0 <= stats.avg_happiness <= 1.0
    assert -1.0 <= stats.avg_support <= 1.0
    assert stats.avg_income > 0

    # Verify BD-specific groupings
    assert len(stats.by_income) > 0, "by_income is empty"
    assert len(stats.by_zone) > 0, "by_zone is empty"
    assert len(stats.by_division) > 0, "by_division is empty"
    assert len(stats.by_religion) > 0, "by_religion is empty"

    # Check division names
    div_names = [d["division"] for d in stats.by_division]
    assert any(d in DIVISIONS for d in div_names)


def test_build_stats_dataframe():
    """Verify stats DataFrame includes BD columns."""
    citizens = generate_population(50, seed=42)
    states = [
        CitizenState(citizen_id=c.id, step=0, happiness=0.5, policy_support=0.0, income=c.base_income)
        for c in citizens
    ]
    df = build_stats_dataframe(states, citizens)
    assert "division" in df.columns
    assert "religion" in df.columns
    assert "is_remittance_family" in df.columns


def test_compute_time_series_stats():
    """Verify time series stats computation across multiple steps."""
    citizens = generate_population(50, seed=42)
    all_states = []
    for step in range(3):
        for c in citizens:
            all_states.append(CitizenState(
                citizen_id=c.id, step=step,
                happiness=0.5 + step * 0.05,
                policy_support=0.0 + step * 0.1,
                income=c.base_income + step * 500
            ))
    step_stats_list = compute_time_series_stats(all_states, citizens, 2)
    assert len(step_stats_list) == 3, f"Expected 3 step stats, got {len(step_stats_list)}"
    # Happiness should increase over steps
    assert step_stats_list[2].avg_happiness > step_stats_list[0].avg_happiness


# ===================================================================
# 6. ML DATA TESTS
# ===================================================================
def test_ml_dataset_basic():
    """Verify MLDataset add/get/size/clear operations."""
    ds = MLDataset()
    assert ds.size() == 0

    X = np.random.randn(40).astype(np.float32)
    Y = np.random.randn(3).astype(np.float32)
    ds.add_sample(X, Y)
    assert ds.size() == 1

    X_arr, Y_arr = ds.get_arrays()
    assert X_arr.shape == (1, 40)
    assert Y_arr.shape == (1, 3)

    ds.clear()
    assert ds.size() == 0


def test_ml_dataset_save_load(tmp_path=None):
    """Verify CSV save/load roundtrip."""
    filepath = "/tmp/test_nagorik_dataset.csv"
    ds = MLDataset()
    for _ in range(10):
        ds.add_sample(np.random.randn(40).astype(np.float32), np.random.randn(3).astype(np.float32))

    ds.save_to_csv(filepath)
    assert os.path.exists(filepath)

    ds2 = MLDataset()
    ds2.load_from_csv(filepath)
    assert ds2.size() == 10

    X1, Y1 = ds.get_arrays()
    X2, Y2 = ds2.get_arrays()
    np.testing.assert_array_almost_equal(X1, X2, decimal=5)
    np.testing.assert_array_almost_equal(Y1, Y2, decimal=5)

    os.remove(filepath)


def test_ml_dataset_merge():
    """Verify dataset merging."""
    ds1 = MLDataset()
    ds2 = MLDataset()
    for _ in range(5):
        ds1.add_sample(np.random.randn(40).astype(np.float32), np.random.randn(3).astype(np.float32))
        ds2.add_sample(np.random.randn(40).astype(np.float32), np.random.randn(3).astype(np.float32))

    ds1.merge(ds2)
    assert ds1.size() == 10


# ===================================================================
# 7. NN MODEL TESTS
# ===================================================================
def test_nn_model_untrained():
    """Verify untrained model raises on predict."""
    model = CitizenReactionModel(hidden_layers=(128, 64, 32))
    assert not model.is_trained
    try:
        model.predict(np.random.randn(1, 40))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_nn_model_train_predict():
    """Verify model can train and predict."""
    model = CitizenReactionModel(hidden_layers=(32, 16), max_iter=50)
    X = np.random.randn(100, 40).astype(np.float32)
    Y = np.random.randn(100, 3).astype(np.float32)

    metrics = model.train(X, Y)
    assert model.is_trained
    assert "train_mae" in metrics
    assert metrics["n_samples"] == 100

    pred = model.predict(X[:5])
    assert pred.shape == (5, 3)


def test_nn_model_save_load():
    """Verify model save/load roundtrip."""
    model = CitizenReactionModel(hidden_layers=(32, 16), max_iter=50)
    X = np.random.randn(100, 40).astype(np.float32)
    Y = np.random.randn(100, 3).astype(np.float32)
    model.train(X, Y)

    model.save("/tmp/test_model.joblib")

    loaded = CitizenReactionModel.load("/tmp/test_model.joblib")
    assert loaded is not None
    assert loaded.is_trained

    pred_orig = model.predict(X[:5])
    pred_loaded = loaded.predict(X[:5])
    np.testing.assert_array_almost_equal(pred_orig, pred_loaded)

    os.remove("/tmp/test_model.joblib")


# ===================================================================
# 8. INTEGRATION TESTS
# ===================================================================
def test_end_to_end_rule_based_simulation():
    """Full simulation with rule-based only (no LLM needed)."""
    citizens = generate_population(50, seed=42)
    rng = np.random.default_rng(42)

    all_states = []
    for c in citizens:
        all_states.append(CitizenState(
            citizen_id=c.id, step=0, happiness=c.base_happiness,
            policy_support=0.0, income=c.base_income
        ))

    for step in range(1, 4):
        for c in citizens:
            prev = [s for s in all_states if s.citizen_id == c.id and s.step == step - 1][0]
            h, s, i = rule_based_update(c, prev, "Economy", rng)
            all_states.append(CitizenState(
                citizen_id=c.id, step=step, happiness=h, policy_support=s, income=i
            ))

    # Verify we have 50 * 4 states
    assert len(all_states) == 200, f"Expected 200 states, got {len(all_states)}"

    # Compute stats
    stats = compute_time_series_stats(all_states, citizens, 3)
    assert len(stats) == 4  # Steps 0-3

    # Build DataFrame
    df = build_stats_dataframe(all_states, citizens)
    assert len(df) == 200
    assert "division" in df.columns


def test_feature_vector_to_nn_pipeline():
    """Test full pipeline: citizen → feature vector → NN prediction → apply deltas."""
    citizens = generate_population(200, seed=42)
    rng = np.random.default_rng(42)

    # Generate fake training data
    X_train, Y_train = [], []
    for c in citizens:
        prev = CitizenState(citizen_id=c.id, step=0, happiness=c.base_happiness,
                           policy_support=0.0, income=c.base_income)
        X = build_feature_vector(c, prev, "Economy")
        h, s, i = rule_based_update(c, prev, "Economy", rng)
        Y = compute_deltas(prev, h, s, i)
        X_train.append(X)
        Y_train.append(Y)

    X_arr = np.vstack(X_train)
    Y_arr = np.vstack(Y_train)
    assert X_arr.shape == (200, 40)
    assert Y_arr.shape == (200, 3)

    # Train a small NN
    model = CitizenReactionModel(hidden_layers=(32, 16), max_iter=100)
    metrics = model.train(X_arr, Y_arr)
    assert model.is_trained

    # Predict and apply
    test_citizen = citizens[0]
    test_prev = CitizenState(citizen_id=test_citizen.id, step=0,
                             happiness=test_citizen.base_happiness,
                             policy_support=0.0, income=test_citizen.base_income)
    test_X = build_feature_vector(test_citizen, test_prev, "Economy")
    deltas = model.predict(test_X.reshape(1, -1))[0]
    h, s, i = apply_deltas(test_prev, deltas)

    assert 0.0 <= h <= 1.0
    assert -1.0 <= s <= 1.0
    assert i >= 0.0


# ===================================================================
# RUN ALL TESTS
# ===================================================================
def main():
    global PASSED, FAILED, ERRORS

    print("=" * 70)
    print("  NAGORIK-GENESIS Test Suite")
    print("=" * 70)

    # 1. Data Models
    print("\n📦 Data Models")
    run_test("constants_count", test_constants_count)
    run_test("constants_values", test_constants_values)
    run_test("citizen_dataclass_fields", test_citizen_dataclass_fields)
    run_test("step_stats_fields", test_step_stats_fields)
    run_test("expert_summary_fields", test_expert_summary_fields)

    # 2. Population
    print("\n👥 Population Generation")
    run_test("population_size", test_population_size)
    run_test("population_income_distribution", test_population_income_distribution)
    run_test("population_custom_income_shares", test_population_custom_income_shares)
    run_test("population_age_range", test_population_age_range)
    run_test("population_youth_bulge", test_population_youth_bulge)
    run_test("population_gender_distribution", test_population_gender_distribution)
    run_test("population_divisions", test_population_divisions)
    run_test("population_division_weights", test_population_division_weights)
    run_test("population_religions", test_population_religions)
    run_test("population_remittance", test_population_remittance)
    run_test("population_income_ranges_bdt", test_population_income_ranges_bdt)
    run_test("population_education_income_correlation", test_population_education_income_correlation)
    run_test("population_professions", test_population_professions)
    run_test("population_family_size_zone_dependent", test_population_family_size_zone_dependent)
    run_test("population_reproducibility", test_population_reproducibility)

    # 3. Utils / Feature Vector
    print("\n🔧 Utils & Feature Vector")
    run_test("format_income_bdt", test_format_income_bdt)
    run_test("format_support", test_format_support)
    run_test("one_hot_encoding", test_one_hot_encoding)
    run_test("one_hot_unknown", test_one_hot_unknown)
    run_test("feature_dimension", test_feature_dimension)
    run_test("build_feature_vector_shape", test_build_feature_vector_shape)
    run_test("build_feature_vector_values", test_build_feature_vector_values)
    run_test("build_feature_vector_all_domains", test_build_feature_vector_all_domains)
    run_test("compute_deltas", test_compute_deltas)
    run_test("apply_deltas_clamping", test_apply_deltas_clamping)
    run_test("policy_presets_count", test_policy_presets_count)
    run_test("policy_presets_domains", test_policy_presets_domains)
    run_test("policy_presets_bengali", test_policy_presets_bengali)
    run_test("citizen_to_dict", test_citizen_to_dict)

    # 4. Rule-Based Engine
    print("\n⚙️ Rule-Based Engine")
    run_test("rule_based_all_domains", test_rule_based_all_domains)
    run_test("rule_based_climate_low_income_devastation", test_rule_based_climate_low_income_devastation)
    run_test("rule_based_income_never_negative", test_rule_based_income_never_negative)
    run_test("rule_based_social_redistributive", test_rule_based_social_redistributive)

    # 5. Stats
    print("\n📊 Statistics")
    run_test("compute_step_stats", test_compute_step_stats)
    run_test("build_stats_dataframe", test_build_stats_dataframe)
    run_test("compute_time_series_stats", test_compute_time_series_stats)

    # 6. ML Data
    print("\n💾 ML Data Management")
    run_test("ml_dataset_basic", test_ml_dataset_basic)
    run_test("ml_dataset_save_load", test_ml_dataset_save_load)
    run_test("ml_dataset_merge", test_ml_dataset_merge)

    # 7. NN Model
    print("\n🧠 Neural Network Model")
    run_test("nn_model_untrained", test_nn_model_untrained)
    run_test("nn_model_train_predict", test_nn_model_train_predict)
    run_test("nn_model_save_load", test_nn_model_save_load)

    # 8. Integration
    print("\n🔗 Integration Tests")
    run_test("end_to_end_rule_based_simulation", test_end_to_end_rule_based_simulation)
    run_test("feature_vector_to_nn_pipeline", test_feature_vector_to_nn_pipeline)

    # Summary
    total = PASSED + FAILED
    print("\n" + "=" * 70)
    print(f"  Results: {PASSED}/{total} passed, {FAILED} failed")
    print("=" * 70)

    if ERRORS:
        print("\nFailed tests:")
        for e in ERRORS:
            print(e)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    exit(main())
