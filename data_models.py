"""
Core data models and domain structures for নাগরিক-GENESIS (NAGORIK-GENESIS).
Defines citizens, states, policies, and simulation results — localized for Bangladesh.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Citizen:
    """Represents a synthetic Bangladeshi citizen with demographic and personality attributes."""

    id: int
    age: int
    gender: str  # Male, Female, Hijra
    city_zone: str  # shohor_kendro, shilpo_elaka, uposhohon, graam, bosti
    income_level: str  # low, middle, high
    education: str  # No Formal Education, Madrasa Education, SSC, HSC, Honours/Bachelor's, Masters, PhD
    profession: str
    family_size: int
    political_view: str  # government_supporter, opposition_supporter, neutral, civil_society_aligned
    risk_tolerance: float  # 0-1
    openness_to_change: float  # 0-1
    base_happiness: float  # 0-1
    base_income: float  # >= 0 (BDT monthly)
    division: str  # Dhaka, Chittagong, Rajshahi, Khulna, Barisal, Sylhet, Rangpur, Mymensingh
    religion: str  # Muslim, Hindu, Buddhist, Christian, Other
    is_remittance_family: bool  # Whether family receives remittance from abroad


@dataclass
class CitizenState:
    """Represents the state of a citizen at a specific simulation step."""

    citizen_id: int
    step: int
    happiness: float  # 0-1
    policy_support: float  # -1 to 1
    income: float  # >= 0 (BDT monthly)
    diary_entry: Optional[str] = None
    llm_updated: bool = False  # True if updated via LLM, False if rule-based or NN


@dataclass
class PolicyInput:
    """Represents a policy or business idea to be simulated."""

    title: str
    description: str
    domain: str  # Economy, Education, Social, Digital & Technology, Infrastructure, Climate & Disaster, Healthcare


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    name: str
    population_size: int
    steps: int
    policy: PolicyInput
    random_seed: Optional[int] = None
    mode: str = "LLM_ONLY"  # LLM_ONLY, HYBRID, NN_ONLY


@dataclass
class StepStats:
    """Aggregated statistics for a simulation step."""

    step: int
    avg_happiness: float
    avg_support: float
    avg_income: float
    by_income: List[Dict[str, Any]] = field(default_factory=list)
    by_zone: List[Dict[str, Any]] = field(default_factory=list)
    by_division: List[Dict[str, Any]] = field(default_factory=list)
    by_religion: List[Dict[str, Any]] = field(default_factory=list)
    inequality_gap_happiness: float = 0.0  # high minus low income avg happiness
    inequality_gap_support: float = 0.0  # high minus low income avg support


@dataclass
class ExpertSummary:
    """Expert perspectives on simulation results — Bangladesh context."""

    step: int
    economist_view: str
    activist_view: str
    garment_industry_view: str
    rural_leader_view: str


# Constants for domains and attributes — Bangladesh localized
DOMAINS = ["Economy", "Education", "Social", "Digital & Technology", "Infrastructure", "Climate & Disaster", "Healthcare"]
CITY_ZONES = ["shohor_kendro", "shilpo_elaka", "uposhohon", "graam", "bosti"]
INCOME_LEVELS = ["low", "middle", "high"]
POLITICAL_VIEWS = ["government_supporter", "opposition_supporter", "neutral", "civil_society_aligned"]
GENDERS = ["Male", "Female", "Hijra"]
EDUCATION_LEVELS = ["No Formal Education", "Madrasa Education", "SSC", "HSC", "Honours/Bachelor's", "Masters", "PhD"]
DIVISIONS = ["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Barisal", "Sylhet", "Rangpur", "Mymensingh"]
RELIGIONS = ["Muslim", "Hindu", "Buddhist", "Christian", "Other"]
SIMULATION_MODES = ["LLM_ONLY", "HYBRID", "NN_ONLY"]
