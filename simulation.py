"""
Core simulation engine for নাগরিক-GENESIS (NAGORIK-GENESIS).
Runs multi-step simulations using LLM, NN, and rule-based updates — Bangladesh context.
"""
import numpy as np
import time
from typing import List, Dict, Any, Optional, Set
from data_models import Citizen, CitizenState, SimulationConfig
from ml_data import MLDataset
from utils import build_feature_vector, compute_deltas, apply_deltas, citizen_to_dict, state_to_dict
from stats import compute_time_series_stats
import logging

logger = logging.getLogger(__name__)


def rule_based_update(
    citizen: Citizen,
    prev_state: CitizenState,
    policy_domain: str,
    rng: np.random.Generator
) -> tuple[float, float, float]:
    """
    Compute rule-based update for a Bangladeshi citizen's state.

    Uses income-dependent effects and personality traits to compute
    realistic changes without calling an LLM. All income deltas in BDT.

    Args:
        citizen: Citizen object with attributes.
        prev_state: Previous CitizenState.
        policy_domain: Policy domain (7 Bangladesh domains).
        rng: NumPy random generator.

    Returns:
        Tuple of (new_happiness, new_policy_support, new_income).
    """
    # ── 1. Base income delta: domain × income level ─────────────────────────
    income_delta = 0.0

    if policy_domain == "Economy":
        if citizen.income_level == "low":
            income_delta = float(rng.uniform(-2000, 1000))
        elif citizen.income_level == "middle":
            income_delta = float(rng.uniform(-5000, 3000))
        else:
            income_delta = float(rng.uniform(-20000, 15000))

    elif policy_domain == "Education":
        if citizen.income_level == "low":
            income_delta = float(rng.uniform(-500, 2000))
        elif citizen.income_level == "middle":
            income_delta = float(rng.uniform(-1000, 1500))
        else:
            income_delta = float(rng.uniform(-3000, 1000))

    elif policy_domain == "Social":
        if citizen.income_level == "low":
            income_delta = float(rng.uniform(0, 3000))
        elif citizen.income_level == "middle":
            income_delta = float(rng.uniform(-1000, 1500))
        else:
            income_delta = float(rng.uniform(-10000, 0))

    elif policy_domain == "Digital & Technology":
        if citizen.income_level == "low":
            income_delta = float(rng.uniform(-1000, 500))
        elif citizen.income_level == "middle":
            income_delta = float(rng.uniform(-2000, 3000))
        else:
            income_delta = float(rng.uniform(-3000, 10000))

    elif policy_domain == "Infrastructure":
        if citizen.income_level == "low":
            income_delta = float(rng.uniform(-500, 2500))
        elif citizen.income_level == "middle":
            income_delta = float(rng.uniform(-1000, 3000))
        else:
            income_delta = float(rng.uniform(-5000, 8000))

    elif policy_domain == "Climate & Disaster":
        if citizen.income_level == "low":
            income_delta = float(rng.uniform(-5000, 500))
        elif citizen.income_level == "middle":
            income_delta = float(rng.uniform(-3000, 1000))
        else:
            income_delta = float(rng.uniform(-2000, 2000))

    elif policy_domain == "Healthcare":
        if citizen.income_level == "low":
            income_delta = float(rng.uniform(-1000, 2000))
        elif citizen.income_level == "middle":
            income_delta = float(rng.uniform(-2000, 1500))
        else:
            income_delta = float(rng.uniform(-5000, 3000))

    # ── 2. Zone multiplier (slum absorbs more shock; affluent zone buffers) ─
    zone_multiplier = {
        "urban_rich": 0.6,
        "urban_middle": 0.9,
        "urban_poor": 1.3,
        "bosti_slum": 1.6,
        "rural": 1.0,
    }.get(getattr(citizen, "city_zone", "urban_middle"), 1.0)
    income_delta *= zone_multiplier

    # ── 3. Profession × domain affinity ─────────────────────────────────────
    profession = getattr(citizen, "profession", "")
    prof_boost = 0.0
    if policy_domain == "Economy" and any(k in profession for k in ["Garment", "Rickshaw", "Street vendor", "Day laborer", "Factory"]):
        prof_boost = float(rng.uniform(0.05, 0.15))   # directly affected by economy
    elif policy_domain == "Education" and any(k in profession for k in ["Teacher", "Student", "Tutor"]):
        prof_boost = float(rng.uniform(0.08, 0.18))
    elif policy_domain == "Healthcare" and any(k in profession for k in ["Doctor", "Nurse", "Healthcare"]):
        prof_boost = float(rng.uniform(0.06, 0.14))
    elif policy_domain == "Climate & Disaster" and any(k in profession for k in ["Farmer", "Fisherman", "Agricultural"]):
        prof_boost = float(rng.uniform(0.10, 0.25))   # farmers hit hardest by climate
    elif policy_domain == "Digital & Technology" and any(k in profession for k in ["Software", "Freelancer", "IT", "Engineer"]):
        prof_boost = float(rng.uniform(0.08, 0.20))
    elif policy_domain == "Infrastructure" and any(k in profession for k in ["Rickshaw", "CNG", "Bus", "Transport", "Driver"]):
        prof_boost = float(rng.uniform(0.05, 0.15))

    # ── 4. Division × domain effects ────────────────────────────────────────
    division = getattr(citizen, "division", "Dhaka")
    div_modifier = 0.0
    if policy_domain == "Climate & Disaster" and division in ["Barishal", "Khulna"]:
        div_modifier = -0.12   # coastal divisions harder hit
    elif policy_domain == "Economy" and division == "Sylhet":
        div_modifier = 0.08    # remittance-heavy division benefits from forex policies
    elif policy_domain in ["Digital & Technology", "Infrastructure"] and division == "Dhaka":
        div_modifier = 0.06    # Dhaka benefits most from tech/infra
    elif policy_domain == "Education" and division in ["Rajshahi", "Rangpur"]:
        div_modifier = 0.07    # northern divisions benefit more from education

    # ── 5. Remittance family boost (Economy policies) ───────────────────────
    remittance_modifier = 0.0
    is_remittance = getattr(citizen, "is_remittance_family", False)
    if is_remittance and policy_domain == "Economy":
        remittance_modifier = float(rng.uniform(0.06, 0.14))

    # ── 6. Family size amplifier (Social/Healthcare) ────────────────────────
    family_size = getattr(citizen, "family_size", 4)
    family_modifier = 0.0
    if policy_domain in ["Social", "Healthcare"] and family_size >= 5:
        family_modifier = (family_size - 4) * 0.03   # +3% happiness per extra member

    # ── 7. Political view modifier ───────────────────────────────────────────
    political_view = getattr(citizen, "political_view", "Neutral")
    pol_support_modifier = 0.0
    if political_view == "Government Supporter":
        pol_support_modifier = float(rng.uniform(0.08, 0.18))
    elif political_view == "Opposition":
        pol_support_modifier = float(rng.uniform(-0.18, -0.06))
    elif political_view == "Islamist":
        # Islamists oppose secular/digital policies, support social safety net
        if policy_domain in ["Digital & Technology"]:
            pol_support_modifier = float(rng.uniform(-0.12, -0.03))
        elif policy_domain == "Social":
            pol_support_modifier = float(rng.uniform(0.03, 0.10))
    # Neutral and Progressive get no modifier

    # ── Combine all factors ──────────────────────────────────────────────────
    personality_factor = (
        (citizen.openness_to_change - 0.5) * 0.4 +
        (citizen.risk_tolerance - 0.5) * 0.4
    )

    # State momentum: very unhappy citizens react more dramatically
    despair_amp = 1.0 + max(0.0, (0.3 - prev_state.happiness)) * 0.8

    income_impact = income_delta / max(prev_state.income, 1)
    happ_delta = (
        income_impact * 0.5
        + personality_factor
        + prof_boost
        + div_modifier
        + remittance_modifier
        + family_modifier
    ) * despair_amp

    support_delta = happ_delta * 0.8 + pol_support_modifier

    # Add noise
    happ_delta += float(rng.uniform(-0.05, 0.05))
    support_delta += float(rng.uniform(-0.05, 0.05))

    # Compute new values with clamping
    new_happiness = float(np.clip(prev_state.happiness + happ_delta, 0.0, 1.0))
    new_policy_support = float(np.clip(prev_state.policy_support + support_delta, -1.0, 1.0))
    new_income = float(max(0.0, prev_state.income + income_delta))

    return new_happiness, new_policy_support, new_income


def run_simulation(
    config: SimulationConfig,
    citizens: List[Citizen],
    llm_client: Any,
    existing_model: Optional[Any] = None,
    feature_scaler: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run a multi-step simulation of Bangladeshi citizen reactions to a policy.

    This is the main simulation engine that coordinates LLM calls, NN predictions,
    and rule-based updates based on the simulation mode.

    Args:
        config: SimulationConfig with population size, steps, policy, and mode.
        citizens: List of Citizen objects.
        llm_client: GeminiClient instance for LLM calls.
        existing_model: Pre-trained CitizenReactionModel (optional).
        feature_scaler: Feature scaler for NN (optional).

    Returns:
        Dict containing simulation results, states, stats, and training data.
    """
    logger.info(f"Starting simulation '{config.name}' with mode {config.mode}")

    # Initialize random generator
    if config.random_seed is not None:
        rng = np.random.default_rng(config.random_seed)
    else:
        rng = np.random.default_rng()

    # Initialize storage
    all_states = []
    training_dataset = MLDataset()

    # Track NN usage statistics
    nn_stats = {
        "total_nn_predictions": 0,
        "total_llm_calls": 0,
        "total_rule_based": 0,
        "nn_prediction_times": [],
        "step_breakdown": []
    }

    # Step 0: Initialize all citizens
    for citizen in citizens:
        state = CitizenState(
            citizen_id=citizen.id,
            step=0,
            happiness=citizen.base_happiness,
            policy_support=0.0,
            income=citizen.base_income,
            diary_entry=None,
            llm_updated=False
        )
        all_states.append(state)

    logger.info(f"Initialized {len(citizens)} citizens at step 0")

    # Map for quick state lookup
    def get_prev_state(citizen_id: int, step: int) -> CitizenState:
        """Get the previous state for a citizen."""
        for state in reversed(all_states):
            if state.citizen_id == citizen_id and state.step == step - 1:
                return state
        raise ValueError(f"No previous state found for citizen {citizen_id} at step {step - 1}")

    # Simulate each step
    for step in range(1, config.steps + 1):
        logger.info(f"Simulating step {step}/{config.steps}...")

        # Track counts for this step
        step_llm_count = 0
        step_nn_count = 0
        step_rule_count = 0

        # Determine LLM sample for this step
        # In LLM_ONLY mode, send ALL citizens to LLM for maximum training data quality.
        # In HYBRID mode, cap at 300 to balance LLM cost with NN coverage.
        if config.mode == "LLM_ONLY":
            llm_sample_size = len(citizens)
        else:
            llm_sample_size = min(300, len(citizens))
        llm_sample_indices = rng.choice(len(citizens), size=llm_sample_size, replace=False)
        llm_sample_ids: Set[int] = {citizens[i].id for i in llm_sample_indices}

        # Process each citizen
        for citizen_idx, citizen in enumerate(citizens):
            # Progress every 10 citizens
            if citizen_idx % 10 == 0:
                logger.info(f"  Step {step}: citizen {citizen_idx+1}/{len(citizens)} ...")

            prev_state = get_prev_state(citizen.id, step)

            # Build feature vector
            X = build_feature_vector(citizen, prev_state, config.policy.domain)

            # Determine update method based on mode
            use_llm = False
            use_nn = False

            if config.mode == "LLM_ONLY":
                use_llm = citizen.id in llm_sample_ids
            elif config.mode == "HYBRID":
                if citizen.id in llm_sample_ids:
                    use_llm = True
                elif existing_model and existing_model.is_trained:
                    use_nn = True
            elif config.mode == "NN_ONLY":
                if existing_model and existing_model.is_trained:
                    use_nn = True

            # Apply update
            if use_llm:
                # LLM-based update
                try:
                    citizen_profile = citizen_to_dict(citizen)
                    current_state = state_to_dict(prev_state)
                    policy_dict = {
                        "title": config.policy.title,
                        "description": config.policy.description,
                        "domain": config.policy.domain
                    }

                    reaction = llm_client.generate_citizen_reaction(
                        citizen_profile, current_state, policy_dict
                    )

                    new_happiness = reaction["new_happiness"]
                    new_policy_support = reaction["new_policy_support"]
                    income_delta = reaction["income_delta"]
                    new_income = max(0.0, prev_state.income + income_delta)
                    diary_entry = reaction.get("diary_entry")

                    # Store training sample
                    Y = compute_deltas(prev_state, new_happiness, new_policy_support, new_income)
                    training_dataset.add_sample(X, Y)

                    # Create new state
                    new_state = CitizenState(
                        citizen_id=citizen.id,
                        step=step,
                        happiness=new_happiness,
                        policy_support=new_policy_support,
                        income=new_income,
                        diary_entry=diary_entry,
                        llm_updated=True
                    )
                    step_llm_count += 1

                except Exception as e:
                    logger.error(f"LLM error for citizen {citizen.id}: {e}")

                    # In HYBRID mode, try to fall back to NN if available
                    if config.mode == "HYBRID" and existing_model and existing_model.is_trained:
                        logger.info(f"Falling back to NN for citizen {citizen.id}")
                        try:
                            # Apply scaler if available
                            if feature_scaler:
                                X_scaled = feature_scaler.transform(X.reshape(1, -1))
                            else:
                                X_scaled = X.reshape(1, -1)

                            deltas = existing_model.predict(X_scaled)[0]
                            new_happiness, new_policy_support, new_income = apply_deltas(prev_state, deltas)

                            new_state = CitizenState(
                                citizen_id=citizen.id,
                                step=step,
                                happiness=new_happiness,
                                policy_support=new_policy_support,
                                income=new_income,
                                diary_entry=None,
                                llm_updated=False
                            )
                            step_nn_count += 1
                        except Exception as nn_e:
                            logger.error(f"NN fallback also failed for citizen {citizen.id}: {nn_e}, using rule-based")
                            new_happiness, new_policy_support, new_income = rule_based_update(
                                citizen, prev_state, config.policy.domain, rng
                            )
                            new_state = CitizenState(
                                citizen_id=citizen.id,
                                step=step,
                                happiness=new_happiness,
                                policy_support=new_policy_support,
                                income=new_income,
                                diary_entry=None,
                                llm_updated=False
                            )
                            step_rule_count += 1
                    else:
                        # No NN available, fall back to rule-based
                        logger.info(f"No NN available, using rule-based for citizen {citizen.id}")
                        new_happiness, new_policy_support, new_income = rule_based_update(
                            citizen, prev_state, config.policy.domain, rng
                        )
                        new_state = CitizenState(
                            citizen_id=citizen.id,
                            step=step,
                            happiness=new_happiness,
                            policy_support=new_policy_support,
                            income=new_income,
                            diary_entry=None,
                            llm_updated=False
                        )
                        step_rule_count += 1

            elif use_nn:
                # NN-based update
                try:
                    # Time the prediction
                    nn_start = time.time()

                    # Apply scaler if available
                    if feature_scaler:
                        X_scaled = feature_scaler.transform(X.reshape(1, -1))
                    else:
                        X_scaled = X.reshape(1, -1)

                    # Explicit null check before prediction
                    if existing_model is None:
                        raise ValueError("Neural network model is None - cannot predict")

                    deltas = existing_model.predict(X_scaled)[0]
                    new_happiness, new_policy_support, new_income = apply_deltas(prev_state, deltas)

                    nn_time = time.time() - nn_start
                    nn_stats["nn_prediction_times"].append(nn_time)

                    new_state = CitizenState(
                        citizen_id=citizen.id,
                        step=step,
                        happiness=new_happiness,
                        policy_support=new_policy_support,
                        income=new_income,
                        diary_entry=None,
                        llm_updated=False
                    )
                    step_nn_count += 1

                except Exception as e:
                    logger.error(f"NN error for citizen {citizen.id}: {e}, falling back to rule-based")
                    new_happiness, new_policy_support, new_income = rule_based_update(
                        citizen, prev_state, config.policy.domain, rng
                    )
                    new_state = CitizenState(
                        citizen_id=citizen.id,
                        step=step,
                        happiness=new_happiness,
                        policy_support=new_policy_support,
                        income=new_income,
                        diary_entry=None,
                        llm_updated=False
                    )
                    step_rule_count += 1

            else:
                # Rule-based update
                new_happiness, new_policy_support, new_income = rule_based_update(
                    citizen, prev_state, config.policy.domain, rng
                )
                new_state = CitizenState(
                    citizen_id=citizen.id,
                    step=step,
                    happiness=new_happiness,
                    policy_support=new_policy_support,
                    income=new_income,
                    diary_entry=None,
                    llm_updated=False
                )
                step_rule_count += 1

            all_states.append(new_state)

        # Record step statistics
        nn_stats["step_breakdown"].append({
            "step": step,
            "llm_calls": step_llm_count,
            "nn_predictions": step_nn_count,
            "rule_based": step_rule_count
        })
        nn_stats["total_llm_calls"] += step_llm_count
        nn_stats["total_nn_predictions"] += step_nn_count
        nn_stats["total_rule_based"] += step_rule_count

        logger.info(f"Step {step} complete. LLM: {step_llm_count}, NN: {step_nn_count}, Rule: {step_rule_count}")

    # Compute statistics for all steps
    logger.info("Computing statistics...")
    step_stats = compute_time_series_stats(all_states, citizens, config.steps)

    # Calculate NN performance metrics
    if nn_stats["nn_prediction_times"]:
        nn_stats["avg_prediction_time_ms"] = np.mean(nn_stats["nn_prediction_times"]) * 1000
        nn_stats["total_prediction_time_seconds"] = sum(nn_stats["nn_prediction_times"])
        nn_stats["predictions_per_second"] = (
            nn_stats["total_nn_predictions"] / sum(nn_stats["nn_prediction_times"])
            if sum(nn_stats["nn_prediction_times"]) > 0 else 0
        )
    else:
        nn_stats["avg_prediction_time_ms"] = 0
        nn_stats["total_prediction_time_seconds"] = 0
        nn_stats["predictions_per_second"] = 0

    # Return results
    results = {
        "config": config,
        "citizens": citizens,
        "all_states": all_states,
        "step_stats": step_stats,
        "training_dataset": training_dataset,
        "mode_used": config.mode,
        "nn_stats": nn_stats
    }

    logger.info(f"Simulation complete. LLM calls: {nn_stats['total_llm_calls']}, "
                f"NN predictions: {nn_stats['total_nn_predictions']}, "
                f"Collected {training_dataset.size()} training samples.")

    return results
