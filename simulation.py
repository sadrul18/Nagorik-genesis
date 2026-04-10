"""
Core simulation engine for নাগরিক-GENESIS (NAGORIK-GENESIS).
Runs multi-step simulations using LLM, NN, and rule-based updates — Bangladesh context.
"""
import numpy as np
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set, Tuple
from data_models import Citizen, CitizenState, SimulationConfig, DIVISIONS, INCOME_LEVELS
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


def _select_stratified_anchors(
    citizens: List[Citizen],
    rng: np.random.Generator,
    per_cell: int = 3,
) -> Set[int]:
    """
    Select a stratified sample of citizens covering all (income_level × division) cells.

    Returns a set of citizen IDs chosen so that every stratum is represented.
    For a population of 5000 with 3 income levels × 8 divisions = 24 cells,
    this selects ~72 citizens (3 per cell) instead of 300 random ones — fewer
    LLM calls but guaranteed full coverage.
    """
    # Bucket citizens by (income_level, division)
    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for c in citizens:
        buckets[(c.income_level, c.division)].append(c.id)

    anchor_ids: Set[int] = set()
    for key, cid_list in buckets.items():
        n_pick = min(per_cell, len(cid_list))
        chosen = rng.choice(cid_list, size=n_pick, replace=False)
        anchor_ids.update(chosen.tolist())

    logger.info(
        f"Stratified anchors: {len(anchor_ids)} citizens across "
        f"{len(buckets)} strata ({per_cell} per cell)"
    )
    return anchor_ids


def _compute_calibration_bounds(
    anchor_states: List[CitizenState],
    citizens: List[Citizen],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Compute per-stratum (income_level × division) statistics from anchor LLM results.

    Returns a dict mapping (income_level, division) → {mean_h, mean_s, mean_i, std_h, std_s, std_i}.
    """
    citizen_map = {c.id: c for c in citizens}
    buckets: Dict[Tuple[str, str], Dict[str, list]] = defaultdict(
        lambda: {"h": [], "s": [], "i": []}
    )

    for st in anchor_states:
        c = citizen_map[st.citizen_id]
        key = (c.income_level, c.division)
        buckets[key]["h"].append(st.happiness)
        buckets[key]["s"].append(st.policy_support)
        buckets[key]["i"].append(st.income)

    bounds = {}
    for key, vals in buckets.items():
        bounds[key] = {
            "mean_h": float(np.mean(vals["h"])),
            "mean_s": float(np.mean(vals["s"])),
            "mean_i": float(np.mean(vals["i"])),
            "std_h": max(float(np.std(vals["h"])), 0.02),
            "std_s": max(float(np.std(vals["s"])), 0.02),
            "std_i": max(float(np.std(vals["i"])), 500.0),
        }
    return bounds


def _calibrate_prediction(
    new_h: float,
    new_s: float,
    new_i: float,
    citizen: Citizen,
    bounds: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[float, float, float]:
    """
    Shift/clamp NN-predicted values toward the anchor-derived stratum bounds.

    Uses a 2-sigma window: if the prediction is within [mean-2σ, mean+2σ] of the
    anchor group, keep it. Otherwise clamp to the boundary.
    """
    key = (citizen.income_level, citizen.division)
    if key not in bounds:
        return new_h, new_s, new_i

    b = bounds[key]
    new_h = float(np.clip(new_h, b["mean_h"] - 2 * b["std_h"], b["mean_h"] + 2 * b["std_h"]))
    new_s = float(np.clip(new_s, b["mean_s"] - 2 * b["std_s"], b["mean_s"] + 2 * b["std_s"]))
    new_i = float(np.clip(new_i, b["mean_i"] - 2 * b["std_i"], b["mean_i"] + 2 * b["std_i"]))

    # Enforce hard bounds
    new_h = float(np.clip(new_h, 0.0, 1.0))
    new_s = float(np.clip(new_s, -1.0, 1.0))
    new_i = max(0.0, new_i)
    return new_h, new_s, new_i


def run_simulation(
    config: SimulationConfig,
    citizens: List[Citizen],
    llm_client: Any,
    existing_model: Optional[Any] = None,
    feature_scaler: Optional[Any] = None,
    target_scaler: Optional[Any] = None,
    knowledge_context: str = ""
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
        target_scaler: Target scaler for inverse-transforming NN predictions (optional).
        knowledge_context: Web-sourced real-world knowledge about the policy (optional).

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
        if config.mode == "LLM_ONLY":
            llm_sample_size = len(citizens)
            llm_sample_indices = rng.choice(len(citizens), size=llm_sample_size, replace=False)
            llm_sample_ids: Set[int] = {citizens[i].id for i in llm_sample_indices}
        elif config.mode == "HYBRID":
            # Stratified anchor selection — guaranteed coverage of all strata
            llm_sample_ids = _select_stratified_anchors(citizens, rng, per_cell=3)
        else:
            llm_sample_ids = set()

        # ── Phase 1 (HYBRID): Process anchor citizens via LLM first ────────
        anchor_states_this_step: List[CitizenState] = []
        calibration_bounds: Dict = {}

        if config.mode == "HYBRID":
            anchor_citizens = [c for c in citizens if c.id in llm_sample_ids]
            for citizen in anchor_citizens:
                prev_state = get_prev_state(citizen.id, step)
                X = build_feature_vector(citizen, prev_state, config.policy.domain)
                try:
                    citizen_profile = citizen_to_dict(citizen)
                    current_state = state_to_dict(prev_state)
                    policy_dict = {
                        "title": config.policy.title,
                        "description": config.policy.description,
                        "domain": config.policy.domain
                    }
                    reaction = llm_client.generate_citizen_reaction(
                        citizen_profile, current_state, policy_dict,
                        knowledge_context=knowledge_context
                    )
                    new_happiness = reaction["new_happiness"]
                    new_policy_support = reaction["new_policy_support"]
                    income_delta = reaction["income_delta"]
                    new_income = max(0.0, prev_state.income + income_delta)
                    diary_entry = reaction.get("diary_entry")

                    Y = compute_deltas(prev_state, new_happiness, new_policy_support, new_income)
                    training_dataset.add_sample(X, Y)

                    new_state = CitizenState(
                        citizen_id=citizen.id, step=step,
                        happiness=new_happiness, policy_support=new_policy_support,
                        income=new_income, diary_entry=diary_entry, llm_updated=True,
                    )
                    all_states.append(new_state)
                    anchor_states_this_step.append(new_state)
                    step_llm_count += 1
                except Exception as e:
                    logger.error(f"LLM error for anchor citizen {citizen.id}: {e}")
                    # Fall back to rule-based for this anchor
                    new_happiness, new_policy_support, new_income = rule_based_update(
                        citizen, prev_state, config.policy.domain, rng
                    )
                    new_state = CitizenState(
                        citizen_id=citizen.id, step=step,
                        happiness=new_happiness, policy_support=new_policy_support,
                        income=new_income, diary_entry=None, llm_updated=False,
                    )
                    all_states.append(new_state)
                    anchor_states_this_step.append(new_state)
                    step_rule_count += 1

            # ── Phase 2: Compute calibration bounds from anchors ────────────
            if anchor_states_this_step:
                calibration_bounds = _compute_calibration_bounds(
                    anchor_states_this_step, citizens
                )
                logger.info(
                    f"Step {step}: calibration bounds from "
                    f"{len(anchor_states_this_step)} anchors across "
                    f"{len(calibration_bounds)} strata"
                )

        # ── Phase 3: Process remaining citizens ────────────────────────────
        for citizen_idx, citizen in enumerate(citizens):
            if citizen_idx % 10 == 0:
                logger.info(f"  Step {step}: citizen {citizen_idx+1}/{len(citizens)} ...")

            # Skip anchors already processed in HYBRID Phase 1
            if config.mode == "HYBRID" and citizen.id in llm_sample_ids:
                continue

            prev_state = get_prev_state(citizen.id, step)
            X = build_feature_vector(citizen, prev_state, config.policy.domain)

            # Determine update method
            use_llm = False
            use_nn = False

            if config.mode == "LLM_ONLY":
                use_llm = citizen.id in llm_sample_ids
            elif config.mode == "HYBRID":
                # Non-anchor citizens: use NN if available, else rule-based
                if existing_model and hasattr(existing_model, 'predict'):
                    use_nn = True
            elif config.mode == "NN_ONLY":
                if existing_model and hasattr(existing_model, 'predict'):
                    use_nn = True

            if use_llm:
                try:
                    citizen_profile = citizen_to_dict(citizen)
                    current_state = state_to_dict(prev_state)
                    policy_dict = {
                        "title": config.policy.title,
                        "description": config.policy.description,
                        "domain": config.policy.domain
                    }
                    reaction = llm_client.generate_citizen_reaction(
                        citizen_profile, current_state, policy_dict,
                        knowledge_context=knowledge_context
                    )
                    new_happiness = reaction["new_happiness"]
                    new_policy_support = reaction["new_policy_support"]
                    income_delta = reaction["income_delta"]
                    new_income = max(0.0, prev_state.income + income_delta)
                    diary_entry = reaction.get("diary_entry")

                    Y = compute_deltas(prev_state, new_happiness, new_policy_support, new_income)
                    training_dataset.add_sample(X, Y)

                    new_state = CitizenState(
                        citizen_id=citizen.id, step=step,
                        happiness=new_happiness, policy_support=new_policy_support,
                        income=new_income, diary_entry=diary_entry, llm_updated=True,
                    )
                    step_llm_count += 1
                except Exception as e:
                    logger.error(f"LLM error for citizen {citizen.id}: {e}")
                    new_happiness, new_policy_support, new_income = rule_based_update(
                        citizen, prev_state, config.policy.domain, rng
                    )
                    new_state = CitizenState(
                        citizen_id=citizen.id, step=step,
                        happiness=new_happiness, policy_support=new_policy_support,
                        income=new_income, diary_entry=None, llm_updated=False,
                    )
                    step_rule_count += 1

            elif use_nn:
                try:
                    nn_start = time.time()

                    if feature_scaler:
                        X_scaled = feature_scaler.transform(X.reshape(1, -1))
                    else:
                        X_scaled = X.reshape(1, -1)

                    if existing_model is None:
                        raise ValueError("Neural network model is None - cannot predict")

                    deltas = existing_model.predict(X_scaled)[0]
                    if target_scaler is not None:
                        deltas = target_scaler.inverse_transform(deltas.reshape(1, -1))[0]
                    new_happiness, new_policy_support, new_income = apply_deltas(prev_state, deltas)

                    # Apply calibration bounds from anchor statistics (HYBRID mode)
                    if calibration_bounds:
                        new_happiness, new_policy_support, new_income = _calibrate_prediction(
                            new_happiness, new_policy_support, new_income,
                            citizen, calibration_bounds,
                        )

                    nn_time = time.time() - nn_start
                    nn_stats["nn_prediction_times"].append(nn_time)

                    new_state = CitizenState(
                        citizen_id=citizen.id, step=step,
                        happiness=new_happiness, policy_support=new_policy_support,
                        income=new_income, diary_entry=None, llm_updated=False,
                    )
                    step_nn_count += 1
                except Exception as e:
                    logger.error(f"NN error for citizen {citizen.id}: {e}, falling back to rule-based")
                    new_happiness, new_policy_support, new_income = rule_based_update(
                        citizen, prev_state, config.policy.domain, rng
                    )
                    new_state = CitizenState(
                        citizen_id=citizen.id, step=step,
                        happiness=new_happiness, policy_support=new_policy_support,
                        income=new_income, diary_entry=None, llm_updated=False,
                    )
                    step_rule_count += 1
            else:
                new_happiness, new_policy_support, new_income = rule_based_update(
                    citizen, prev_state, config.policy.domain, rng
                )
                new_state = CitizenState(
                    citizen_id=citizen.id, step=step,
                    happiness=new_happiness, policy_support=new_policy_support,
                    income=new_income, diary_entry=None, llm_updated=False,
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
