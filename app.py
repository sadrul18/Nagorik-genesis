"""
NAGORIK-GENESIS: Synthetic Society Policy Simulator — Bangladesh Edition
Main Streamlit application entry point.

A hybrid AI system that combines LLM-based micro-simulations with
a neural network that learns from the LLM, enabling scalable
synthetic society simulations for Bangladeshi policy analysis.
"""
import streamlit as st
from pathlib import Path
import sys
import os
import pandas as pd
import joblib

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings
from population import generate_population
from llm_client import create_llm_client
from simulation import run_simulation
from nn_model import train_reaction_model, load_reaction_model
from ml_data import MLDataset
from web_knowledge import create_web_knowledge_client
from ui_sections import (
    render_sidebar_controls,
    render_learning_status_panel,
    render_overview_tab,
    render_groups_tab,
    render_citizens_tab,
    render_experts_tab,
    render_scenarios_tab,
    render_nn_analytics_tab
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NAGORIK-GENESIS",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #006a4e 0%, #f42a41 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = {}

    if "current_population" not in st.session_state:
        st.session_state.current_population = None

    if "nn_model" not in st.session_state:
        try:
            model = load_reaction_model()
            if model and model.is_trained:
                st.session_state.nn_model = model
                st.session_state.feature_scaler = getattr(model, 'scaler', None)
                # Load target scaler for inverse-transforming NN predictions
                target_scaler_path = Path("models/target_scaler.joblib")
                if target_scaler_path.exists():
                    st.session_state.target_scaler = joblib.load(target_scaler_path)
                    logger.info("✅ Loaded target scaler for NN prediction inverse transform")
                else:
                    st.session_state.target_scaler = None
                    logger.warning("⚠️ target_scaler.joblib not found — NN predictions will be in scaled space")
                logger.info(f"✅ Loaded existing NN model (trained: {model.is_trained})")
            else:
                st.session_state.nn_model = None
                st.session_state.feature_scaler = None
                st.session_state.target_scaler = None
        except Exception as e:
            logger.info(f"No existing NN model found: {e}")
            st.session_state.nn_model = None

    if "feature_scaler" not in st.session_state:
        st.session_state.feature_scaler = None

    if "target_scaler" not in st.session_state:
        st.session_state.target_scaler = None

    if "total_training_samples" not in st.session_state:
        st.session_state.total_training_samples = 0

    if "training_dataset" not in st.session_state:
        st.session_state.training_dataset = MLDataset()

    if "current_scenario" not in st.session_state:
        st.session_state.current_scenario = None

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None

    if "summary_client" not in st.session_state:
        st.session_state.summary_client = None


def run_simulation_sync(config, citizens, llm_client, nn_model, feature_scaler, target_scaler=None, knowledge_context=""):
    """Run simulation synchronously."""
    return run_simulation(config, citizens, llm_client, nn_model, feature_scaler, target_scaler, knowledge_context=knowledge_context)


def main():
    """Main application function."""
    initialize_session_state()

    # Title and description
    st.markdown('<h1 class="main-header">🇧🇩 NAGORIK-GENESIS</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Bangladesh Policy Simulator — Synthetic Society with Hybrid AI</p>',
        unsafe_allow_html=True
    )

    # Responsible AI disclaimer
    st.warning("""
    ⚠️ **Responsible AI Disclaimer**

    This is a synthetic simulation for exploratory purposes only. It does NOT predict
    real-world behavior. This tool is designed to explore possible scenarios and highlight
    potential blind spots — not to replace real data, surveys, or expert analysis.

    Results should be interpreted as thought experiments, not predictive models.
    The neural network learns from LLM outputs and approximates reactions for scalability.

    All income and financial values are in Bangladeshi Taka (BDT).
    """)

    # Initialize LLM client
    try:
        settings = get_settings()
        if st.session_state.llm_client is None:
            st.session_state.llm_client = create_llm_client(
                backend=settings.llm_backend,
                ollama_model=settings.ollama_model,
                ollama_host=settings.ollama_host,
                gemini_api_key=settings.gemini_api_key,
                backup_keys=settings.backup_api_keys
            )
            if settings.llm_backend == "ollama":
                st.success(f"🦙 Ollama backend — model: `{settings.ollama_model}` (local, unlimited)")
            else:
                key_count = len(settings.backup_api_keys or []) + 1
                st.success(f"🔑 Gemini backend — {key_count} API keys loaded")

        if st.session_state.summary_client is None:
            if settings.llm_backend == "gemini" and settings.summary_api_key:
                st.session_state.summary_client = create_llm_client(
                    backend="gemini",
                    gemini_api_key=settings.summary_api_key
                )
            else:
                st.session_state.summary_client = st.session_state.llm_client

        # Initialize Web Knowledge client
        if "web_knowledge_client" not in st.session_state:
            st.session_state.web_knowledge_client = create_web_knowledge_client(
                tavily_api_key=settings.tavily_api_key,
                ollama_model=settings.ollama_model,
                ollama_host=settings.ollama_host,
            )
            if st.session_state.web_knowledge_client.is_available():
                backend_name = "Tavily" if settings.tavily_api_key else "DuckDuckGo"
                st.success(f"🌐 Web knowledge — {backend_name} search active")

    except Exception as e:
        st.error(f"❌ **Configuration Error**: {str(e)}")
        st.stop()

    # Sidebar controls
    config = render_sidebar_controls()

    # Generate population button
    if st.sidebar.button("👥 Generate New Population", use_container_width=True):
        with st.spinner("Generating population..."):
            try:
                settings = get_settings()
                population = generate_population(
                    size=1000,
                    seed=settings.random_seed
                )
                st.session_state.current_population = population
                st.sidebar.success(f"Generated {len(population)} citizens!")
                logger.info(f"Generated population of {len(population)} citizens")
            except Exception as e:
                st.sidebar.error(f"Error generating population: {e}")
                logger.error(f"Population generation error: {e}")

    if st.session_state.current_population:
        st.sidebar.info(f"Current population: {len(st.session_state.current_population)} citizens")

    # Train NN button
    if st.sidebar.button("🧠 Train Neural Network", use_container_width=True):
        if st.session_state.total_training_samples < 500:
            st.sidebar.warning(f"Need at least 500 samples. Current: {st.session_state.total_training_samples}")
        else:
            with st.spinner("Training neural network..."):
                try:
                    X, Y = st.session_state.training_dataset.get_arrays()
                    model, metrics = train_reaction_model(X, Y)

                    model.save(
                        "models/citizen_reaction_mlp.joblib",
                        "models/feature_scaler.joblib"
                    )

                    st.session_state.nn_model = model
                    st.session_state.feature_scaler = model.scaler

                    st.sidebar.success(f"""
                    ✅ Model trained successfully!
                    - Training samples: {metrics['n_samples']}
                    - Train MAE: {metrics['train_mae']:.4f}
                    - Val MAE: {metrics.get('val_mae', 'N/A')}
                    """)
                    logger.info(f"NN model trained with {metrics['n_samples']} samples")
                except Exception as e:
                    st.sidebar.error(f"Training error: {e}")
                    logger.error(f"NN training error: {e}")

    # Learning status panel
    render_learning_status_panel(
        num_samples=st.session_state.total_training_samples,
        model_trained=st.session_state.nn_model is not None and st.session_state.nn_model.is_trained,
        current_mode=config.mode if config else "N/A"
    )

    # Run simulation if config is provided
    if config:
        if st.session_state.current_population is None:
            with st.spinner("Generating population..."):
                try:
                    population = generate_population(
                        size=config.population_size,
                        seed=config.random_seed
                    )
                    st.session_state.current_population = population
                except Exception as e:
                    st.error(f"Error generating population: {e}")
                    return
        else:
            if len(st.session_state.current_population) != config.population_size:
                with st.spinner("Regenerating population to match requested size..."):
                    population = generate_population(
                        size=config.population_size,
                        seed=config.random_seed
                    )
                    st.session_state.current_population = population

        # Search web for policy context (before simulation)
        knowledge_context = ""
        wk_client = st.session_state.get("web_knowledge_client")
        if wk_client and wk_client.is_available() and config.mode != "NN_ONLY":
            with st.spinner("🌐 Searching web for policy context..."):
                try:
                    knowledge_context = wk_client.search_policy_context(
                        policy_title=config.policy.title,
                        policy_description=config.policy.description,
                        policy_domain=config.policy.domain,
                    )
                    if knowledge_context:
                        st.info(f"🌐 Web knowledge loaded ({len(knowledge_context)} chars)")
                except Exception as e:
                    st.warning(f"⚠️ Web knowledge search failed (continuing without): {e}")

        with st.spinner(f"Running simulation '{config.name}'..."):
            try:
                results = run_simulation_sync(
                    config,
                    st.session_state.current_population,
                    st.session_state.llm_client,
                    st.session_state.nn_model,
                    st.session_state.feature_scaler,
                    st.session_state.target_scaler,
                    knowledge_context=knowledge_context
                )

                st.session_state.scenarios[config.name] = results
                st.session_state.current_scenario = config.name

                new_samples = results["training_dataset"]
                st.session_state.training_dataset.merge(new_samples)
                st.session_state.total_training_samples = st.session_state.training_dataset.size()

                if st.session_state.total_training_samples > 0:
                    st.session_state.training_dataset.save_to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "llm_training_samples.csv"))

                st.success(f"""
                ✅ Simulation complete!
                - Collected {new_samples.size()} new training samples
                - Total training samples: {st.session_state.total_training_samples}
                """)
                logger.info(f"Simulation '{config.name}' completed with {new_samples.size()} samples")

                # Generate expert summary for final step
                if results["step_stats"]:
                    final_stats = results["step_stats"][-1]
                    with st.spinner("Generating expert summary..."):
                        try:
                            summary = st.session_state.summary_client.generate_expert_summary(
                                {
                                    "step": final_stats.step,
                                    "avg_happiness": final_stats.avg_happiness,
                                    "avg_support": final_stats.avg_support,
                                    "avg_income": final_stats.avg_income,
                                    "by_income": final_stats.by_income,
                                    "by_division": final_stats.by_division,
                                    "by_religion": final_stats.by_religion,
                                    "inequality_gap_happiness": final_stats.inequality_gap_happiness
                                },
                                {
                                    "title": config.policy.title,
                                    "description": config.policy.description,
                                    "domain": config.policy.domain
                                }
                            )
                            results["expert_summary"] = summary
                            logger.info("✅ Expert summaries generated successfully")
                            st.success("✅ Expert summaries generated!")
                        except Exception as e:
                            error_msg = f"Error generating expert summary: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            st.warning(f"⚠️ {error_msg}")
                            results["expert_summary"] = None

            except Exception as e:
                st.error(f"Simulation error: {e}")
                logger.error(f"Simulation error: {e}", exc_info=True)
                return

    # Display current scenario results
    if st.session_state.current_scenario and st.session_state.current_scenario in st.session_state.scenarios:
        current_results = st.session_state.scenarios[st.session_state.current_scenario]

        max_step = current_results["config"].steps
        selected_step = st.slider("Select Step", 0, max_step, max_step)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Summary",
            "👥 Groups",
            "👤 Citizens",
            "🎓 Experts",
            "🧠 NN Analytics",
            "📂 Scenarios"
        ])

        with tab1:
            render_overview_tab(current_results["step_stats"], selected_step)

        with tab2:
            render_groups_tab(current_results["step_stats"], selected_step)

        with tab3:
            render_citizens_tab(
                current_results["citizens"],
                current_results["all_states"],
                selected_step
            )

        with tab4:
            render_experts_tab(current_results.get("expert_summary"))

        with tab5:
            nn_stats = current_results.get("nn_stats", {})
            training_samples = len(pd.read_csv("data/llm_training_samples.csv")) if os.path.exists("data/llm_training_samples.csv") else 0
            mode = current_results["config"].mode if hasattr(current_results["config"], "mode") else "HYBRID"
            render_nn_analytics_tab(st.session_state.nn_model, nn_stats, training_samples, mode)

        with tab6:
            render_scenarios_tab(st.session_state.scenarios)

    else:
        st.info("""
        👋 **Welcome to NAGORIK-GENESIS!**

        To get started:
        1. Select a policy preset from the sidebar or create your own
        2. Choose a simulation mode (LLM_ONLY, HYBRID, or NN_ONLY)
        3. Click the "🚀 Run Simulation" button

        **Hybrid AI Architecture:**
        - Start with LLM_ONLY mode to collect training data
        - Once you have 500+ samples, train the neural network
        - Switch to HYBRID or NN_ONLY mode for faster, scalable simulations

        **🇧🇩 Bangladesh Policy Presets Available:**
        Fuel Subsidy Removal, RMG Wage Hike, Digital BD Scholarship, Post-Flood Relief,
        Metro Rail Expansion, Slum Eviction, Universal Health Card, Freelancing Tax Incentive
        """)


if __name__ == "__main__":
    main()
