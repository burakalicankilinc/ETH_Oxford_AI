# EFFECTS.md

## 1. I/O Boundary (Effect Inventory)
The system is authorized to interact with the following external resources and non-deterministic factors:

* **Network (External APIs):**
    * **Yahoo Finance (`yfinance`):** Fetches historical price data and ticker history.
    * **Valyu API:** Conducts web searches for news and proprietary research data using an external service.
    * **Google Generative AI:** Interfaces with the `gemini-2.5-flash` model for reasoning and synthesis.
* **Storage (File System):**
    * **Image Generation:** Writes forecast plots to the local disk as `.png` files via Matplotlib.
* **Entropy (Randomness):**
    * **Stochastic Modeling:** Uses `np.random.normal` to generate paths for Monte Carlo simulations in the Brownian model.
    * **LLM Sampling:** Operates with a `temperature` of `1.0`, resulting in non-deterministic text outputs.
* **Environment:**
    * **Secrets Management:** Retrieves the `VALYU_API_KEY` from system environment variables.
* **Observability:**
    * **Console Output:** Logs execution states, trend directions, and final investment memos to `stdout`.

---

## 2. Effect Definitions (Code References)
The following code symbols define the boundaries and provide the capabilities:

* **Tool Definitions:**
    * `mlModel`: Entrypoint for Prophet-based time-series forecasting and `.png` file output.
    * `brownianModel`: Entrypoint for Geometric Brownian Motion simulations and `yfinance` data retrieval.
    * `generalInfo` / `specificInfo`: Entrypoints for external research via the `Valyu` client.
* **Agent Interfaces:**
    * `trend_agent`: A specialized agent wrapping quantitative tools (`mlModel`, `brownianModel`).
    * `noise_agent`: A specialized agent wrapping search and sentiment tools.
* **State Management:**
    * `AgentState`: A `TypedDict` that tracks the user `query` and aggregates a list of `results` strings.

---

## 3. Pure Core (Business Logic)
The deterministic internal logic of the application manages how information is transformed and routed.

* **Information Flow:**
    1.  **Routing:** The system uses `route_query` to parse the user request into a `RouteSchema`, determining which specialized agents to "hire".
    2.  **Orchestration:** The graph executes `trend_node` and `noise_node` (quant vs. research) based on the routing decision.
    3.  **Synthesis:** The `aggregator` node merges independent analysis strings into a final investment memo using the LLM.
* **Main Execution Paths:**
    * **Quant Path:** Historical Data → Statistical Simulation/Prophet Training → Price Prediction.
    * **Research Path:** News Search → Snippet Extraction → Sentiment Summary.

---

## 4. Runtime (What you used)
* **Language:** Python 3.
* **Effect Library/Runtime:** **LangGraph** (StateGraph) for workflow orchestration and **LangChain** for LLM/Tool abstraction.
* **Runtime Style:**
    * **Graph-Based:** The application is compiled into a `StateGraph` and executed via `app.stream()`.
    * **Dynamic Dispatch:** Uses `conditional_edges` at the `START` node to determine the execution path at runtime based on the input query.
