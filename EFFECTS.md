# EFFECTS.md

## 1. I/O Boundary

* **Network:**
    * Extracts historical ticker price data.
    * Uses an AI specialised in searching news and articles to gather information about the stock.  to conduct web searches for news and research on the stock, by considering macroeconomic trends and industry-specific information. 
    * Uses another reasoning model for data analysis and final prediction.
* **Storage:**
    * One of the tools plots graphs after predicting the stock prices, saving them to the local disk as `.png` files for further analysis.
* **Entropy**
    * Uses numpy's normal distribution model, `np.random.normal`, to generate paths for Monte Carlo simulations.
    * The reasoning model operates with a medium `temperature` of `1.0`, resulting in higher non-deterministic text outputs compared to lower values.


---

## 2. Effect Definitions

* **Network:**
    * Yahoo Finance (`yfinance`) for fetching price data.
    * ValyU AI to conduct web searches for news and research on the stock, categorising information into macroeconomic and industry-specific areas. 
    * The `gpt-4.1` model is used for reasoning and analysis of the stock, given the necessary tools/information and providing the final answer & prediction.
* **Storage:**
    * The `@tool` mlModel uses Facebook Prophet to forecast prices, and also saves the plotted graphs to the local disk as `.png` files.
* **Entropy**
    * `np.random.normal` is used in the mlModel to generate random paths.
    * The `gpt-4.1` model operates with a `temperature` of `1.0`, a medium value which corresponds to non-deterministic text outputs.

---

## 3. Pure Core

* **Information Flow:**
    1.  **Routing:** The system is initialised by using `route_query` to transform user requests into `RouteSchema`, in which it determines which specialised agents to "hire".
    2.  **Orchestration:** Depending on the routing decision, the graph would execute the corresponding `trend_node` and `noise_node` (quant vs. research), each with a set of previously defined tools. 
    3.  **Synthesis:** The `aggregator` node merges independent analysis strings into a final investment memo using the LLM. 
* **Main Execution Paths:**
    * **Quant Path:** Historical Data → Statistical Simulation/Prophet Training → Price Prediction.
    * **Research Path:** News/Research Extraction → URL/Snippet Extraction → Sentiment Summary.

---

## 4. Runtime
* **Language:** Python 3.
* **Effect Library/Runtime:** **LangGraph** (StateGraph) for workflow orchestration and **LangChain** for LLM/Tool abstraction.
* **Runtime Style:**
    * The application is compiled into a `StateGraph` and launched via `app.stream()`.
    * Uses `conditional_edges` at the `START` node to determine the execution path at runtime based on the input query.

