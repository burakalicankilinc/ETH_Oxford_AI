import numpy as np
from mpc import MPCController
from predictor import SimpleGBMPredictor

def run_system_test():
    """
    Validates that the MPC logic can process data and make a 
    decision based on a provided trend signal.
    """
    print("--- STARTING MPC SYSTEM TEST ---")

    # 1. Mock historical data (Simulating 7 days of stock prices)
    # This mimics the output of market_data_tool in the agent system
    history = np.array([150.0, 152.5, 151.2, 153.8, 155.0, 154.5, 156.2])
    current_price = history[-1]
    
    # 2. Setup the MPC Components
    # SimpleGBMPredictor uses gbm.py logic internally
    predictor = SimpleGBMPredictor()
    
    # Define an action grid: Sell 10, Hold, or Buy 10 shares
    controller = MPCController(
        predictor=predictor, 
        action_grid=[-10, 0, 10],
        transaction_cost=0.001 # 0.1% fee
    )
    
    # 3. Test Scenario A: Use historical drift (No agent override)
    decision_hist = controller.select_action(current_price, 0, history)
    
    # 4. Test Scenario B: Use Agent-provided trend (Mu = 0.10 override)
    # This mimics the Trend Agent telling the MPC 'I expect 10% growth'
    agent_mu = 0.10
    decision_agent = controller.select_action(
        current_price, 0, history, mu_override=agent_mu
    )
    
    print(f"\n[Scenario A: Historical Baseline]")
    print(f"Calculated Mu: {decision_hist['mu']:.4f}")
    print(f"Optimal Action: {decision_hist['action']} shares")

    print(f"\n[Scenario B: Agent-Influenced]")
    print(f"Agent Signal (Mu): {decision_agent['mu']}")
    print(f"Optimal Action: {decision_agent['action']} shares")
    print(f"Expected Target Price: ${decision_agent['expected_final_price']:.2f}")

    print("\n--- TEST COMPLETE: ALL COMPONENTS SYNCED ---")

if __name__ == "__main__":
    run_system_test()