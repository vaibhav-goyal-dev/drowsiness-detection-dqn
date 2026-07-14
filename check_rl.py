import torch, yaml, os
import numpy as np
import matplotlib.pyplot as plt
from rl_agent import DQNAgent

def generate_rl_report(config_path='config.yaml'):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    agent = DQNAgent(cfg['rl'])
    rl_path = cfg['paths']['global_rl']
    agent.load(rl_path)
    
    print("\n" + "="*52)
    print("  RL AGENT STATUS REPORT")
    print("="*52)
    print(f"  Training Steps:   {agent.steps}")
    print(f"  Current Epsilon:  {agent.eps:.4f}")
    print(f"  Buffer Size:      {len(agent.buf)}/{agent.buf.buf.maxlen}")
    print("="*52)
    
    # Simulate some scenarios
    scenarios = [
        # [p_smooth, ear, mar, perclos, consec_norm]
        ("Normal Eyes", [0.1, 0.35, 0.1, 0.0, 0.0]),
        ("Blinking (Short)", [0.6, 0.15, 0.1, 0.1, 0.2]),
        ("Heavy Drowsy", [0.9, 0.12, 0.1, 0.6, 0.9]),
        ("Yawning", [0.4, 0.32, 0.7, 0.1, 0.1]),
    ]
    
    print("\n  POLICY SIMULATION")
    print(f"  {'Scenario':<18} | {'CNN Prob':<10} | {'RL Decision':<12}")
    print("-" * 52)
    
    for name, state in scenarios:
        s_arr = np.array(state, dtype=np.float32)
        # select_action uses epsilon-greedy, let's use the raw Q-decision for report
        s_tensor = torch.FloatTensor(s_arr).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q = agent.online(s_tensor)
            action = int(q.argmax(dim=1).item())
        
        act_str = "DROWSY" if action == 1 else "ALERT"
        print(f"  {name:<18} | {state[0]:<10.2f} | {act_str}")
    print("-" * 52)

if __name__ == "__main__":
    generate_rl_report()
