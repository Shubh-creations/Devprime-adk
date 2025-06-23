benchmark_adk/
├── agent/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   └── model.py
│
├── environment/
│   ├── __init__.py
│   ├── gridworld_env.py
│   ├── robot_env.py
│   └── wrappers.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── visualizer.py
│   └── replay_buffer.py
│
├── config/
│   ├── gridworld_config.yaml
│   ├── robot_config.yaml
│   └── train_config.yaml
│
├── experiments/
│   ├── gridworld_benchmark.py
│   └── robot_benchmark.py
│
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
└── STORY.md

- Python – Core language for development
- PyTorch – Deep learning framework for agent models
- OpenAI Gym – Standardized interface for reinforcement learning environments
- NumPy & Pandas – Numerical processing and data analysis
- Matplotlib & Seaborn – Visualization of training metrics and performance
- YAML – Configurable experiment settings
- GitHub Actions – CI/CD workflows for testing and validation
- Jupyter Notebooks (planned) – Interactive analysis and visualization
- (Optional Extensions)
- Webots / Isaac Sim / PyBullet – For real-time robotic simulations
- Hydra / Argparse – For scalable CLI-based configuration
- TensorBoard – Model training logs and comparison dashboards


