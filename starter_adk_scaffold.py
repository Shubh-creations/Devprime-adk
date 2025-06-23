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

# STORY.md
## Inspiration

Our inspiration came from a gap we noticed in lightweight yet robust reinforcement learning (RL) toolkits that balance educational clarity with research-grade capabilities. We wanted a framework that could scale from simple GridWorld simulations to robotic environments, all while supporting benchmarking and reproducibility.

## What it does

Benchmark-ADK allows researchers and developers to train, evaluate, and benchmark RL agents using modular, reproducible tools. It supports DQN and PPO agents, gym-compatible environments, experiment scripts, and a training pipeline compatible with real-time and simulated systems.

## How we built it

We started by sketching out the desired structure: agents, environments, utilities, and config layers. We implemented modular agents using PyTorch, environment interfaces using OpenAI Gym conventions, and YAML-based configuration loading. For benchmarking, we added evaluation scripts, plotting utilities, and GitHub Actions-based CI/CD.

## Challenges we ran into

- Making the environment API generic enough to support both grid simulations and robotic agents
- Deciding how to abstract agent models so users could extend easily
- Balancing simplicity with extensibility for new algorithms like PPO
- Keeping the training pipeline modular while still being intuitive

## Accomplishments that we're proud of

- Designing a clean, extensible codebase
- Making it runnable on both local laptops and cloud instances
- Implementing two working agent architectures
- Setting up automated testing with GitHub Actions

## What we learned

- The value of strict modular design in RL pipelines
- How to implement custom gym environments and wrappers
- Best practices for reproducible experiments
- Managing trade-offs between flexibility and clarity

## What's next for Devprime

- Add support for more algorithms like A3C and SAC
- Integrate real robot simulators (Webots, PyBullet, Isaac Sim)
- Include Jupyter dashboards for real-time metrics
- Host benchmark results online and enable plug-and-play experiment uploads

## Built With

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

## Try It Out / See the Code

- 🔗 **GitHub Repository**: [github.com/Shubh-creations/benchmark-adk](https://github.com/Shubh-creations/benchmark-adk)
- ▶️ **Try on Google Colab**: [Run a demo notebook](https://colab.research.google.com/github/Shubh-creations/benchmark-adk/blob/main/notebooks/gridworld_demo.ipynb)
- 🧪 **Run Benchmark Locally**:
  ```bash
  git clone https://github.com/Shubh-creations/benchmark-adk.git
  cd benchmark-adk
  pip install -r requirements.txt
  python train.py --config config/gridworld_config.yaml
  ```
