# Benchmark-ADK: Reinforcement Learning Toolkit
# A modular framework for training and benchmarking RL agents

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Request

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from agent.dqn_agent import DQNAgent
from agent.ppo_agent import PPOAgent
from environment.gridworld_env import GridWorldEnv
from environment.robot_env import RobotEnv
from utils.logger import Logger
from utils.visualizer import Visualizer
from utils.replay_buffer import ReplayBuffer

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run-agent")
async def run_agent(request: Request):
    data = await request.json()
    # Placeholder for your agent logic
    return {"result": "Agent ran successfully", "input": data}

class BenchmarkADK:
    """
    Main class for the Benchmark-ADK reinforcement learning toolkit.
    Provides a unified interface for training and evaluating RL agents.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the Benchmark-ADK framework.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = Logger(self.config.get('logging', {}))
        self.visualizer = Visualizer(self.config.get('visualization', {}))
        self.replay_buffer = ReplayBuffer(self.config.get('replay_buffer', {}))
        
        # Initialize environment and agent
        self.env = self._create_environment()
        self.agent = self._create_agent()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_environment(self):
        """Create environment based on configuration."""
        env_type = self.config['environment']['type']
        
        if env_type == 'gridworld':
            return GridWorldEnv(**self.config['environment']['params'])
        elif env_type == 'robot':
            return RobotEnv(**self.config['environment']['params'])
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
    
    def _create_agent(self):
        """Create agent based on configuration."""
        agent_type = self.config['agent']['type']
        
        if agent_type == 'dqn':
            return DQNAgent(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.n,
                **self.config['agent']['params']
            )
        elif agent_type == 'ppo':
            return PPOAgent(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.n,
                **self.config['agent']['params']
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    def train(self, num_episodes: int = 1000):
        """
        Train the agent for specified number of episodes.
        
        Args:
            num_episodes (int): Number of training episodes
        """
        self.logger.info(f"Starting training for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train agent if enough samples
                if len(self.replay_buffer) > self.config['training']['batch_size']:
                    batch = self.replay_buffer.sample(self.config['training']['batch_size'])
                    self.agent.train(batch)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                self.logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.2f}")
                
                # Visualize training progress
                self.visualizer.plot_training_progress(episode_rewards, episode_lengths)
        
        self.logger.info("Training completed")
        return episode_rewards, episode_lengths
    
    def evaluate(self, num_episodes: int = 100):
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes (int): Number of evaluation episodes
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info(f"Starting evaluation for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.agent.select_action(state, epsilon=0.0)  # No exploration
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    if reward > 0:  # Assuming positive reward indicates success
                        success_rate += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        success_rate /= num_episodes
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': success_rate,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        self.logger.info(f"Evaluation completed: Mean Reward = {metrics['mean_reward']:.2f}, Success Rate = {success_rate:.2f}")
        return metrics
    
    def benchmark(self, num_runs: int = 5):
        """
        Run multiple training and evaluation cycles for benchmarking.
        
        Args:
            num_runs (int): Number of benchmark runs
            
        Returns:
            dict: Benchmark results
        """
        self.logger.info(f"Starting benchmark with {num_runs} runs")
        
        benchmark_results = []
        
        for run in range(num_runs):
            self.logger.info(f"Benchmark run {run + 1}/{num_runs}")
            
            # Train agent
            train_rewards, train_lengths = self.train(self.config['training']['episodes'])
            
            # Evaluate agent
            eval_metrics = self.evaluate(self.config['evaluation']['episodes'])
            
            # Store results
            run_results = {
                'run': run + 1,
                'train_rewards': train_rewards,
                'train_lengths': train_lengths,
                'eval_metrics': eval_metrics
            }
            benchmark_results.append(run_results)
        
        # Aggregate results
        final_metrics = self._aggregate_benchmark_results(benchmark_results)
        
        # Visualize benchmark results
        self.visualizer.plot_benchmark_results(benchmark_results, final_metrics)
        
        self.logger.info("Benchmark completed")
        return benchmark_results, final_metrics
    
    def _aggregate_benchmark_results(self, benchmark_results: list) -> dict:
        """Aggregate results from multiple benchmark runs."""
        all_eval_rewards = []
        all_success_rates = []
        
        for result in benchmark_results:
            all_eval_rewards.extend(result['eval_metrics']['episode_rewards'])
            all_success_rates.append(result['eval_metrics']['success_rate'])
        
        return {
            'mean_final_reward': np.mean(all_eval_rewards),
            'std_final_reward': np.std(all_eval_rewards),
            'mean_success_rate': np.mean(all_success_rates),
            'std_success_rate': np.std(all_success_rates),
            'num_runs': len(benchmark_results)
        }
    
    def save_model(self, path: str):
        """Save the trained agent model."""
        self.agent.save_model(path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained agent model."""
        self.agent.load_model(path)
        self.logger.info(f"Model loaded from {path}")


def main():
    """Main function to run the Benchmark-ADK framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark-ADK: RL Training and Benchmarking')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'benchmark'], 
                       default='train', help='Mode to run')
    parser.add_argument('--episodes', type=int, help='Number of episodes (overrides config)')
    parser.add_argument('--runs', type=int, help='Number of benchmark runs (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize framework
    if __name__ == "__main__":
        agent = DQNAgent()
        action = agent.act(None)
        print(f"Agent action: {action}")

