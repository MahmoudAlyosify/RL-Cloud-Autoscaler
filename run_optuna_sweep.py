import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_factory import make_vec_env, make_env
import argparse

def objective(trial):
    lr       = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    n_steps  = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    
    # Reward weights tuning
    beta    = trial.suggest_float("beta",    0.05, 0.5)   # latency weight
    w_drop  = trial.suggest_float("w_drop",  10.0, 100.0) # drop-penalty weight
    delta   = trial.suggest_float("delta",   1.0, 10.0)   # thrash weight

    # Make vectorized environment with suggested reward weights
    # Note: custom reward weights might require an override in env_factory or cloud_env
    # Here we assume make_vec_env can pass it through env_kwargs
    train_env = make_vec_env(
        n_envs=4, 
        seed=42, 
        use_subprocess=True, 
        norm_reward=True,
        reward_weights=(1.0, beta, w_drop, delta)
    )

    model = PPO("MlpPolicy", train_env, learning_rate=lr, n_steps=n_steps,
                ent_coef=ent_coef, gamma=0.99, verbose=0)
    
    # Short trial budget for Optuna sweep
    model.learn(total_timesteps=20_000)
    
    # Evaluate model
    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=0, reward_weights=(1.0, beta, w_drop, delta))]),
        norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99
    )
    
    # Sync normalization stats
    eval_env.obs_rms = train_env.obs_rms
    eval_env.training = False

    rewards = []
    for ep in range(5):
        obs = eval_env.reset()
        done = [False]
        R = 0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = eval_env.step(action)
            R += r[0]
        rewards.append(R)
        
    eval_env.close()
    train_env.close()
    
    return float(np.mean(rewards))

def main():
    parser = argparse.ArgumentParser(description="Run Optuna Sweep for PPO Hyperparameters")
    parser.add_argument("--trials", type=int, default=50, help="Number of optuna trials")
    args = parser.parse_args()

    print(f"Starting Optuna sweep with {args.trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    
    print("\n" + "="*60)
    print("Optimization finished!")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Mean Reward): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*60)

if __name__ == "__main__":
    main()
