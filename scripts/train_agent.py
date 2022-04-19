from drlqap.taskgenerators import generators as tasks
from drlqap.experiment import run_experiment_from_config
import logging
import sys
from drlqap.agent_configs import agents, agent_training_steps
from argparse import ArgumentParser, RawTextHelpFormatter, Action
from pathlib import Path

def list_configs():
    return "\n".join([
        "Agent types:",
        *(f"- {agent}" for agent in agents.keys()),
        "\n",
        "Task generators:",
        *(f"- {task}" for task in tasks.keys()),
    ])

def main():
    parser = ArgumentParser(description="Train an agent on a given task", epilog=list_configs(), formatter_class=RawTextHelpFormatter)
    parser.add_argument('experiment_path', type=Path, help="Output folder")
    parser.add_argument('agent_name', type=str, help="Agent configuration name")
    parser.add_argument('task_name', type=str, help="Output folder")
    parser.add_argument('--seed', type=int, default=0, help="Seed for random number generator (default 0)")
    parser.add_argument('--steps', type=int, default=None, help="Override number of training steps")

    # Some generic tunable arguments to avoid too many configurations
    agent_args = dict()
    class AgentArgAction(Action):
        def __call__(self, parser, namespace, values, option_string):
            agent_args[self.dest] = values

    parser.add_argument('-lr', '--learning_rate', type=float, action=AgentArgAction, help='Override learning rate')
    parser.add_argument('-gd', '--gnn_depth', type=int, action=AgentArgAction, help='Override number of GNN layers')
    parser.add_argument('-md', '--mlp_depth', type=int, action=AgentArgAction, help='Override depth of MLP encoders')
    parser.add_argument('-hs', '--hidden_size', type=int, action=AgentArgAction, help='Override hidden size')
    parser.add_argument('-wd', '--weight_decay', type=float, action=AgentArgAction, help='Override weight decay')

    args = parser.parse_intermixed_args()

    if args.agent_name not in agents:
        print(f"Unknown agent type '{args.agent_name}'")
    if args.task_name not in tasks:
        print(f"Unknown task generator '{args.task_name}'")

    run_experiment_from_config(
        experiment_path=args.experiment_path,
        agent_name=args.agent_name,
        task_name=args.task_name,
        agent_arguments=agent_args,
        seed=args.seed,
        steps=args.steps
    )

    return 0

if __name__ == "__main__":
    sys.exit(main())
