from typing import Any, Literal, List, Tuple
from constants import TASKS, train_params
from train import Train

class Sweep:

    def __init__(
        self,
        network: Literal["RNN, LSTM"],
        full_plasticity_tasks: List[List[Any]],
        sparse_plasticity_tasks: List[List[Any]],
    ):
        trainer = Train(network=network)
        trainer.define_optimizer(sparse_plasticity=False)

        print("****** TRAINING USING FULL PLASTICITY ******")
        trainer.create_task_loaders(full_plasticity_tasks)
        trainer.train_model(train_params["n_full_plasticity_epochs"])

        for i, tasks in enumerate(sparse_plasticity_tasks[:1]):
            print("XXX", tasks)
            print(f"****** TRAINING USING SPARSE PLASTICITY, ROUND: {i} ******")
            trainer.define_optimizer(sparse_plasticity=True)
            trainer.create_task_loaders(tasks)
            trainer.train_model(train_params["n_sparse_plasticity_epochs"])

def generate_task_variants(task: str, target_offset: Tuple[int], dir_offset: Tuple[int]):
    task_variants = []
    for to in target_offset:
        if "delayed_match" in task:
            for do in dir_offset:
                task_variants.append([task, do, to])
        else:
            task_variants.append([task, to])
    return task_variants

def create_task_set(
    task_family: List,
    target_offset: Tuple[int]=[0, 1, 2, 3, 4, 5, 6, 7],
    dir_offset: Tuple[int]=[0, 1, 2],
):
    task_set = []
    for task in task_family:
        task_set += generate_task_variants(task, target_offset, dir_offset)

    return task_set

def split_train_set(tasks0: List, tasks1: List, train_pct: float = 0.9):
    """NOT IN USE"""

    available_tasks = get_available_tasks()
    train_set_0 = []

    if train_pct is not None:
        n = int(train_pct * len(available_tasks))
        np.random.shuffle(available_tasks)
        idx = np.arange(len(available_tasks))

        idx0, idx1 = idx[:n], idx[n:]
        train_set_0 = [available_tasks[i] for i in idx0]
        train_set_1 = [available_tasks[i] for i in idx1]
    else:
        train_set_0 = [t for t in available_tasks if holdout_task not in t[0]]
        train_set_1 = [t for t in available_tasks if holdout_task in t[0]]

    return train_set_0, train_set_1


# full plasticity tasks
full_plasticity_tasks = create_task_set(TASKS[:5])
sparse_plasticity_tasks = [create_task_set([t]) for t in TASKS[5:]]

sweep = Sweep(
    network="LSTM",
    full_plasticity_tasks=full_plasticity_tasks,
    sparse_plasticity_tasks=sparse_plasticity_tasks,
)