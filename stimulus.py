import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Literal, Mapping, Optional, Union, Tuple
import attr
from torch.utils.data import Dataset

class TaskDataset(Dataset):

    def __init__(
            self,
            tasks: List,
            n_batches: int,
            n_context: Optional[int] = 200,  # fix this at maximum number of tasks
            RL: bool = False,
    ):
        self.n_context = n_total_tasks if n_context is None else n_context
        self.stim = Stimulus(task_set=tasks, n_context=n_context, RL=RL)
        self.n_batches = n_batches
        self.RL = RL

    def __len__(self) -> int:
        return 1024 * self.n_batches

    def get_stim_properties(self) -> Dict[str, Any]:
        return attr.asdict(self.stim)

    def __getitem__(self, idx) -> Mapping[str, torch.Tensor]:
        numpy_batch = self.stim.generate_trial()
        batch = {}

        for k, v in numpy_batch.items():
            if v is not None:
                batch[k] = torch.from_numpy(np.array(v))

        return batch


@attr.s(auto_attribs=True)
class Stimulus:

    task_set: Optional[List] = attr.ib(default=None)
    n_context: int = attr.ib(default=1)
    n_motion_tuned: int = attr.ib(default=36)
    n_motion_dirs: int = attr.ib(default=8)
    n_fix_tuned: int = attr.ib(default=1)
    n_time_steps: int = attr.ib(default=100)
    dt: int = attr.ib(default=20)
    dead_time: int = attr.ib(default=100)
    mask_length: int = attr.ib(default=20)
    fix_time: int = attr.ib(default=400)
    delay_times: List[int] = attr.ib(default=[400, 600, 800])
    contrast_set: List[int] = attr.ib(default=[-0.4, -0.2, -0.1, 0.1, 0.2, 0.4])
    tuning_height: float = attr.ib(default=2)
    input_noise: float = attr.ib(default=0.1)
    RL: bool = attr.ib(default=False)

    motion_dirs: np.ndarray = attr.ib(init=False)
    # stimulus_dirs: np.ndarray = attr.ib(init=False)
    pref_motion_dirs: np.ndarray = attr.ib(init=False)
    n_inputs_per_rf: int = attr.ib(init=False)
    tasks: List = attr.ib(init=False)
    n_output: int = attr.ib(init=False)
    n_stimulus: int = attr.ib(init=False)

    def __attrs_post_init__(self):

        self.motion_dirs = np.linspace(
            0, 2 * np.pi - 2 * np.pi / self.n_motion_dirs, self.n_motion_dirs
        )
        self.pref_motion_dirs = np.linspace(
            0, 2 * np.pi - 2 * np.pi / (self.n_motion_tuned // 2), (self.n_motion_tuned // 2)
        )[None, :]
        self.n_inputs_per_rf = self.n_motion_tuned // 2
        self.tasks = self.define_tasks()
        self.n_stimulus = self.n_motion_tuned + self.n_fix_tuned
        self.n_output = self.n_motion_dirs + 1  # outputs are all motion directions plus fixation

    def define_tasks(self) -> List[List[Any]]:

        tasks = []
        if self.task_set is not None:
            for task_params in self.task_set:
                if "task_go" in task_params[0]:
                    tasks.append([self.task_go, *task_params])
                elif "decision_making" in task_params[0]:
                    tasks.append([self.decision_making, *task_params])
                elif "delayed_match" in task_params[0]:
                    tasks.append([self.delayed_match, *task_params])
                else:
                    raise Exception('Bad task variant.')
        else:
            for variant in ["task_go"]:
                tasks.append([self.task_go, variant, 0])

        return tasks

    def circ_tuning(self, motion_dir_idx: int, rf: int) -> np.ndarray:

        assert rf in [0, 1], "rf must be 0 or 1"
        theta = (motion_dir_idx / self.n_motion_dirs) * 2 * np.pi
        ang_dist = np.angle(np.exp(1j * theta - 1j * self.pref_motion_dirs))
        resp = self.tuning_height * np.exp(-0.5 * (8 * ang_dist / np.pi) ** 2)
        zero_resp = np.zeros((1, self.n_motion_tuned // 2), dtype=np.float32)
        if rf == 0:
            return np.concatenate((resp, zero_resp), axis=-1)
        elif rf == 1:
            return np.concatenate((zero_resp, resp), axis=-1)


    def reset_trial_info(self) -> None:

        self.trial_info = {
            "stim_input": np.zeros((self.n_time_steps, self.n_motion_tuned), dtype=np.float32),
            "fix_input": np.zeros((self.n_time_steps, self.n_fix_tuned), dtype=np.float32),
            "context_input": np.zeros((self.n_time_steps, self.n_context), dtype=np.float32),
            "target": np.zeros((self.n_time_steps, self.n_output), dtype=np.float32),
            "reward_data": np.zeros((self.n_time_steps, self.n_output), dtype=np.float32),
            "mask": np.ones((self.n_time_steps, 1), dtype=np.float32),
        }
        # mask out dead time
        self.trial_info['mask'][:int(self.dead_time // self.dt), :] = 0
        # add noise
        for name in ["stim_input", "fix_input", "context_input"]:
            self.trial_info[name] += self.input_noise * np.random.randn(*self.trial_info[name].shape)

    def task_go(
            self,
            variant: Literal[
                "task_go",
                "task_go_rt",
                "task_go_delay",
                "task_go_oic",
                "task_go_oic_rt",
                "task_go_oic_delay",
            ],
            target_offset: int = 0,
            dir_offset: int = 0,
    ) -> None:

        # Event times
        if variant == 'task_go' or variant == 'task_go_oic':
            stim_on = np.random.randint(self.fix_time, self.fix_time + 1000) // self.dt
            stim_off = -1
            fix_end = (self.fix_time + 1000) // self.dt
            resp_on = fix_end
        elif variant == 'task_go_rt' or variant == 'task_go_oic_rt':
            stim_on = np.random.randint(self.fix_time, self.fix_time + 1000) // self.dt
            stim_off = -1
            fix_end = self.n_time_steps
            resp_on = stim_on
        elif variant == 'task_go_delay' or variant == 'task_go_oic_delay':
            stim_on = self.fix_time // self.dt
            stim_off = (self.fix_time + 300) // self.dt
            fix_end = stim_off + np.random.choice(self.delay_times) // self.dt
            resp_on = fix_end

        # choose which RF
        rf = np.random.randint(2)
        # choose input motion direction
        stim_dir = np.random.randint(0, self.n_motion_dirs)

        # target direction based on stim direction and offset
        if "oic" in variant:
            cat0_dirs = (np.arange(self.n_motion_dirs // 2) + dir_offset) % self.n_motion_dirs
            cat0 = stim_dir in cat0_dirs
            if cat0:
                target_idx = target_offset % self.n_motion_dirs
            else:
                target_idx = (target_offset + self.n_motion_dirs // 2) % self.n_motion_dirs
        else:
            target_idx = (stim_dir + target_offset) % self.n_motion_dirs

        # add neural responses
        self.trial_info["stim_input"][stim_on:stim_off, :] += self.circ_tuning(stim_dir, rf)
        self.trial_info["fix_input"][:fix_end, :] += self.tuning_height
        self.trial_info["target"][resp_on:, target_idx] = 1
        self.trial_info["target"][:resp_on, -1] = 1
        self.trial_info["mask"][resp_on:resp_on + self.mask_length // self.dt, :] = 0

        # For classifier
        self.trial_info["target_offset"] = target_offset
        self.trial_info["stim_dir0"] = stim_dir if rf == 0 else -1
        self.trial_info["stim_dir1"] = stim_dir if rf == 1 else -1


    def decision_making(
            self,
            variant: Literal[
                "decision_making",
                "decision_making_rf0",
                "decision_making_rf1",
                "ctx_decision_making_rf0",
                "ctx_decision_making_rf1",
                "decision_making_multisensory",
                "delayed_decision_making",
                "delayed_decision_making_rf0",
                "delayed_decision_making_rf1",
                "delayed_ctx_decision_making_rf0",
                "delayed_ctx_decision_making_rf1",
                "delayed_decision_making_multisensory",
            ],
            target_offset: int = 0
    ) -> None:

        # Determine the motion direction of stim0 and stim 1
        # Must be different
        stim_dir0, stim_dir1 = np.random.choice(self.n_motion_dirs, 2, replace=False)

        # Determine the strengths of the stimuli in each modality
        stim_contrast0 = np.random.choice(self.contrast_set)
        stim_contrast1 = np.random.choice(list(set(self.contrast_set) - set([-stim_contrast0])))
        mean_gamma = 0.8 + 0.4 * np.random.rand()
        gamma_s0_rf0 = mean_gamma + stim_contrast0
        gamma_s1_rf0 = mean_gamma - stim_contrast0
        gamma_s0_rf1 = mean_gamma + stim_contrast1
        gamma_s1_rf1 = mean_gamma - stim_contrast1

        # Determine response directions and convert to output indices
        target_dir_rf0 = np.where(gamma_s0_rf0 > gamma_s1_rf0, stim_dir0, stim_dir1)
        target_dir_rf1 = np.where(gamma_s0_rf1 > gamma_s1_rf1, stim_dir0, stim_dir1)
        target_dir_sum = np.where(
            gamma_s0_rf0 + gamma_s0_rf1 > gamma_s1_rf0 + gamma_s1_rf1, stim_dir0, stim_dir1
        )

        # Set event times
        stim0_on = self.fix_time // self.dt
        if "delayed" in variant:
            stim0_off = stim0_on + 300 // self.dt
            stim1_on = stim0_off + np.random.choice(self.delay_times) // self.dt
            stim1_off = stim1_on + 300 // self.dt
        else:
            stim0_off = stim0_on + np.random.choice(self.delay_times) // self.dt
            stim1_on = stim0_on
            stim1_off = stim0_off
        resp_on = stim1_off

        if "rf0" in variant:
            rf = 0
        elif "rf1" in variant:
            rf = 1
        else:
            rf = np.random.randint(2)

        # calculate stimulus
        if rf == 0 and "ctx" not in variant and "multisensory" not in variant:
            self.trial_info["stim_input"][stim0_on:stim0_off, :] += gamma_s0_rf0 * self.circ_tuning(stim_dir0, 0)
            self.trial_info["stim_input"][stim1_on:stim1_off, :] += gamma_s1_rf0 * self.circ_tuning(stim_dir1, 0)
        elif rf == 1 and "ctx" not in variant and "multisensory" not in variant:
            self.trial_info["stim_input"][stim0_on:stim0_off, :] += gamma_s0_rf1 * self.circ_tuning(stim_dir0, 1)
            self.trial_info["stim_input"][stim1_on:stim1_off, :] += gamma_s1_rf1 * self.circ_tuning(stim_dir1, 1)
        else:
            self.trial_info["stim_input"][stim0_on:stim0_off, :] += gamma_s0_rf0 * self.circ_tuning(stim_dir0, 0)
            self.trial_info["stim_input"][stim1_on:stim1_off, :] += gamma_s1_rf0 * self.circ_tuning(stim_dir1, 0)
            self.trial_info["stim_input"][stim0_on:stim0_off, :] += gamma_s0_rf1 * self.circ_tuning(stim_dir0, 1)
            self.trial_info["stim_input"][stim1_on:stim1_off, :] += gamma_s1_rf1 * self.circ_tuning(stim_dir1, 1)

        # determine target response
        if "multisensory" in variant:
            target_idx = (target_dir_sum + target_offset) % self.n_motion_dirs
        elif rf == 0:
            target_idx = (target_dir_rf0 + target_offset) % self.n_motion_dirs
        elif rf == 1:
            target_idx = (target_dir_rf1 + target_offset) % self.n_motion_dirs


        self.trial_info["fix_input"][:resp_on, :] += self.tuning_height
        self.trial_info["target"][resp_on:, target_idx] = 1
        self.trial_info["target"][:resp_on, -1] = 1
        self.trial_info["mask"][resp_on:resp_on + self.mask_length // self.dt, :] = 0

        # For classifier
        self.trial_info["target_offset"] = target_offset
        self.trial_info["stim_dir0"] = stim_dir0
        self.trial_info["stim_dir1"] = stim_dir1

    def delayed_match(
            self,
            variant: Literal[
                "delayed_match_sample",
                "delayed_match_category",
                "delayed_match_sample_delayed_response",
                "delayed_match_category_delayed_response",
            ],
            dir_offset: int = 0,
            target_offset: int = 0
    ) -> None:

        # choose input motion direction of RF 0
        stim0 = np.random.randint(0, self.n_motion_dirs)
        # motion direction belonging to category 0
        category0 = (np.arange(self.n_motion_dirs // 2) + dir_offset) % self.n_motion_dirs

        if variant in ['delayed_match_sample', 'delayed_match_sample_delayed_response']:
            match = np.random.choice([True, False])
            matching_dir = (stim0 + dir_offset) % self.n_motion_dirs
            not_matching_dirs = list(set(np.arange(self.n_motion_dirs)) - set([matching_dir]))
            if match:
                stim1 = matching_dir
            else:
                stim1 = np.random.choice(not_matching_dirs)
        elif variant in ['delayed_match_category', 'delayed_match_category_delayed_response']:
            stim1 = np.random.randint(0, self.n_motion_dirs)
            stim0_cat = 0 if stim0 in category0 else 1
            stim1_cat = 0 if stim1 in category0 else 1
            match = stim0_cat == stim1_cat
        else:
            raise Exception('Bad variant.')

        # choose which RF
        rf = np.random.randint(2)
        # determine the target response
        match_target_idx = (stim0 + target_offset) % self.n_motion_dirs
        non_match_target_idx = (stim0 + target_offset + self.n_motion_dirs // 2) % self.n_motion_dirs
        target_idx = np.where(match, match_target_idx, non_match_target_idx)

        # Set event times
        stim0_on = self.fix_time // self.dt
        stim0_off = (self.fix_time + 200) // self.dt
        stim1_on = stim0_off + np.random.choice(self.delay_times) // self.dt
        stim1_off = stim1_on + 200 // self.dt

        if variant in ['delayed_match_sample', 'delayed_match_category']:
            resp_on = stim1_on
            self.trial_info["fix_input"][:, :] += self.tuning_height
        else:
            resp_on = stim1_off + np.random.choice([100, 200, 300]) // self.dt
            self.trial_info["fix_input"][:, :resp_on] += self.tuning_height

        self.trial_info["stim_input"][stim0_on:stim0_off, :] += self.circ_tuning(stim0, rf)
        self.trial_info["stim_input"][stim1_on:stim1_off, :] += self.circ_tuning(stim1, rf)
        self.trial_info["target"][resp_on:, target_idx] = 1
        self.trial_info["target"][:resp_on, -1] = 1
        self.trial_info["mask"][resp_on:resp_on + self.mask_length // self.dt, :] = 0

        # For classifier
        self.trial_info["target_offset"] = target_offset
        self.trial_info["stim_dir0"] = stim0 if rf == 0 else -1
        self.trial_info["stim_dir1"] = stim0 if rf == 1 else -1

    def generate_trial(self, task_num: Optional[int] = None) -> Dict[str, Union[np.ndarray, None]]:

        self.reset_trial_info()
        # randomly select task
        if task_num is None:
            task_num = np.random.choice(len(self.tasks))
        # generate trial info
        task = self.tasks[task_num]  # Selects a task from the list

        task[0](*task[1:])
        # add context signal
        self.trial_info["context_input"][:, task_num] += self.tuning_height

        self.trial_info["stimulus"] = np.concatenate(
            (self.trial_info["stim_input"], self.trial_info["fix_input"]), axis=-1
        )

        # TODO: don't need stim_input or fix_input

        self.trial_info["reward"] = self.add_reward_data() if self.RL else None

        return self.trial_info

    def add_reward_data(
            self,
            correct_reward: float = 1.0,
            fix_break_penalty: float = -1.0,
            wrong_choice_penalty: float = -0.01,
    ) -> np.ndarray:

        reward = np.zeros_like(self.trial_info['target'])

        # Determine timings
        fixation_time = np.where(self.trial_info['target'][:, -1])[0]
        response_time = fixation_time[-1] + 1

        # Designate responses
        correct_response = np.where(self.trial_info['target'][response_time, :] == 1)[0]
        incorrect_responses = np.where(self.trial_info['target'][response_time, :-1] == 0)[0]

        # Build reward data
        reward[fixation_time, :-1] = fix_break_penalty
        reward[response_time:, correct_response] = correct_reward
        for i in incorrect_responses:
            reward[response_time:, i] = wrong_choice_penalty

        # Penalize fixating throughout entire trial if response was required
        if not self.trial_info['target'][-1, -1] == 1:
            reward[-1, -1] = fix_break_penalty
        else:
            reward[-1, -1] = correct_reward

        return reward

    def visaulize_trial(self, task_num: Optional[int] = None) -> None:
        """Useful for diagnostic purposes"""
        stim, ctx, target, mask, reward = self.generate_trial(task_num)
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(stim, aspect="auto")
        ax[0, 0].set_title("Stimulus")
        ax[0, 1].imshow(ctx, aspect="auto")
        ax[0, 1].set_title("Context")
        ax[1, 0].imshow(target, aspect="auto")
        ax[1, 0].set_title("Target")
        ax[1, 1].imshow(reward, aspect="auto")
        ax[1, 1].set_title("reward")

        plt.tight_layout()
        plt.show()