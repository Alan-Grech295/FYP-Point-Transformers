class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TrialManager(metaclass=Singleton):
    def __init__(self):
        self.trials = []
        self.current_trial = 0
        self.init = False
        self.should_check = True

    @staticmethod
    def add_trials(*args):
        instance = TrialManager()
        instance.trials.extend(args)

    @staticmethod
    def reset():
        instance = TrialManager()
        instance.current_trial = 0
        instance.init = False

    @staticmethod
    def is_trial(*trials) -> bool:
        instance = TrialManager()

        if instance.should_check:
            for t in trials:
                assert t in instance.trials, f"Requested trial {t} does not exist"

        return instance.trials[instance.current_trial] in trials

    @staticmethod
    def __getitem__(trial: str) -> bool:
        return TrialManager.is_trial(trial)

    @staticmethod
    def next_trial() -> bool:
        instance = TrialManager()
        if not instance.init:
            instance.init = True
            return instance.current_trial < len(instance.trials)

        instance.current_trial += 1
        return instance.current_trial < len(instance.trials)

    @staticmethod
    def supress_checks():
        TrialManager().should_check = False

    @staticmethod
    def set_trial(trial: str):
        instance = TrialManager()
        try:
            trial_idx = instance.trials.index(trial)
        except ValueError:
            trial_idx = len(instance.trials)
            instance.trials.append(trial)

        instance.current_trial = trial_idx

    @property
    def trial_name(self) -> str:
        return self.trials[self.current_trial]
