import time
from logging import getLogger

from gym.wrappers import Monitor as _GymMonitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder as _GymStatsRecorder


class Monitor(_GymMonitor):
    """`Monitor` with PFRL's `ContinuingTimeLimit` support.

    `Agent` in PFRL might reset the env even when `done=False`
    if `ContinuingTimeLimit` returns `info['needs_reset']=True`,
    which is not expected for `gym.Monitor`.

    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
    """

    def _start(
        self,
        directory,
        video_callable=None,
        force=False,
        resume=False,
        write_upon_reset=False,
        uid=None,
        mode=None,
    ):
        if self.env_semantics_autoreset:
            raise NotImplementedError(
                "Detect 'semantics.autoreset=True' in `env.metadata`, "
                "which means the env is from deprecated OpenAI Universe."
            )
        ret = super()._start(
            directory=directory,
            video_callable=video_callable,
            force=force,
            resume=resume,
            write_upon_reset=write_upon_reset,
            uid=uid,
            mode=mode,
        )
        env_id = self.stats_recorder.env_id
        self.stats_recorder = _StatsRecorder(
            directory,
            "{}.episode_batch.{}".format(self.file_prefix, self.file_infix),
            autoreset=False,
            env_id=env_id,
        )
        if mode is not None:
            self._set_mode(mode)
        return ret


class _StatsRecorder(_GymStatsRecorder):
    """`StatsRecorder` with PFRL's `ContinuingTimeLimit` support.

    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/stats_recorder.py
    """

    def __init__(
        self,
        directory,
        file_prefix,
        autoreset=False,
        env_id=None,
        logger=getLogger(__name__),
    ):
        super().__init__(directory, file_prefix, autoreset=autoreset, env_id=env_id)
        self._save_completed = True
        self.logger = logger

    def before_reset(self):
        assert not self.closed

        if self.done is not None and not self.done and self.steps > 0:
            self.logger.debug(
                "Tried to reset the env which is not done=True. "
                "StatsRecorder completes the last episode."
            )
            self.save_complete()

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_step(self, observation, reward, done, info):
        self._save_completed = False
        return super().after_step(observation, reward, done, info)

    def save_complete(self):
        if not self._save_completed:
            super().save_complete()
            self._save_completed = True

    def close(self):
        self.save_complete()
        super().close()
