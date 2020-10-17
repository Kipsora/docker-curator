import os

import filelock

from pytools.legacy.pytorch import curator
from pytools.legacy.pytorch.engines import Engine
from pytools.pyutils.io import io_registry_group
from pytools.pyutils.io.file_system import get_relative_path, remove

__all__ = ['RotatingSaver']


@curator.register_module('engine')
class RotatingSaver(Engine):
    def __init__(self, ckpt_path, num_ckpts=5):
        super().__init__()

        self._register_hook_key('save')
        self._register_hook_key('save:read_logs')
        self._register_hook_key('save:remove_ckpt')
        self._register_hook_key('save:write_data')
        self._register_hook_key('save:write_logs')
        self._register_hook_key('load')
        self._register_hook_key('load:parse_logs')
        self._register_hook_key('load:read_logs')
        self._register_hook_key('load:read_data')

        self._ckpt_path = get_relative_path(ckpt_path)
        self._ckpt_logs = None
        self._ckpt_file = None

        self._current_name = None
        self._latest_saved_name = None

        self._erased_ckpt_file = None

        self._num_ckpts = num_ckpts

    @property
    def num_ckpts(self):
        return self._num_ckpts

    @property
    def ckpt_path(self):
        return self._ckpt_path

    @property
    def ckpt_logs_file(self):
        return os.path.join(self._ckpt_path, 'checkpoints.json')

    @property
    def erased_ckpt_file(self):
        return self._erased_ckpt_file

    @property
    def last_saved_name(self):
        return self._latest_saved_name

    @property
    def ckpt_file(self):
        return self._ckpt_file

    @property
    def ckpt_logs(self):
        return self._ckpt_logs

    @property
    def ckpt_logs_lock_file(self):
        return os.path.join(self._ckpt_path, 'checkpoints.json.lock')

    def _get_ckpt_logs(self):
        if os.path.exists(self.ckpt_logs_file):
            return io_registry_group.load(self.ckpt_logs_file)
        else:
            return {'checkpoints': []}

    def _get_ckpt_file(self, name):
        return os.path.join(self._ckpt_path, f'model{name}.pth')

    def save(self, name, data):
        self._before('save')
        self._current_name = str(name)

        if self._latest_saved_name != self._current_name:
            with filelock.FileLock(self.ckpt_logs_lock_file):
                self._before('save:read_logs')
                self._ckpt_logs = self._get_ckpt_logs()
                self._finish('save:read_logs')

                while len(self._ckpt_logs['checkpoints']) >= self._num_ckpts > 0:
                    self._before('save:remove_ckpt')
                    self._erased_ckpt_file = os.path.join(self._ckpt_path, self._ckpt_logs['checkpoints'].pop(0))
                    remove(self._erased_ckpt_file)
                    self._finish('save:remove_ckpt')

                self._before('save:write_data')
                self._ckpt_file = self._get_ckpt_file(self._current_name)
                io_registry_group.dump(self._ckpt_file, data)
                self._finish('save:write_data')

                self._before('save:write_logs')
                self._ckpt_logs['checkpoints'].append(get_relative_path(self._ckpt_file, self._ckpt_path))
                io_registry_group.dump(self.ckpt_logs_file, self._ckpt_logs)
                self._finish('save:write_logs')

        self._finish('save')

        self._latest_saved_name = self._current_name

    def load(self, name=None):
        name = name if name is None else str(name)

        self._before('load')

        self._before('load:parse_logs')
        with filelock.FileLock(self.ckpt_logs_lock_file):
            self._before('load:read_logs')
            self._ckpt_logs = self._get_ckpt_logs()
            self._before('load:read_logs')
            if name is None or (os.path.isdir(name) and get_relative_path(name) == self._ckpt_path):
                self._ckpt_file = os.path.join(self._ckpt_path, self._ckpt_logs['checkpoints'][-1])
            else:
                self._ckpt_file = self._get_ckpt_file(name)
                if not get_relative_path(self._ckpt_file, self._ckpt_path) in self._ckpt_logs['checkpoints']:
                    self._ckpt_file = get_relative_path(name)
                    if not get_relative_path(self._ckpt_file, self._ckpt_path) in self._ckpt_logs['checkpoints']:
                        raise FileNotFoundError(f'The checkpoint of "{name}" is not found')
        self._finish('load:parse_logs')

        self._before('load:read_data')
        result = io_registry_group.load(self._ckpt_file)
        self._finish('load:read_data')

        self._finish('load')
        return result
