## [0.1.0](https://github.com/Kipsora/pytools/compare/v0.0.1...v0.1.0) (2020-08-12)


### Bug Fixes

* **logging:** add raise & modify `handler_level` pos ([6136449](https://github.com/Kipsora/pytools/commit/613644910dd6f562e6a9b6dcd08de226144f2ee3))
* **logging:** remove incorrect type annotation ([0b75bbe](https://github.com/Kipsora/pytools/commit/0b75bbe578bdda49fc38c2d84179dea0f0aa8f6b))
* **misc:** add `remove_prefix` to `__all__` ([c418aa6](https://github.com/Kipsora/pytools/commit/c418aa69708535d8e99d9c75f4727638c5153213))
* **misc:** change behavior of check decorator ([a3596e7](https://github.com/Kipsora/pytools/commit/a3596e7e0aafc88dcf20ed698febf5deea3d81a3))
* **misc:** let `python_member` work & rename arg ([816e95a](https://github.com/Kipsora/pytools/commit/816e95a1f86de45483c0f1a16feeda81118ad953))
* **misc:** modify to use new nested API ([f57a237](https://github.com/Kipsora/pytools/commit/f57a237a722f884aa190f2afa6cd688f06d2ebf7))
* **misc:** only print deprecated message once ([6496f36](https://github.com/Kipsora/pytools/commit/6496f3603c5335a56fc40fd58ccddeb7d592a914))
* **misc:** return self when load config ([a8faf4a](https://github.com/Kipsora/pytools/commit/a8faf4a3f274c4a7f21542ef5ea6e8b4eaa619b7))
* **misc:** use existing method instead ([22de050](https://github.com/Kipsora/pytools/commit/22de050080d19f45d8ebc54ae4cde241d1889633))
* **misc:** use the as-is key for `AttrDict` ([a20f18b](https://github.com/Kipsora/pytools/commit/a20f18bd521bd1de0b1068485615891c7f497be6))
* **pytorch:** fix many checkers ([14e2048](https://github.com/Kipsora/pytools/commit/14e2048dee99a8d9a28d82696f92639f960a1de2))
* **pytorch:** fix path issue for session manager ([d3b7c63](https://github.com/Kipsora/pytools/commit/d3b7c6300555a8c3ee282eb81361586bf1cdd407))
* **pytorch:** make `use_scalar_value` effective ([f2d7623](https://github.com/Kipsora/pytools/commit/f2d7623279f732dcd2e0f091ca997c048e19339c))
* **pytorch:** remove distillation import ([d19b545](https://github.com/Kipsora/pytools/commit/d19b545446b96d4750d65af5d07bfa2a3f8e1038))
* **pytorch:** use reduction dict & fix bugs ([bc80742](https://github.com/Kipsora/pytools/commit/bc80742a98e2409b3972670771ff64443cf9f9ec))
* **trainer:** prune project specific trainer codes ([d9085a9](https://github.com/Kipsora/pytools/commit/d9085a9ef62064b14dc4ffa189f5dcce1e48643f))


### Features

* **io:** add `get_absolute_path` ([5e781b4](https://github.com/Kipsora/pytools/commit/5e781b4478b9711879399a0e320e88d569e05d74))
* **io:** add two fs funcs & `use_real_path` option ([62d195b](https://github.com/Kipsora/pytools/commit/62d195bcfd892dbe0f02438997a9ed242f8f9fa6))
* **logging:** add group access to wrapper ([3b7cdf2](https://github.com/Kipsora/pytools/commit/3b7cdf2d989e44e8ed8f4fdaadeaed8c58524368))
* **logging:** add `LoggerWrapper` ([6bd7fef](https://github.com/Kipsora/pytools/commit/6bd7fef5ac5422beefcb75cbc7c69ea2acf0b3b9))
* **logging:** add msec display option for logger ([aebcd7f](https://github.com/Kipsora/pytools/commit/aebcd7fc0a5817d44d7a330471044ae5bfcb2893))
* **logging:** add option for keeping formatters & levels ([439361f](https://github.com/Kipsora/pytools/commit/439361fc0313e761981a531dcff8706fa41a4ad3))
* **misc:** add `classproperty` & `cached_property` ([91ba3be](https://github.com/Kipsora/pytools/commit/91ba3bee6cadda3500edf17295d6bd77fe7c5960))
* **misc:** add deprecated decorator ([2343b9c](https://github.com/Kipsora/pytools/commit/2343b9c4f9d5598bec00e0a59499928f64a22bec))
* **misc:** add parsed_path option ([e63f1bf](https://github.com/Kipsora/pytools/commit/e63f1bf10667dd40066110a67a12e50a161d2638))
* **misc:** add `set_config` method ([cdef43e](https://github.com/Kipsora/pytools/commit/cdef43e0d071032fa89f76a659728000b470089e))
* **misc:** replace nested func with handler class ([52548af](https://github.com/Kipsora/pytools/commit/52548af73b8bbe9b84afaf2192015583f62a51ca))
* **pytf:** add pytf package and some helper funcs ([c84bba2](https://github.com/Kipsora/pytools/commit/c84bba26e0aa6c8f7798ae4a8062c723eed4e700))
* **pytorch:** add global state helpers ([01abcc3](https://github.com/Kipsora/pytools/commit/01abcc327afb989ff2ca3f94f46ebc6dfdf08c16))
* **pytorch:** add new training interface ([e8e423f](https://github.com/Kipsora/pytools/commit/e8e423f83a1702978b01599c26ad532bd3d98679))
* **string:** add `row_pad_suffix` ([3457dc6](https://github.com/Kipsora/pytools/commit/3457dc614c49aebcaf87dc440a1c82f5a0e3bd32))


### BREAKING CHANGES

* **pytorch:** this modifies the way to use pytorch training package.

  To migrate the code follow the example below:

  Before:

  ```python
  >>> from pytools import pytorch
  ```

  After:

  ```python
  >>> from pytools.legacy import pytorch
  ```

  Note that the legacy modules contain outdated codes and will be removed in
  future releases.

  The new pytorch training interface is rewritten in hook based, which is
  considered as a more general way to train a neural network.
* **misc:** this modifies the way to use `ConfigBuilder`.

  To migrate the code follow the example below:

  Before:

  ```python
  >>> builder = ConfigBuilder()
  >>> builder.load_python_config("./config/dataset.py", recursive=True)
  ```

  After:

  ```python
  >>> builder = ConfigBuilder()
  >>> builder.load_python_config("./config/dataset.py", use_recursive_config=True)
  ```

  To make more meaningful & united interface.
* **misc:** this modifies the way to use `nested_merge` and `nested_dictify`

  To migrate the code follow the example below:

  Before:

  ```python
  >>> a = nested_dictify(a)
  >>> b = nested_dictify(b)
  >>> c = nested_merge(a, b)
  ```

  After:

  ```python
  >>> dictifier = AttrDictifier()
  >>> merger = DictOnlyMerger()
  >>> a = dictifier.dictify(a)
  >>> b = dictifier.dictify(b)
  >>> c = merger.merge(a, b)
  ```

  To merge two object there will be complicated situations and use handler class will be easier to handle more cases.



## 0.0.1 (2020-03-25)


### Bug Fixes

* **misc:** fix dead loop when fallback is not set ([63f2b10](https://github.com/Kipsora/pytools/commit/63f2b10a592f6025db220e67e8c2a0e9b1438e01))
* **pytorch:** add `is_distributed_enabled` as args ([14601a0](https://github.com/Kipsora/pytools/commit/14601a00188da02dadc9f6c60ceec5e1c9744b31))
* **pytorch:** initialize bas group when activating ([519dab4](https://github.com/Kipsora/pytools/commit/519dab4b2f52cc5720a7b55e617b4bd907978b65))
* **pytorch:** use the unique output for composing ([5042678](https://github.com/Kipsora/pytools/commit/5042678c913e8bcb1cfbee49e9b2d253cdd0cffb))


### Code Refactoring

* move `synchronized_member_fn` to misc ([366eada](https://github.com/Kipsora/pytools/commit/366eada01bbe329cde12c44519ceb537fef5424a))


### Features

* **io:** add `ensure_directory` helper ([b606728](https://github.com/Kipsora/pytools/commit/b60672840a3c8ce1eb17a31b360321d8fea01e25))
* **io:** print `__jsonstr__` when jsonifying data ([2e2ea65](https://github.com/Kipsora/pytools/commit/2e2ea65ada22220b50439653fae8ee8123fddc24))
* **logging:** use handler level for loggers ([8f95762](https://github.com/Kipsora/pytools/commit/8f957628765f0da452ab815f341266c742c5c6a8))
* **misc:** add `__jsonstr__` of built `registry` obj ([b6e5180](https://github.com/Kipsora/pytools/commit/b6e51800a40a55ed7a602c2333313aeb24e0e0c9))
* **misc:** add `build_fn` to `registry` ([749f51d](https://github.com/Kipsora/pytools/commit/749f51d32806fdcabf8aacc5279cfd81804cc995))
* **misc:** add fallback to `AttrDict` ([501f857](https://github.com/Kipsora/pytools/commit/501f8571e72eb52c5e16f550b4cc9defb3b05ffd))
* **misc:** add `get_parsed_config` helper ([a03b2ff](https://github.com/Kipsora/pytools/commit/a03b2ffc6e05cbc4f202eba49987e25437d9ddea))
* **misc:** use P`ythonConfigManager` for management ([15e05ba](https://github.com/Kipsora/pytools/commit/15e05ba9eaf2216173a5f7ea8297dda9e55a357c))
* **misc:** use str-helpers to extract config path ([c8c13d8](https://github.com/Kipsora/pytools/commit/c8c13d892dd6317ce389f879f5d52d8364fb7e7e))
* **pytorch:** add class for composing criterion ([a3ae94a](https://github.com/Kipsora/pytools/commit/a3ae94af9b421020dac960179f0120cea12668e8))
* **pytorch:** add distributed module ([49d394a](https://github.com/Kipsora/pytools/commit/49d394af50b1c36f63ff53ca4410cae4dc0d5e6a))
* **pytorch:** add `get_rank` and `get_size` ([fc49c43](https://github.com/Kipsora/pytools/commit/fc49c431fa9b786c2ee3cf6d0fe4a707bea4d67c))
* **pytorch:** add `get_sampler` & `get_local_rank` ([2294c55](https://github.com/Kipsora/pytools/commit/2294c551c05f0990ca8af74f92c5ba93b219aa15))
* **pytorch:** add ImageNet normalizer & transform ([0edeb77](https://github.com/Kipsora/pytools/commit/0edeb77afcdc9b76162684d9c3c9188207919cb8))
* **pytorch:** add `meters` module ([d940813](https://github.com/Kipsora/pytools/commit/d940813739f7f4fff7922273f8c60ca0ecdff56e))
* **pytorch:** add shuffle to `get_sampler` ([2ed24be](https://github.com/Kipsora/pytools/commit/2ed24be6c6f3f95dd1da64c76990c1fffea578b4))
* **string:** add string helper functions ([bfe40da](https://github.com/Kipsora/pytools/commit/bfe40da1f5928142bdf5822e9bac0ef66121fac7))


### Styles

* **misc:** move `ConfigManager` -> `ConfigBuilder` ([744a73c](https://github.com/Kipsora/pytools/commit/744a73cc225e68cb7dc988b50f3ac8af9abb801a))
* **misc:** move `python_config` to config ([18b1540](https://github.com/Kipsora/pytools/commit/18b1540e415054f0f2993314c7a9be3fa1b94ccc))
* **misc:** `set_random_seed` -> `set_python_seed` ([5a4e53c](https://github.com/Kipsora/pytools/commit/5a4e53c0eb51c36ba59fddd110136ca350be28b5))
* **misc:** use `nested_dictify` for dictify data ([60a07bd](https://github.com/Kipsora/pytools/commit/60a07bd7d2505890186c42b3206c75aa41084a5f))
* **pytorch:** remove inputs for objectives ([99d1953](https://github.com/Kipsora/pytools/commit/99d1953c073dd80ed0f4d86fc32f8b3923715431))
* **pytorch:** rename distributed checkers ([4a5109f](https://github.com/Kipsora/pytools/commit/4a5109f4b058b066e1ebb14f517fce415788e0eb))


### BREAKING CHANGES

* **pytorch:** this modifies the way of using `activate`

  To migrate the code follow the example below:

  Before:

  ```python3
  >>> with distributed.activate(is_distributed_enabled, *args,
  ...                           **kwargs) as group:
  ```

  After:

  ```python3
  >>> with distributed.activate(is_distributed_enabled, local_rank, *args,
  ...                           **kwargs) as group:
  ```

  This adds interface to incorporate `local_rank`.
* **misc:** this modifies the way of using `ConfigManager`

  To migrate the code follow the example below:

  Before:

  ```python3
  >>> config = ConfigManager().get_config()
  ```

  After:

  ```python3
  >>> config = ConfigBuilder().get_config()
  ```

  This context is more accurate by ysing "Builder" instead of "Manager".
* **misc:** this modifies the way to use `set_random_seed`

  To migrate the code follow the example below:

  Before:

  ```python3
  >>> set_random_seed(0)
  ```

  After:

  ```python3
  >>> set_python_seed(0)
  ```

  Use `set_python_seed` is more accurate in this context.
* **pytorch:** this modifies the way to use `Objective`

  To migrate the code follow the example below:

  Before:

  ```python3
  >>> objective = Objective()
  >>> loss = objective(inputs, outputs, targets)
  ```

  After:

  ```python3
  >>> objective = Objective()
  >>> loss = objective(outputs, targets)
  ```

  This is intended for consistency with pytorch criterion and advanced
  objectives should be implemented in another class such as using
  `ComposeObjective`.
* **pytorch:** this modifies the way to use distributed checkers

  To migrate the code follow the example below:

  Before:

  ```python3
  >>> from pytools.pytorch.distributed import check_if_activated
  >>> from pytools.pytorch.distributed import check_if_distributed_enabled
  ```

  After:

  ```python3
  >>> from pytools.pytorch.distributed import is_distribute_activated
  >>> from pytools.pytorch.distributed import is_distribute_mode
  ```

  After change we can use more formal form of calling checkers.
* this modifies the way to use `synchronized_member_fn`

  To migrate the code follow the example below:

  Before:

  ```python3
  >>> from pytools.pyutils.concurrent.synchronized import synchronized_member_fn
  ```

  After:

  ```python3
  >>> from pytools.pyutils.misc.decorator import synchronized_member_fn
  ```

  This is becuase `syncrhonized_member_fn` is better in a "decorator" context.
* **misc:** this modifies the way to use PythonConfigManager

  To migrate the code follow the example below:

  Before:

  ```python3
  >>> from pytools.pyutils.misc.python_config import PythonConfigManager
  >>> manager = PythonConfigManager()
  ```

  After:

  ```python3
  >>> from pytools.pyutils.misc.config import ConfigManager
  >>> manager = ConfigManager()
  ```

  This creates new possibilities to integrating non-python configs including JSON and YAML files.
* **misc:** this modifies the way to use `nested_attr`

  To migrate the code follow the example below:

  Before:

  ```python3
  data = nested_attrdict(data)
  ```

  After:

  ```python3
  data = nested_dictify(data)
  ```

  This is intended for integrating `dict_class` to the arguments for use subclass of `AttrDict`.
* **misc:** this modifies the way to use `python_config`.

  To migrate the code follow the example below:

  Before:

  ```python3
  config = load_python_config(path)
  ```

  After:

  ```python3
  manager = PythonConfigManager()
  manager.load_python_config(path)
  config = manager.get_config()
  ```

  This modification is intended for integrating further functions.



