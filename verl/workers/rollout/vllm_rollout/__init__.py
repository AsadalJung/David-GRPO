# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib.metadata import PackageNotFoundError, version
from packaging import version as vs


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)
SPMD_MIN_VERSION = '0.6.6.post2.dev252+g8027a724'

if package_version is None:
    raise ValueError('vllm is not installed in the active environment')
elif vs.parse(package_version) <= vs.parse('0.6.3'):
    vllm_mode = 'customized'
    from .vllm_rollout import vLLMRollout
    from .vllm_rollout_coa import vLLMRollout as vLLMRollout_Search
elif vs.parse(package_version) >= vs.parse(SPMD_MIN_VERSION):
    vllm_mode = 'spmd'
    from .vllm_rollout_spmd import vLLMRollout
    from .vllm_rollout_coa import vLLMRollout as vLLMRollout_Search
else:
    raise ValueError(
        f'vllm version {package_version} is not supported by rollout integration. '
        f'Supported versions are <= 0.6.3 (customized) and >= {SPMD_MIN_VERSION} (spmd).'
    )
