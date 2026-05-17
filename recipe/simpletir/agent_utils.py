import asyncio
import os
import re
import io
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from verl import DataProto

if os.getenv("SANDBOX_ENDPOINT", None) is not None:
    from sandbox.local_sandbox import parallel_sandbox
else:
    from sandbox.internal_sandbox import parallel_sandbox

MAX_LENGTH_TRUNCATE_CONTENT = 20000


async def parallel_sandbox_with_namespace(
    tasks: List[Tuple[str, dict]],
    num_processes: int = 200,
) -> Tuple[List[bool], List[str], List[str], List[dict]]:
    """
    Execute multiple code snippets with their respective namespaces.

    Args:
        tasks: List of (code, namespace) tuples
        num_processes: Maximum number of concurrent processes

    Returns:
        Tuple of (success_flags, stdouts, stderrs, updated_namespaces)
    """
    namespaces = [ns.copy() for _, ns in tasks]

    wrapped_codes = []
    for code, namespace in tasks:
        namespace_init = _serialize_namespace(namespace)
        wrapped_code = f"""
import pickle
import dill
import base64
import sys
import io

namespace = {{}}
{namespace_init}

old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

success = True
try:
    exec('''{code}''', namespace)
except Exception as e:
    success = False
    import traceback
    sys.stderr.write(traceback.format_exc())
finally:
    stdout_value = sys.stdout.getvalue()
    stderr_value = sys.stderr.getvalue()
    sys.stdout = old_stdout
    sys.stderr = old_stderr

print(stdout_value, end='')

sys.stderr.write("__VERL_META_START__\\n")
sys.stderr.write(f"SUCCESS={{success}}\\n")
filtered_ns = {{k: v for k, v in namespace.items() if not k.startswith('__')}}
try:
    try:
        import dill
        encoded = base64.b64encode(dill.dumps(filtered_ns)).decode('ascii')
    except ImportError:
        safe_ns = {{}}
        for k, v in filtered_ns.items():
            try:
                pickle.dumps(v)
                safe_ns[k] = v
            except:
                pass
        encoded = base64.b64encode(pickle.dumps(safe_ns)).decode('ascii')
    sys.stderr.write(f"NAMESPACE={{encoded}}\\n")
except Exception as e:
    sys.stderr.write(f"NAMESPACE=PICKLE_ERROR:{{str(e)}}\\n")
sys.stderr.write("__VERL_META_END__\\n")
sys.stderr.write(stderr_value)
"""
        wrapped_codes.append(wrapped_code)
    ok_flags, raw_stdouts, raw_stderrs = await parallel_sandbox(wrapped_codes, num_processes=num_processes)

    success_flags = []
    stdouts = []
    stderrs = []
    updated_namespaces = []

    for i, (ok, raw_stdout, raw_stderr) in enumerate(zip(ok_flags, raw_stdouts, raw_stderrs)):
        if not ok:
            success_flags.append(False)
            stdouts.append("")
            stderrs.append(raw_stderr)
            updated_namespaces.append(namespaces[i])
        else:
            try:
                stdouts.append(raw_stdout)

                meta_info = _parse_meta_from_stderr(raw_stderr)
                success_flags.append(meta_info['success'])
                stderrs.append(meta_info['stderr'])

                if not meta_info['namespace_encoded'].startswith("PICKLE_ERROR"):
                    try:
                        import base64
                        try:
                            import dill
                            updated_ns = dill.loads(base64.b64decode(meta_info['namespace_encoded']))
                        except ImportError:
                            import pickle
                            updated_ns = pickle.loads(base64.b64decode(meta_info['namespace_encoded']))
                        updated_namespaces.append(updated_ns)
                    except Exception as e:
                        print(f"[ERROR] Failed to deserialize namespace: {e}")
                        updated_namespaces.append(namespaces[i])
                else:
                    print(f"[ERROR] Namespace serialization failed: {meta_info['namespace_encoded']}")
                    updated_namespaces.append(namespaces[i])
            except Exception as e:
                success_flags.append(False)
                stdouts.append(raw_stdout)
                stderrs.append(f"Parse error: {e}\n{raw_stderr}")
                updated_namespaces.append(namespaces[i])

    return success_flags, stdouts, stderrs, updated_namespaces


def _serialize_namespace(namespace: dict) -> str:
    if not namespace:
        return ""

    try:
        import dill
        import base64
        encoded = base64.b64encode(dill.dumps(namespace)).decode('ascii')
        return f"namespace = dill.loads(base64.b64decode('{encoded}'))"
    except Exception as e:
        print(f"[WARNING] Failed to serialize namespace with dill: {e}")
        lines = []
        for key, value in namespace.items():
            if key.startswith('__'):
                continue
            try:
                import pickle
                import base64
                encoded = base64.b64encode(pickle.dumps(value)).decode('ascii')
                lines.append(f"namespace['{key}'] = pickle.loads(base64.b64decode('{encoded}'))")
            except:
                pass
        return '\n'.join(lines)


def _parse_meta_from_stderr(stderr: str) -> dict:
    result = {
        'namespace_encoded': '',
        'stderr': '',
        'success': False
    }

    import re
    meta_match = re.search(r'__VERL_META_START__\n(.*?)__VERL_META_END__\n', stderr, re.DOTALL)

    if meta_match:
        meta_content = meta_match.group(1)
        success_match = re.search(r'SUCCESS=(True|False)', meta_content)
        if success_match:
            result['success'] = success_match.group(1) == 'True'

        namespace_match = re.search(r'NAMESPACE=(.+)', meta_content)
        if namespace_match:
            result['namespace_encoded'] = namespace_match.group(1).strip()

        result['stderr'] = stderr[meta_match.end():]
    else:
        result['stderr'] = stderr

    return result


def truncate_content(
    content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT
) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )

def check_reward_hacking(text: str) -> bool:
    python_code_block_pattern = r'```(python|py)\s*(.*?)```'
    matches = list(re.finditer(python_code_block_pattern, text, re.DOTALL))

    if len(matches) < 2:
        return False

    for i in range(len(matches) - 1):
        end_of_current_block = matches[i].end()
        start_of_next_block = matches[i + 1].start()
        interstitial_text = text[end_of_current_block:start_of_next_block]

        if interstitial_text.strip() == "":
            return True

    previous_logic_union = set()

    for i, match in enumerate(matches):
        raw_code = match.group(2)

        if not raw_code.strip():
            return True

        processed_code = auto_print_last_expression(raw_code)
        current_logic_set = set()
        lines = processed_code.splitlines()

        for line in lines:
            line_no_comment = re.sub(r'#.*', '', line)
            clean_line = "".join(line_no_comment.split())
            if not clean_line:
                continue
            if clean_line.startswith("print(") or clean_line.startswith("display(") or clean_line.startswith("pprint("):
                continue
            current_logic_set.add(clean_line)

        if not current_logic_set:
            if previous_logic_union:
                return True
            else:
                continue

        if current_logic_set.issubset(previous_logic_union):
            return True

        if previous_logic_union and previous_logic_union.issubset(current_logic_set):
            return True

        previous_logic_union.update(current_logic_set)

    return False

@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int


class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        keys: List[str],
        cut_left: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict["attention_mask"].sum(dim=1).max()
        result = tensor_dict.copy()

        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(
        self, tensor: torch.Tensor, pad_to_left: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert padding structure and return sorted tensor with indices."""
        mask = (
            tensor != self.config.pad_token_id
            if pad_to_left
            else tensor == self.config.pad_token_id
        )
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(
        self, tensors: List[torch.Tensor], pad_to_left: bool = True
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(
        self,
        responses: torch.Tensor,
        responses_str: List[str],
        active_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with pad tokens.
        """
        assert active_mask.sum() == responses.shape[0], (
            f"{active_mask.sum()} != {responses.shape[0]}"
        )
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len),
            self.config.pad_token_id,
            dtype=responses.dtype,
            device=responses.device,
        )
        padded_responses[active_mask] = responses

        # Create masked response strings
        padded_responses_str = [""] * batch_size

        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1

        return padded_responses, padded_responses_str


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    rollout_n: int
    mask_void_turns: bool


class AgentHelper:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config

        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length,
            )
        )

        self.prompt_dict = {
            "no_tool_prompt": "\n",
            "final_prompt": "\n",
        }
        self.error_n_line = 1
        self.sample_namespaces = {}

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, add_special_tokens=False, return_tensors="pt", padding="longest"
        )["input_ids"]

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        python_code_block_pattern = r'```(python|py)\s*(.*?)\s*```'

        for i in range(len(responses_str)):
            tool_call_pos = responses_str[i].find("<tool_call>")
            if tool_call_pos != -1:
                responses_str[i] = responses_str[i][:tool_call_pos + len("<tool_call>")]

                code_block_matches = list(re.finditer(python_code_block_pattern, responses_str[i], re.DOTALL))
                if code_block_matches:
                    last_code_block_match = code_block_matches[-1]
                    last_code_block_end_pos = last_code_block_match.end()
                    responses_str[i] = responses_str[i][:last_code_block_end_pos]
                    responses_str[i] = responses_str[i] + "<tool_call>"

        responses = self._batch_tokenize(responses_str)

        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""

        next_obs_ids = self.tokenizer(
            next_obs,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,  # Prevents adding special tokens
        )["input_ids"]

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(
                f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}"
            )
            next_obs_ids = next_obs_ids[:, : self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(
        self,
        rollings: DataProto,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor,
    ) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding(
            [rollings.batch["input_ids"], cur_responses, next_obs_ids]
        )

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict(
            {
                "input_ids": new_input_ids[:, -max_len:],
                "position_ids": new_position_ids[:, -max_len:],
                "attention_mask": new_attention_mask[:, -max_len:],
            }
        )
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _info_masked_concatenate_with_padding(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True,
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(
                info.size(), pad_id, dtype=info.dtype, device=info.device
            )  # information mask
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(
        self,
        right_side: Dict,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None,
    ) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = (
                self._info_masked_concatenate_with_padding(
                    right_side["responses"],
                    right_side["responses_with_info_mask"],
                    cur_responses,
                    next_obs_ids,
                    pad_to_left=False,
                )
            )
        else:
            responses, responses_with_info_mask = (
                self._info_masked_concatenate_with_padding(
                    right_side["responses"],
                    right_side["responses_with_info_mask"],
                    cur_responses,
                    pad_to_left=False,
                )
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return {
            "responses": responses[:, :max_len],
            "responses_with_info_mask": responses_with_info_mask[:, :max_len],
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
        if active_batch size is not divisible by num_gpus, pad with first sequence
        then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch["input_ids"].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()

        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        if hasattr(active_batch, "meta_info"):
            padded_active_batch.meta_info = active_batch.meta_info

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        # Handle meta_info if present
        if hasattr(padded_output, "meta_info") and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(
        self,
        gen_batch,
        initial_input_ids: torch.Tensor,
        timeout: int = 5,
    ) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        batch_size = gen_batch.batch["input_ids"].shape[0]

        self.timeout = timeout

        original_left_side = {
            "input_ids": initial_input_ids[:, -self.config.max_start_length :]
        }
        original_right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
        }

        active_mask = torch.ones(batch_size * self.config.rollout_n, dtype=torch.bool)
        void_turn_mask = torch.ones(
            batch_size * self.config.rollout_n, dtype=torch.bool
        )  # if void turn, set to False
        turns_stats = torch.ones(batch_size * self.config.rollout_n, dtype=torch.int)
        use_code_stats = torch.zeros(
            batch_size * self.config.rollout_n, dtype=torch.int
        )
        delay_call_time = torch.zeros(
            batch_size * self.config.rollout_n, dtype=torch.int
        )
        valid_code_stats = torch.zeros(
            batch_size * self.config.rollout_n, dtype=torch.int
        )
        success_code_lines = []
        fail_code_lines = []
        success_code_strip_lines = []
        fail_code_strip_lines = []
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        for i in range(batch_size * self.config.rollout_n):
            if i not in self.sample_namespaces:
                self.sample_namespaces[i] = {}

        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch, keys=["input_ids", "attention_mask", "position_ids"]
            )

            # Generate responses for active batches
            if step != 0:
                rollings_active = DataProto.from_dict(
                    {k: v[active_mask] for k, v in rollings.batch.items()}
                )
                rollings_active.meta_info["n"] = 1
            else:
                rollings_active = rollings
                if self.config.rollout_n == 1:
                    rollings_active.meta_info["n"] = 1
                else:
                    repeated_rollings_dict = {}
                    for k, v in rollings.batch.items():
                        repeated_rollings_dict[k] = v.repeat_interleave(
                            self.config.rollout_n, dim=0
                        )
                    rollings = DataProto.from_dict(repeated_rollings_dict)
                    for k, v in original_left_side.items():
                        original_left_side[k] = v.repeat_interleave(
                            self.config.rollout_n, dim=0
                        )
                    for k, v in original_right_side.items():
                        original_right_side[k] = v.repeat_interleave(
                            self.config.rollout_n, dim=0
                        )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # Post-process responses
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(
                gen_output.batch["responses"]
            )
            responses_ids, responses_str = self.tensor_fn._example_level_pad(
                responses_ids, responses_str, active_mask
            )

            # Execute code and get next inputs
            next_obs, dones, is_void_turn, code_info = self.execute_predictions(
                responses_str, active_mask
            )

            curr_active_mask = torch.tensor(
                [not done for done in dones], dtype=torch.bool
            )
            active_mask = active_mask * curr_active_mask
            void_turn_mask = void_turn_mask * torch.tensor(
                [not v for v in is_void_turn], dtype=torch.bool
            )
            active_num_list.append(active_mask.sum().item())
            use_code_stats += torch.tensor(code_info["use_code"], dtype=torch.int)
            delay_call_time += torch.tensor(code_info["delay_call_num"], dtype=torch.int)
            valid_code_stats += torch.tensor(code_info["valid_code"], dtype=torch.int)
            success_code_lines.extend(code_info["success_code_lines"])
            fail_code_lines.extend(code_info["fail_code_lines"])
            success_code_strip_lines.extend(code_info["success_code_strip_lines"])
            fail_code_strip_lines.extend(code_info["fail_code_strip_lines"])

            if step == self.config.max_turns - 2:
                for i, obs in enumerate(next_obs):
                    if len(obs) > 0:
                        next_obs[i] += self.prompt_dict["final_prompt"]

            if step < self.config.max_turns - 1:
                turns_stats[curr_active_mask] += 1
            next_obs_ids = self._process_next_obs(next_obs)
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(
                original_right_side, responses_ids, next_obs_ids
            )

        meta_info["turns_stats"] = turns_stats.tolist()
        meta_info["active_mask"] = active_mask.tolist()
        meta_info["void_turn_mask"] = void_turn_mask.tolist()
        meta_info["use_code_stats"] = use_code_stats.tolist()
        meta_info["delay_call_time"] = delay_call_time.tolist()
        meta_info["valid_code_stats"] = valid_code_stats.tolist()
        meta_info["success_code_lines"] = success_code_lines
        meta_info["fail_code_lines"] = fail_code_lines
        meta_info["success_code_strip_lines"] = success_code_strip_lines
        meta_info["fail_code_strip_lines"] = fail_code_strip_lines

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        for i in range(batch_size * self.config.rollout_n):
            if i in self.sample_namespaces:
                del self.sample_namespaces[i]

        return self._compose_final_output(
            original_left_side, original_right_side, void_turn_mask, meta_info
        )

    def _compose_final_output(
        self,
        left_side: Dict,
        right_side: Dict,
        void_turn_mask: torch.Tensor,
        meta_info: Dict,
    ) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output["prompts"] = left_side["input_ids"]

        # Combine input IDs
        final_output["input_ids"] = torch.cat(
            [left_side["input_ids"], right_side["responses"]], dim=1
        )

        # Create attention mask and position ids
        final_output["attention_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(final_output["responses"]),
            ],
            dim=1,
        )
        final_output["info_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(
                    final_output["responses_with_info_mask"]
                ),
            ],
            dim=1,
        )

        # create void turn mask
        final_output["void_turn_mask"] = void_turn_mask

        final_output["position_ids"] = self.tensor_fn.create_position_ids(
            final_output["attention_mask"]
        )

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        for key in final_output.batch.keys():
            final_output.batch[key] = final_output.batch[key].long()

        return final_output

    def execute_predictions(self, predictions: List[str], active_mask) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM

        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding

        Returns:
            List of observation strings
        """
        next_obs = [None] * len(active_mask)
        dones = [0] * len(active_mask)
        use_code = [0] * len(active_mask)
        delay_call_num = [0] * len(active_mask)
        valid_code = [0] * len(active_mask)
        is_void_turn = [0] * len(active_mask)  # default no void turn
        success_code_lines = []
        success_code_strip_lines = []
        fail_code_lines = []
        fail_code_strip_lines = []

        code_actions = []
        tool_call_pattern = r"<tool_call>"
        python_code_block_pattern = r'```(python|py)\s*(.*?)\s*```'

        for i, prediction in enumerate(predictions):
            boxed_positions = list(re.finditer(r"\\boxed{", prediction))
            if boxed_positions:
                last_boxed_position = boxed_positions[-1].end()
                after_last_boxed = prediction[last_boxed_position:]
                if not re.search(python_code_block_pattern, after_last_boxed, re.DOTALL):
                    next_obs[i] = ""
                    dones[i] = 1

            if re.search(tool_call_pattern, prediction):
                code_blocks = re.findall(python_code_block_pattern, prediction, re.DOTALL)
                temp_delay_call_num = max(len(code_blocks) - 1, 0)
                if temp_delay_call_num > 0:
                    cheat_flag = check_reward_hacking(prediction)
                    if cheat_flag:
                        temp_delay_call_num = 0
                delay_call_num[i] = temp_delay_call_num

                code = "\n".join([block[1] for block in code_blocks if block[1]])

                if code_blocks:
                    use_code[i] = 1
                else:
                    use_code[i] = 0
                    code = None

            else:

                code = None
                use_code[i] = 0
                
            code_actions.append((i, code))

        tasks = []
        index_mapping = []
        for i, code in code_actions:
            if not active_mask[i]:
                next_obs[i] = ""
                dones[i] = 1
                use_code[i] = 0
                valid_code[i] = 0
            elif code is None:
                if dones[i] == 0:
                    # Neither answer(\boxed) nor code is detected, directly stop the generation
                    next_obs[i] = self.prompt_dict["no_tool_prompt"]

                    dones[i] = 1
                    # It is likely that single turn response length is exceeded
                    # If so, stop following generations due to void turns
                    # But seems that no responses is overlong, so comment it now
                    # if responses_ids[i].shape[0] >= self.config.max_response_length:
                    if self.config.mask_void_turns:
                        is_void_turn[i] = 1
            else:
                # Auto-print last expression if it's a simple variable/expression
                code = auto_print_last_expression(code)
                tasks.append((code, self.sample_namespaces[i]))
                index_mapping.append(i)

        if tasks:
            sandbox_success, sandbox_stdout, sandbox_stderr, updated_namespaces = asyncio.run(
                parallel_sandbox_with_namespace(tasks, num_processes=256)
            )
            for j, env_idx in enumerate(index_mapping):
                success = sandbox_success[j]
                stdout = str(sandbox_stdout[j])
                stderr = str(sandbox_stderr[j])
                total_lines, code_lines = count_lines(tasks[j][0])

                if success:
                    self.sample_namespaces[env_idx] = updated_namespaces[j]

                obs = ""
                if len(stderr) > 0:
                    valid_code[env_idx] = 0
                    fail_code_lines.append(total_lines)
                    fail_code_strip_lines.append(code_lines)

                    stderr_lines = stderr.splitlines()
                    truncated_stderr = truncate_content(
                        "\n".join(stderr_lines[-self.error_n_line :]), max_length=512
                    )
                    obs = f"\nCode execution result: {truncated_stderr}\n</tool_call>\n"
                elif len(stdout) > 0:
                    valid_code[env_idx] = 1
                    success_code_lines.append(total_lines)
                    success_code_strip_lines.append(code_lines)

                    truncated_stdout = truncate_content(stdout, max_length=512)
                    obs = f"\nCode execution result: {truncated_stdout}\n</tool_call>\n"
                else:
                    # no stdout nor stderr
                    # this can happen upon sandbox error or the code block itself
                    # did not print anything
                    if not success:
                        obs = "\nCode execution result: interpreter timeout\n</tool_call>\n"
                    else:
                        obs = "\nCode execution result: \n</tool_call>\n"
                next_obs[env_idx] = obs
                if "\\boxed{" in stdout:
                    dones[env_idx] = 1

        code_info = {
            "use_code": use_code,
            "delay_call_num": delay_call_num,
            "valid_code": valid_code,
            "success_code_lines": success_code_lines,
            "fail_code_lines": fail_code_lines,
            "success_code_strip_lines": success_code_strip_lines,
            "fail_code_strip_lines": fail_code_strip_lines,
        }
        print(
            f"[debug] void turn number: {sum(is_void_turn)} out of {active_mask.sum()} samples"
        )

        return next_obs, dones, is_void_turn, code_info


def auto_print_last_expression(code_str: str) -> str:
    """
    Automatically add print statement for the last expression if it's not already printing.

    Args:
        code_str: The original code string.

    Returns:
        Modified code string with auto-print for last expression.
    """
    lines = code_str.strip().splitlines()
    if not lines:
        return code_str

    # Find the last non-empty, non-comment line
    last_line_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            last_line_idx = i
            break

    if last_line_idx == -1:
        return code_str

    last_line = lines[last_line_idx].strip()

    # Check if the last line is a statement that should NOT be auto-printed
    # Skip if it's already a print statement, function definition, control flow, etc.
    skip_patterns = [
        r'^\s*print\s*\(',
        r'^\s*def\s+',
        r'^\s*class\s+',
        r'^\s*if\s+',
        r'^\s*for\s+',
        r'^\s*while\s+',
        r'^\s*with\s+',
        r'^\s*try\s*:',
        r'^\s*except\s*',
        r'^\s*finally\s*:',
        r'^\s*return\s+',
        r'^\s*import\s+',
        r'^\s*from\s+',
        r'^\s*break\s*$',
        r'^\s*continue\s*$',
        r'^\s*pass\s*$',
        r'^\s*raise\s+',
        r'^\s*assert\s+',
        r'^\s*del\s+',
        r'.*:\s*$',  # Lines ending with colon (control flow blocks)
    ]

    for pattern in skip_patterns:
        if re.match(pattern, last_line):
            return code_str

    # Check if it contains assignment (but not comparison)
    # Allow augmented assignments like +=, -=, etc. to be printed
    if '=' in last_line and not any(op in last_line for op in ['==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '//=', '%=', '**=', '&=', '|=', '^=', '>>=', '<<=', '@=']):
        return code_str

    # Try to parse the last line as an expression using ast
    # This is a more robust way to check if it's a valid expression
    try:
        import ast
        # Try to parse as an expression
        ast.parse(last_line, mode='eval')

        # If successful, it's a valid expression that can be printed
        # Get the indentation of the last line
        indent = len(lines[last_line_idx]) - len(lines[last_line_idx].lstrip())
        indent_str = lines[last_line_idx][:indent]

        # Replace the last line with a print statement
        lines[last_line_idx] = f"{indent_str}print({last_line})"
        return "\n".join(lines)
    except SyntaxError:
        # If it's not a valid expression, leave it as is
        return code_str


def count_lines(code_str: str) -> tuple[int, int]:
    """Count the number of lines in the code string.

    Args:
        code_str: The full text of the code.

    Returns:
        total_lines: The total number of lines in the code.
        code_lines: The number of lines in the code excluding empty lines and comment lines.
    """
    lines = code_str.splitlines()
    total_lines = len(lines)

    code_lines = sum(
        1 for line in lines if line.strip() and not line.lstrip().startswith("#")
    )

    return total_lines, code_lines
