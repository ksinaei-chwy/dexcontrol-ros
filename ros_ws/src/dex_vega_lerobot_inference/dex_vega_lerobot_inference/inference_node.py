"""ROS 2 node for observe-only and explicitly armed LeRobot inference."""

from __future__ import annotations

import json
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist, TwistStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger

from dex_vega_lerobot_recorder.camera_sources import (
    CameraValidationError,
    DirectRgbCameraSource,
)
from dex_vega_lerobot_recorder.configuration import RecorderConfig, load_config

from .action_adapter import (
    ActionAdapter,
    ActionSafetyConfig,
    ActionValidationError,
    AdaptedAction,
    load_joint_limits_from_urdf,
)
from .artifact import (
    COMMIT_PATTERN,
    ResolvedArtifact,
    resolve_model_artifact,
    resolve_tokenizer,
)
from .contracts import (
    ACTION_CHUNK_SIZE as PI05_ACTION_CHUNK_SIZE,
    ACTION_DIMENSION,
    BASE_COMMAND_TOPIC,
    COMMAND_TOPICS,
    HAND_RATIO_NAMES,
    MODEL_REPO_ID,
    TASK,
)
from .inference_worker import LatestObservationWorker
from .groot_artifact import GrootArtifactBundle, resolve_groot_artifacts
from .groot_contracts import (
    ACTION_CHUNK_SIZE as GROOT_ACTION_CHUNK_SIZE,
    BASE_MODEL_REPO_ID as GROOT_BASE_MODEL_REPO_ID,
    BASE_MODEL_REVISION as GROOT_BASE_MODEL_REVISION,
    CHECKPOINT_TAG as GROOT_CHECKPOINT_TAG,
    COSMOS_PROCESSOR_REPO_ID as GROOT_COSMOS_PROCESSOR_REPO_ID,
    COSMOS_PROCESSOR_REVISION as GROOT_COSMOS_PROCESSOR_REVISION,
    MODEL_REPO_ID as GROOT_MODEL_REPO_ID,
    MODEL_REVISION as GROOT_MODEL_REVISION,
)
from .groot_policy_runtime import GrootPolicyRuntime
from .observation_adapter import (
    CameraSample,
    ObservationAdapter,
    ObservationSnapshot,
    ObservationValidationError,
    image_message_to_rgb,
)
from .policy_runtime import Pi05PolicyRuntime, PolicyPrediction
from .policy_rpc import PolicyRuntimeClient
from .state_machine import RuntimeState, SafetyStateMachine, StateTransitionError


VALID_MODES = {"observe_only", "dry_run", "replay", "armed"}


@dataclass
class _PredictionEnvelope:
    observation: ObservationSnapshot
    prediction: PolicyPrediction
    actions: deque[np.ndarray]
    received_monotonic_ns: int


@dataclass(frozen=True)
class _BooleanStatus:
    value: bool
    received_monotonic_ns: int


class InferenceNode(Node):
    """Own ROS safety gates while delegating model math to LeRobot 0.6.0."""

    def __init__(self) -> None:
        super().__init__("dex_vega_lerobot_inference")
        self._declare_parameters()
        self._policy_type = str(self.get_parameter("policy_type").value)
        if self._policy_type not in {"pi05", "groot"}:
            raise ValueError("policy_type must be pi05 or groot")
        self._mode = str(self.get_parameter("mode").value)
        if self._mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {self._mode!r}")
        allow_publication = bool(self.get_parameter("allow_command_publication").value)
        self._execution_capable = command_publication_capable(
            self._mode, allow_publication
        )
        self._state_machine = SafetyStateMachine(self._execution_capable)
        self._lifecycle_lock = threading.RLock()
        self._data_lock = threading.Lock()
        self._prediction_lock = threading.Lock()
        self._last_warning: dict[str, float] = {}
        self._latest_camera: CameraSample | None = None
        self._latest_snapshot: ObservationSnapshot | None = None
        self._prediction: _PredictionEnvelope | None = None
        self._pending_worker_error: str | None = None
        self._worker: LatestObservationWorker | None = None
        self._runtime: Any = None
        self._runtime_info: Any = None
        self._model_artifact: Any = None
        self._tokenizer_artifact: Any = None
        self._base_model_artifact: Any = None
        self._processor_artifact: Any = None
        self._model_thread: threading.Thread | None = None
        self._camera_source: DirectRgbCameraSource | None = None
        self._estop_status: _BooleanStatus | None = None
        self._teleop_status: _BooleanStatus | None = None
        self._last_applied_joint_rx_ns = 0
        self._last_applied_base_rx_ns = 0
        self._last_inference_submit_ns = 0
        self._last_action_publish_ns = 0
        self._armed_since_ns = 0
        self._last_action_available_ns = 0
        self._trial_active = False
        self._predictions_received = 0
        self._actions_published = 0
        self._rate_limited_actions = 0
        self._hand_clamped_actions = 0
        self._joint_clamped_actions = 0
        self._base_clamped_actions = 0
        self._last_hand_clamp: dict[str, dict[str, float]] | None = None
        self._last_joint_clamp: dict[str, dict[str, float]] | None = None
        self._shadow_actions_evaluated = 0
        self._shadow_rate_limited_actions = 0
        self._shadow_hand_clamped_actions = 0
        self._shadow_joint_clamped_actions = 0
        self._shadow_base_clamped_actions = 0
        self._shadow_queue_starvations = 0
        self._shadow_stale_queue_actions = 0
        self._shadow_stale_observation_actions = 0
        self._shadow_action_errors = 0
        self._shadow_wait_timeout_recorded = False
        self._last_shadow_error = ""
        self._last_shadow_hand_clamp: dict[str, dict[str, float]] | None = None
        self._last_shadow_joint_clamp: dict[str, dict[str, float]] | None = None
        self._last_validation_error = ""

        self._recorder_config = self._load_recorder_config()
        self._observation_adapter = ObservationAdapter(self._recorder_config)
        self._action_adapter = self._make_action_adapter(self._recorder_config)

        qos_depth = int(self.get_parameter("qos_depth").value)
        self._predicted_action_pub = self.create_publisher(
            Float32MultiArray,
            "/dex_vega_lerobot_inference/predicted_action",
            qos_depth,
        )
        self._status_pub = self.create_publisher(
            String, "/dex_vega_lerobot_inference/status", qos_depth
        )
        self._diagnostics_pub = self.create_publisher(
            DiagnosticArray, "/dex_vega_lerobot_inference/diagnostics", qos_depth
        )
        self._command_publishers = self._create_command_publishers(qos_depth)

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._on_joint_state,
            qos_depth,
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("measured_base_twist_topic").value),
            self._on_measured_base,
            qos_depth,
        )
        self.create_subscription(
            JointState,
            "/dexcontrol/applied_joint_commands",
            self._on_applied_joint,
            qos_depth,
        )
        self.create_subscription(
            TwistStamped,
            "/dexcontrol/applied_base_twist",
            self._on_applied_base,
            qos_depth,
        )
        self.create_subscription(
            Bool,
            "/dexcontrol/estop_state",
            self._on_estop_state,
            qos_depth,
        )
        self.create_subscription(
            String,
            "/dex_pico_teleop/status",
            self._on_teleop_status,
            qos_depth,
        )

        camera_source = str(self.get_parameter("camera_source").value)
        if camera_source == "auto":
            camera_source = "ros_image" if self._mode == "replay" else "direct"
        if camera_source == "ros_image":
            self.create_subscription(
                Image,
                str(self.get_parameter("replay_image_topic").value),
                self._on_replay_image,
                qos_profile_sensor_data,
            )
        elif camera_source == "direct":
            camera = self._recorder_config.head_camera
            self._camera_source = DirectRgbCameraSource(
                width=camera.resolution.width,
                height=camera.resolution.height,
                stream_name=camera.stream_name,
                topic=camera.topic,
                transport=camera.transport,
                rtc_channel=camera.rtc_channel,
                codec=camera.codec,
            )
        else:
            raise ValueError("camera_source must be auto, direct, or ros_image")

        prefix = "/dex_vega_lerobot_inference"
        self.create_service(SetBool, f"{prefix}/arm", self._on_arm)
        self.create_service(Trigger, f"{prefix}/recover", self._on_recover)
        self.create_service(Trigger, f"{prefix}/begin_trial", self._on_begin_trial)
        self.create_service(Trigger, f"{prefix}/end_trial", self._on_end_trial)

        control_hz = float(self.get_parameter("control_frequency_hz").value)
        if control_hz <= 0.0:
            raise ValueError("control_frequency_hz must be positive")
        self._control_timer = self.create_timer(1.0 / control_hz, self._on_control_tick)
        self._status_timer = self.create_timer(1.0, self._publish_status)
        self.add_on_set_parameters_callback(self._on_parameter_change)

        self._state_machine.begin_model_load()
        if bool(self.get_parameter("load_model").value):
            self._model_thread = threading.Thread(
                target=self._load_model,
                daemon=True,
                name=f"{self._policy_type}ModelLoader",
            )
            self._model_thread.start()
        else:
            self._state_machine.model_ready()
            self._state_machine.reason = "model loading disabled for adapter-only validation"

        self.get_logger().info(
            f"{self._policy_type} inference node mode={self._mode}; "
            f"execution_capable={self._execution_capable}; task={TASK!r}"
        )
        if not self._execution_capable:
            self.get_logger().info(
                "No bridge command publishers were created; this process cannot actuate the robot"
            )

    def _declare_parameters(self) -> None:
        workspace = str(Path.cwd().resolve())
        self.declare_parameter("policy_type", "pi05")
        self.declare_parameter("mode", "observe_only")
        self.declare_parameter("allow_command_publication", False)
        self.declare_parameter("execution_readiness_acknowledged", False)
        self.declare_parameter("load_model", True)
        self.declare_parameter("project_root", workspace)
        self.declare_parameter("recorder_config_file", "")
        self.declare_parameter("robot_urdf_path", "")
        self.declare_parameter("model_local_path", "")
        self.declare_parameter("model_repo_id", MODEL_REPO_ID)
        self.declare_parameter("model_revision", "")
        self.declare_parameter("checkpoint_tag", "")
        self.declare_parameter(
            "model_download_directory", "data/models/pi05-dexmate-blue-bird"
        )
        self.declare_parameter("tokenizer_local_path", "")
        self.declare_parameter("tokenizer_revision", "")
        self.declare_parameter(
            "tokenizer_download_directory", "data/models/paligemma-3b-pt-224"
        )
        self.declare_parameter("base_model_local_path", "")
        self.declare_parameter("base_model_revision", "")
        self.declare_parameter("cosmos_processor_local_path", "")
        self.declare_parameter("cosmos_processor_revision", "")
        self.declare_parameter("allow_model_download", False)
        self.declare_parameter("allow_non_commit_revision", False)
        self.declare_parameter("local_files_only", True)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("require_cuda", True)
        self.declare_parameter("require_bfloat16", True)
        self.declare_parameter("runtime_backend", "external")
        self.declare_parameter(
            "policy_server_socket", ".runtime/pi05_policy_server.sock"
        )
        # The project is mounted at /workspace inside the pinned NVIDIA
        # runtime, while the ROS process retains the host workspace path.
        self.declare_parameter("policy_server_project_root", "/workspace")
        self.declare_parameter("policy_server_timeout_seconds", 120.0)
        self.declare_parameter("task", TASK)
        self.declare_parameter("camera_source", "auto")
        self.declare_parameter(
            "replay_image_topic", "/dex_vega_lerobot_inference/replay/head_image"
        )
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter(
            "measured_base_twist_topic", "/dexcontrol/measured_base_twist"
        )
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("control_frequency_hz", 30.0)
        self.declare_parameter("inference_frequency_hz", 5.0)
        self.declare_parameter("execution_horizon", 5)
        self.declare_parameter("maximum_state_age_seconds", 0.10)
        self.declare_parameter("maximum_receive_age_seconds", 0.10)
        self.declare_parameter("maximum_capture_age_seconds", 0.30)
        self.declare_parameter("maximum_transport_delay_seconds", 0.25)
        self.declare_parameter("maximum_synchronization_skew_seconds", 0.10)
        self.declare_parameter("maximum_action_queue_age_seconds", 0.25)
        self.declare_parameter("maximum_observation_to_action_age_seconds", 2.0)
        self.declare_parameter("action_wait_timeout_seconds", 0.50)
        self.declare_parameter("maximum_execution_duration_seconds", 5.0)
        self.declare_parameter("bridge_status_timeout_seconds", 0.25)
        self.declare_parameter("estop_status_timeout_seconds", 0.25)
        self.declare_parameter("teleop_status_timeout_seconds", 1.5)
        self.declare_parameter("require_bridge_status", True)
        self.declare_parameter("require_estop_status", True)
        self.declare_parameter("require_teleop_disabled", True)
        self.declare_parameter("require_bridge_subscribers", True)
        self.declare_parameter("require_exclusive_command_publishers", True)
        self.declare_parameter("max_torso_target_delta_per_cycle", 0.02)
        self.declare_parameter("max_head_target_delta_per_cycle", 0.02)
        self.declare_parameter("max_arm_target_delta_per_cycle", 0.02)
        self.declare_parameter("max_hand_ratio_delta_per_cycle", 0.03)
        self.declare_parameter("max_base_linear_velocity", 0.10)
        self.declare_parameter("max_base_angular_velocity", 0.20)
        self.declare_parameter("max_base_linear_acceleration", 0.30)
        self.declare_parameter("max_base_angular_acceleration", 0.60)

    def _load_recorder_config(self) -> RecorderConfig:
        value = str(self.get_parameter("recorder_config_file").value)
        if value:
            path = Path(value)
        else:
            path = (
                Path(get_package_share_directory("dex_vega_lerobot_recorder"))
                / "config"
                / "dexmate_blue_bird.yaml"
            )
        return load_config(path)

    def _make_action_adapter(self, config: RecorderConfig) -> ActionAdapter:
        value = str(self.get_parameter("robot_urdf_path").value)
        if value:
            urdf_path = Path(value)
        else:
            urdf_path = (
                Path(get_package_share_directory("dexmate_vega_description"))
                / "urdf"
                / "vega_1p_f5d6.package.urdf"
            )
        safety = ActionSafetyConfig(
            max_torso_target_delta_per_cycle=float(
                self.get_parameter("max_torso_target_delta_per_cycle").value
            ),
            max_head_target_delta_per_cycle=float(
                self.get_parameter("max_head_target_delta_per_cycle").value
            ),
            max_arm_target_delta_per_cycle=float(
                self.get_parameter("max_arm_target_delta_per_cycle").value
            ),
            max_hand_ratio_delta_per_cycle=float(
                self.get_parameter("max_hand_ratio_delta_per_cycle").value
            ),
            max_base_linear_velocity=float(
                self.get_parameter("max_base_linear_velocity").value
            ),
            max_base_angular_velocity=float(
                self.get_parameter("max_base_angular_velocity").value
            ),
            max_base_linear_acceleration=float(
                self.get_parameter("max_base_linear_acceleration").value
            ),
            max_base_angular_acceleration=float(
                self.get_parameter("max_base_angular_acceleration").value
            ),
        )
        return ActionAdapter(config, load_joint_limits_from_urdf(urdf_path), safety)

    def _create_command_publishers(self, qos_depth: int) -> dict[str, Any] | None:
        if not self._execution_capable:
            return None
        publishers = {
            component: self.create_publisher(JointState, topic, qos_depth)
            for component, topic in COMMAND_TOPICS.items()
        }
        publishers["base"] = self.create_publisher(Twist, BASE_COMMAND_TOPIC, qos_depth)
        return publishers

    def _load_model(self) -> None:
        try:
            root = Path(str(self.get_parameter("project_root").value)).expanduser().resolve()
            allow_download = bool(self.get_parameter("allow_model_download").value)
            allow_non_commit = bool(self.get_parameter("allow_non_commit_revision").value)
            local_only = bool(self.get_parameter("local_files_only").value)
            backend = str(self.get_parameter("runtime_backend").value)
            if backend not in {"external", "embedded"}:
                raise ValueError("runtime_backend must be external or embedded")
            runtime: Any = None
            if self._policy_type == "groot":
                if allow_non_commit:
                    raise ValueError("GR00T deployment never accepts tags or mutable revisions")
                if backend == "external":
                    duplicated_selection = [
                        name
                        for name in (
                            "model_local_path",
                            "model_revision",
                            "checkpoint_tag",
                            "base_model_local_path",
                            "base_model_revision",
                            "cosmos_processor_local_path",
                            "cosmos_processor_revision",
                        )
                        if str(self.get_parameter(name).value)
                    ]
                    if duplicated_selection:
                        raise ValueError(
                            "external GR00T artifact selection belongs only to the "
                            "policy server; remove ROS parameters: "
                            + ", ".join(duplicated_selection)
                        )
                    socket_path = root / str(
                        self.get_parameter("policy_server_socket").value
                    )
                    runtime = PolicyRuntimeClient(
                        socket_path,
                        timeout_seconds=float(
                            self.get_parameter("policy_server_timeout_seconds").value
                        ),
                    )
                    artifacts = resolve_external_groot_artifacts(
                        runtime.info,
                        local_project_root=root,
                        server_project_root=Path(
                            str(self.get_parameter("policy_server_project_root").value)
                        ),
                    )
                else:
                    model_local = str(self.get_parameter("model_local_path").value)
                    base_local = str(
                        self.get_parameter("base_model_local_path").value
                    )
                    processor_local = str(
                        self.get_parameter("cosmos_processor_local_path").value
                    )
                    if not model_local or not base_local or not processor_local:
                        raise ValueError(
                            "embedded GR00T requires model_local_path, "
                            "base_model_local_path, and cosmos_processor_local_path"
                        )
                    artifacts = resolve_groot_artifacts(
                        project_root=root,
                        model_local_path=model_local,
                        base_model_local_path=base_local,
                        cosmos_processor_local_path=processor_local,
                        model_revision=(
                            str(self.get_parameter("model_revision").value)
                            or GROOT_MODEL_REVISION
                        ),
                        base_model_revision=(
                            str(self.get_parameter("base_model_revision").value)
                            or GROOT_BASE_MODEL_REVISION
                        ),
                        cosmos_processor_revision=(
                            str(
                                self.get_parameter(
                                    "cosmos_processor_revision"
                                ).value
                            )
                            or GROOT_COSMOS_PROCESSOR_REVISION
                        ),
                        checkpoint_tag=(
                            str(self.get_parameter("checkpoint_tag").value)
                            or GROOT_CHECKPOINT_TAG
                        ),
                        allow_download=allow_download,
                        local_files_only=local_only,
                    )
                model = artifacts.model
                tokenizer = None
                base_model = artifacts.base_model
                processor = artifacts.cosmos_processor
            elif backend == "external":
                duplicated_selection = [
                    name
                    for name in (
                        "model_local_path",
                        "model_revision",
                        "checkpoint_tag",
                        "tokenizer_local_path",
                        "tokenizer_revision",
                    )
                    if str(self.get_parameter(name).value)
                ]
                if duplicated_selection:
                    raise ValueError(
                        "external PI0.5 checkpoint/tokenizer selection belongs only "
                        "to the policy server; remove ROS parameters: "
                        + ", ".join(duplicated_selection)
                    )
                socket_path = root / str(
                    self.get_parameter("policy_server_socket").value
                )
                runtime = PolicyRuntimeClient(
                    socket_path,
                    timeout_seconds=float(
                        self.get_parameter("policy_server_timeout_seconds").value
                    ),
                )
                model, tokenizer = resolve_external_pi05_artifacts(
                    runtime.info,
                    local_project_root=root,
                    server_project_root=Path(
                        str(self.get_parameter("policy_server_project_root").value)
                    ),
                )
                base_model = None
                processor = None
            else:
                model_local = str(self.get_parameter("model_local_path").value)
                model_download = root / str(
                    self.get_parameter("model_download_directory").value
                )
                model = resolve_model_artifact(
                    project_root=root,
                    local_path=model_local or None,
                    repo_id=str(self.get_parameter("model_repo_id").value) or None,
                    revision=str(self.get_parameter("model_revision").value) or None,
                    download_directory=model_download,
                    checkpoint_tag=(
                        str(self.get_parameter("checkpoint_tag").value) or None
                    ),
                    allow_download=allow_download,
                    allow_non_commit_revision=allow_non_commit,
                    local_files_only=local_only,
                )
                tokenizer_local = str(self.get_parameter("tokenizer_local_path").value)
                tokenizer_download = root / str(
                    self.get_parameter("tokenizer_download_directory").value
                )
                tokenizer = resolve_tokenizer(
                    project_root=root,
                    local_path=tokenizer_local or None,
                    revision=(
                        str(self.get_parameter("tokenizer_revision").value) or None
                    ),
                    download_directory=tokenizer_download,
                    allow_download=allow_download,
                    allow_non_commit_revision=allow_non_commit,
                    local_files_only=local_only,
                )
                base_model = None
                processor = None
            if backend == "external":
                if runtime is None:
                    socket_path = root / str(
                        self.get_parameter("policy_server_socket").value
                    )
                    runtime = PolicyRuntimeClient(
                        socket_path,
                        timeout_seconds=float(
                            self.get_parameter("policy_server_timeout_seconds").value
                        ),
                    )
                self._validate_external_runtime_identity(
                    runtime,
                    model,
                    tokenizer,
                    base_model=base_model,
                    processor=processor,
                    expected_policy_type=self._policy_type,
                    local_project_root=root,
                    server_project_root=Path(
                        str(self.get_parameter("policy_server_project_root").value)
                    ),
                )
            elif backend == "embedded":
                common = {
                    "project_root": root,
                    "device": str(self.get_parameter("device").value),
                    "require_cuda": bool(self.get_parameter("require_cuda").value),
                    "require_bfloat16": bool(
                        self.get_parameter("require_bfloat16").value
                    ),
                }
                if self._policy_type == "groot":
                    runtime = GrootPolicyRuntime(artifacts=artifacts, **common)
                else:
                    runtime = Pi05PolicyRuntime(
                        model=model,
                        tokenizer=tokenizer,
                        **common,
                    )
            worker = LatestObservationWorker(
                runtime,
                on_result=self._on_inference_result,
                on_error=self._on_inference_error,
            )
            with self._lifecycle_lock:
                self._runtime = runtime
                self._runtime_info = runtime.info
                self._model_artifact = model
                self._tokenizer_artifact = tokenizer
                self._base_model_artifact = base_model
                self._processor_artifact = processor
                self._worker = worker
                self._state_machine.model_ready()
            self.get_logger().info(
                f"{self._policy_type} model loaded from local artifact; "
                f"commit={runtime.info.model_commit or 'unrecorded-local'}; "
                f"checkpoint={runtime.info.checkpoint_tag or 'unspecified'}; "
                f"load={runtime.info.load_seconds:.2f}s"
            )
        except Exception as exc:  # noqa: BLE001 - artifact/CUDA boundary
            with self._lifecycle_lock:
                self._state_machine.fault(f"model load failed: {exc}")
            self.get_logger().error(f"{self._policy_type} model load failed: {exc}")

    @staticmethod
    def _validate_external_runtime_identity(
        runtime: Any,
        model: Any,
        tokenizer: Any | None,
        *,
        base_model: Any | None,
        processor: Any | None,
        expected_policy_type: str,
        local_project_root: Path,
        server_project_root: Path,
    ) -> None:
        info = runtime.info
        if info.policy_type != expected_policy_type:
            raise RuntimeError(
                "policy server type differs from the configured policy type"
            )
        expected_chunk_size = (
            GROOT_ACTION_CHUNK_SIZE
            if expected_policy_type == "groot"
            else PI05_ACTION_CHUNK_SIZE
        )
        if info.action_chunk_size != expected_chunk_size:
            raise RuntimeError("policy server action chunk size differs from the pin")
        if info.action_dimension != ACTION_DIMENSION:
            raise RuntimeError("policy server physical action dimension is not 27")
        if not project_relative_paths_match(
            info.model_path,
            server_project_root,
            model.local_path,
            local_project_root,
        ):
            raise RuntimeError("policy server loaded a different model directory")
        if model.resolved_commit and info.model_commit != model.resolved_commit:
            raise RuntimeError("policy server model commit differs from the pinned artifact")
        if info.checkpoint_tag != model.checkpoint_tag:
            raise RuntimeError("policy server checkpoint tag differs from the model manifest")
        if tokenizer is not None:
            if not project_relative_paths_match(
                info.tokenizer_path,
                server_project_root,
                tokenizer.local_path,
                local_project_root,
            ):
                raise RuntimeError("policy server loaded a different tokenizer directory")
            if tokenizer.resolved_commit and info.tokenizer_commit != tokenizer.resolved_commit:
                raise RuntimeError(
                    "policy server tokenizer commit differs from the pinned artifact"
                )
        if base_model is not None:
            if not project_relative_paths_match(
                info.base_model_path,
                server_project_root,
                base_model.local_path,
                local_project_root,
            ):
                raise RuntimeError("policy server loaded a different GR00T base directory")
            if info.base_model_commit != base_model.resolved_commit:
                raise RuntimeError("policy server GR00T base commit differs from the pin")
        if processor is not None:
            if not project_relative_paths_match(
                info.processor_path,
                server_project_root,
                processor.local_path,
                local_project_root,
            ):
                raise RuntimeError("policy server loaded a different Cosmos processor directory")
            if info.processor_commit != processor.resolved_commit:
                raise RuntimeError("policy server Cosmos processor commit differs from the pin")

    def _on_joint_state(self, message: JointState) -> None:
        try:
            self._observation_adapter.update_measured_joints(
                message.name,
                message.position,
                self._stamp_or_now(message),
            )
        except ValueError as exc:
            self._warn_throttled("joint_state", f"invalid measured joint state: {exc}")

    def _on_measured_base(self, message: TwistStamped) -> None:
        try:
            self._observation_adapter.update_measured_base(
                (
                    message.twist.linear.x,
                    message.twist.linear.y,
                    message.twist.angular.z,
                ),
                self._stamp_or_now(message),
            )
        except ValueError as exc:
            self._warn_throttled("base_state", f"invalid measured base velocity: {exc}")

    def _on_applied_joint(self, _message: JointState) -> None:
        self._last_applied_joint_rx_ns = time.monotonic_ns()

    def _on_applied_base(self, _message: TwistStamped) -> None:
        self._last_applied_base_rx_ns = time.monotonic_ns()

    def _on_estop_state(self, message: Bool) -> None:
        self._estop_status = _BooleanStatus(bool(message.data), time.monotonic_ns())
        if message.data:
            self._safe_stop("bridge reports E-stop active", estop=True)

    def _on_teleop_status(self, message: String) -> None:
        try:
            data = json.loads(message.data)
            enabled = bool(data["enabled"])
        except (KeyError, TypeError, json.JSONDecodeError):
            self._warn_throttled("teleop_status", "invalid /dex_pico_teleop/status payload")
            return
        self._teleop_status = _BooleanStatus(enabled, time.monotonic_ns())
        if enabled and self._state_machine.may_publish:
            self._safe_stop("Pico teleop became enabled while inference was armed", fault=True)

    def _on_replay_image(self, message: Image) -> None:
        try:
            rgb = image_message_to_rgb(message)
            receive_stamp = self.get_clock().now().nanoseconds
            self._set_camera(
                CameraSample(
                    rgb=rgb,
                    source_stamp_ns=self._stamp_or_now(message),
                    receive_stamp_ns=receive_stamp,
                    transport_delay_seconds=0.0,
                )
            )
        except ObservationValidationError as exc:
            self._warn_throttled("replay_image", f"invalid replay image: {exc}")

    def _poll_direct_camera(self, now_ns: int) -> None:
        if self._camera_source is None:
            return
        validation = self._recorder_config.validation
        try:
            frame = self._camera_source.snapshot(
                now_ns,
                maximum_receive_age_seconds=validation.maximum_receive_age_seconds,
                maximum_capture_age_seconds=validation.maximum_capture_age_seconds,
                maximum_transport_delay_seconds=validation.maximum_transport_delay_seconds,
            )
            self._set_camera(
                CameraSample(
                    frame.rgb,
                    frame.source_stamp_ns,
                    frame.receive_stamp_ns,
                    frame.transport_delay_seconds,
                )
            )
        except CameraValidationError as exc:
            self._last_validation_error = str(exc)
            self._warn_throttled("camera", f"camera observation unavailable: {exc}")

    def _set_camera(self, camera: CameraSample) -> None:
        with self._data_lock:
            current = self._latest_camera
            if current is None or camera.source_stamp_ns > current.source_stamp_ns:
                self._latest_camera = camera

    def _on_control_tick(self) -> None:
        now_ros_ns = self.get_clock().now().nanoseconds
        now_mono_ns = time.monotonic_ns()
        if self._pending_worker_error:
            error = self._pending_worker_error
            self._pending_worker_error = None
            self._safe_stop(f"model inference failed: {error}", fault=True)
            return
        if self._estop_status is not None and self._estop_status.value:
            self._safe_stop("bridge reports E-stop active", estop=True)
            return
        self._poll_direct_camera(now_ros_ns)
        self._maybe_submit_observation(now_ros_ns, now_mono_ns)
        if self._state_machine.may_publish:
            gate_failure = self._continuous_gate_failure(now_mono_ns)
            if gate_failure:
                self._safe_stop(gate_failure, fault=True)
                return
            self._execute_next_action(now_mono_ns)
        elif self._mode == "dry_run":
            self._evaluate_next_shadow_action(now_mono_ns)

    def _maybe_submit_observation(self, now_ros_ns: int, now_mono_ns: int) -> None:
        with self._data_lock:
            camera = self._latest_camera
        if camera is None:
            self._last_validation_error = "missing head camera"
            if self._state_machine.may_publish:
                self._safe_stop("missing head camera", fault=True)
            return
        try:
            snapshot = self._observation_adapter.snapshot(
                camera,
                now_ros_ns,
                maximum_state_age_seconds=float(
                    self.get_parameter("maximum_state_age_seconds").value
                ),
                maximum_receive_age_seconds=float(
                    self.get_parameter("maximum_receive_age_seconds").value
                ),
                maximum_capture_age_seconds=float(
                    self.get_parameter("maximum_capture_age_seconds").value
                ),
                maximum_transport_delay_seconds=float(
                    self.get_parameter("maximum_transport_delay_seconds").value
                ),
                maximum_synchronization_skew_seconds=float(
                    self.get_parameter("maximum_synchronization_skew_seconds").value
                ),
            )
        except ObservationValidationError as exc:
            self._last_validation_error = str(exc)
            if self._state_machine.may_publish and "duplicate" not in str(exc):
                self._safe_stop(f"invalid inference observation: {exc}", fault=True)
            return
        self._latest_snapshot = snapshot
        self._last_validation_error = ""
        worker = self._worker
        frequency = float(self.get_parameter("inference_frequency_hz").value)
        if worker is None or frequency <= 0.0:
            return
        if now_mono_ns - self._last_inference_submit_ns < int(1e9 / frequency):
            return
        worker.submit(snapshot)
        self._last_inference_submit_ns = now_mono_ns

    def _on_inference_result(
        self, observation: ObservationSnapshot, result: object
    ) -> None:
        if not isinstance(result, PolicyPrediction):
            self._on_inference_error(TypeError("runtime returned an unknown prediction type"))
            return
        try:
            chunk = self._action_adapter.validate_chunk(result.actions)
            horizon = int(self.get_parameter("execution_horizon").value)
            if horizon <= 0 or horizon > len(chunk):
                raise ActionValidationError(
                    f"execution_horizon must be within [1, {len(chunk)}]"
                )
        except Exception as exc:  # noqa: BLE001 - worker boundary
            self._on_inference_error(exc)
            return
        envelope = _PredictionEnvelope(
            observation=observation,
            prediction=result,
            actions=deque(row.copy() for row in chunk[:horizon]),
            received_monotonic_ns=time.monotonic_ns(),
        )
        with self._prediction_lock:
            self._prediction = envelope
        self._predictions_received += 1
        self._last_action_available_ns = envelope.received_monotonic_ns
        self._shadow_wait_timeout_recorded = False
        message = Float32MultiArray()
        message.data = [float(value) for value in chunk[0]]
        self._predicted_action_pub.publish(message)

    def _on_inference_error(self, error: Exception) -> None:
        self._pending_worker_error = str(error)

    def _execute_next_action(self, now_mono_ns: int) -> None:
        envelope, action = self._pop_next_action()
        if envelope is None or action is None:
            self._publish_zero_base()
            reference_ns = max(self._armed_since_ns, self._last_action_available_ns)
            timeout = float(self.get_parameter("action_wait_timeout_seconds").value)
            if reference_ns and (now_mono_ns - reference_ns) / 1e9 > timeout:
                self._safe_stop("no fresh action chunk available", fault=True)
            return

        age_failure = self._prediction_age_gate_failure(envelope, now_mono_ns)
        if age_failure:
            self._safe_stop(age_failure, fault=True)
            return
        current = self._latest_snapshot
        if current is None:
            self._safe_stop("measured state unavailable during execution", fault=True)
            return
        control_hz = float(self.get_parameter("control_frequency_hz").value)
        try:
            adapted = self._action_adapter.adapt(
                action,
                current.state,
                cycle_seconds=1.0 / control_hz,
            )
        except ActionValidationError as exc:
            self._safe_stop(f"unsafe policy action: {exc}", fault=True)
            return
        self._publish_action(adapted)
        self._state_machine.executing()

    def _pop_next_action(
        self,
    ) -> tuple[_PredictionEnvelope | None, np.ndarray | None]:
        with self._prediction_lock:
            envelope = self._prediction
            if envelope is not None and envelope.actions:
                action = envelope.actions.popleft()
                if not envelope.actions:
                    self._prediction = None
            else:
                action = None
        return envelope, action

    def _prediction_age_gate_failure(
        self,
        envelope: _PredictionEnvelope,
        now_mono_ns: int,
    ) -> str:
        queue_age = (now_mono_ns - envelope.received_monotonic_ns) / 1e9
        total_age = envelope.prediction.timings.observation_to_result_seconds + queue_age
        if queue_age > float(self.get_parameter("maximum_action_queue_age_seconds").value):
            return f"stale action queue ({queue_age:.3f}s)"
        if total_age > float(
            self.get_parameter("maximum_observation_to_action_age_seconds").value
        ):
            return f"stale observation-to-action path ({total_age:.3f}s)"
        return ""

    def _evaluate_next_shadow_action(self, now_mono_ns: int) -> None:
        """Exercise the armed queue and action adapter without ROS command output."""
        envelope, action = self._pop_next_action()
        if envelope is None or action is None:
            reference_ns = self._last_action_available_ns
            timeout = float(self.get_parameter("action_wait_timeout_seconds").value)
            if (
                reference_ns
                and (now_mono_ns - reference_ns) / 1e9 > timeout
                and not self._shadow_wait_timeout_recorded
            ):
                self._shadow_queue_starvations += 1
                self._shadow_wait_timeout_recorded = True
                self._last_shadow_error = "no fresh action chunk available"
            return

        age_failure = self._prediction_age_gate_failure(envelope, now_mono_ns)
        if age_failure:
            if age_failure.startswith("stale action queue"):
                self._shadow_stale_queue_actions += 1
            else:
                self._shadow_stale_observation_actions += 1
            self._last_shadow_error = age_failure
            return
        current = self._latest_snapshot
        if current is None:
            self._shadow_action_errors += 1
            self._last_shadow_error = "measured state unavailable during shadow evaluation"
            return
        control_hz = float(self.get_parameter("control_frequency_hz").value)
        try:
            adapted = self._action_adapter.adapt(
                action,
                current.state,
                cycle_seconds=1.0 / control_hz,
            )
        except ActionValidationError as exc:
            self._shadow_action_errors += 1
            self._last_shadow_error = f"unsafe policy action: {exc}"
            return

        self._shadow_actions_evaluated += 1
        self._shadow_rate_limited_actions += int(adapted.rate_limited)
        self._shadow_hand_clamped_actions += int(adapted.hand_clamped)
        self._shadow_joint_clamped_actions += int(adapted.joint_clamped)
        self._shadow_base_clamped_actions += int(adapted.base_clamped)
        if adapted.hand_clamped:
            raw_ratios = adapted.policy_action[20:24]
            bounded_ratios = np.clip(raw_ratios, 0.0, 1.0)
            self._last_shadow_hand_clamp = {
                name: {"raw": float(raw), "clamped": float(bounded)}
                for name, raw, bounded in zip(
                    HAND_RATIO_NAMES, raw_ratios, bounded_ratios, strict=True
                )
                if raw != bounded
            }
        if adapted.joint_clamped:
            self._last_shadow_joint_clamp = {
                name: {"raw": raw, "clamped": bounded}
                for name, (raw, bounded) in adapted.joint_clamps.items()
            }
        self._last_shadow_error = ""

    def _publish_action(self, action: AdaptedAction) -> None:
        publishers = self._command_publishers
        if publishers is None or not self._state_machine.may_publish:
            raise RuntimeError("command publication attempted outside an armed runtime")
        if action.joint_clamped:
            clamped = {
                name: {"raw": raw, "clamped": bounded}
                for name, (raw, bounded) in action.joint_clamps.items()
            }
            self._last_joint_clamp = clamped
            details = ", ".join(
                f"{name}={values['raw']:.17g}->{values['clamped']:.17g}"
                for name, values in clamped.items()
            )
            self._warn_throttled(
                "joint_limit_clamp",
                "Clamped finite joint target(s) to authoritative URDF limits: "
                + details,
            )
        if action.hand_clamped:
            raw_ratios = action.policy_action[20:24]
            bounded_ratios = np.clip(raw_ratios, 0.0, 1.0)
            clamped = {
                name: {"raw": float(raw), "clamped": float(bounded)}
                for name, raw, bounded in zip(
                    HAND_RATIO_NAMES, raw_ratios, bounded_ratios, strict=True
                )
                if raw != bounded
            }
            self._last_hand_clamp = clamped
            details = ", ".join(
                f"{name}={values['raw']:.6f}->{values['clamped']:.6f}"
                for name, values in clamped.items()
            )
            self._warn_throttled(
                "hand_ratio_clamp",
                "Clamped out-of-range postprocessed hand ratio(s) to [0, 1]: "
                + details,
            )
        stamp = self.get_clock().now().to_msg()
        for component, (names, positions) in action.component_positions.items():
            message = JointState()
            message.header.stamp = stamp
            message.name = [str(name) for name in names]
            message.position = [float(value) for value in positions]
            publishers[component].publish(message)
        twist = Twist()
        twist.linear.x = float(action.base_twist[0])
        twist.linear.y = float(action.base_twist[1])
        twist.angular.z = float(action.base_twist[2])
        publishers["base"].publish(twist)
        self._last_action_publish_ns = time.monotonic_ns()
        self._actions_published += 1
        self._rate_limited_actions += int(action.rate_limited)
        self._hand_clamped_actions += int(action.hand_clamped)
        self._joint_clamped_actions += int(action.joint_clamped)
        self._base_clamped_actions += int(action.base_clamped)

    def _publish_zero_base(self) -> None:
        if self._command_publishers is None:
            return
        message = Twist()
        self._command_publishers["base"].publish(message)

    def _on_arm(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        if not request.data:
            self._safe_stop("explicitly disarmed", fault=False)
            response.success = True
            response.message = "disarmed; base zeroed and all action queues reset"
            return response
        reason = (
            "begin_trial must be called before arming"
            if not self._trial_active
            else self._arming_gate_failure(time.monotonic_ns())
        )
        try:
            self._state_machine.arm(not reason, reason)
        except StateTransitionError as exc:
            response.success = False
            response.message = str(exc)
            return response
        self._reset_queues("arming")
        self._armed_since_ns = time.monotonic_ns()
        self._publish_zero_base()
        response.success = True
        response.message = "armed; waiting for a fresh postprocessed action chunk"
        return response

    def _on_recover(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        reason = self._arming_gate_failure(time.monotonic_ns(), require_ready_state=False)
        try:
            self._state_machine.recover(not reason, reason)
        except StateTransitionError as exc:
            response.success = False
            response.message = str(exc)
            return response
        self._reset_queues("recovery")
        response.success = True
        response.message = "recovered to non-executing state; explicit re-arm required"
        return response

    def _on_begin_trial(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if self._state_machine.state not in {RuntimeState.READY, RuntimeState.OBSERVE_ONLY}:
            response.success = False
            response.message = f"cannot begin trial in {self._state_machine.state.value}"
            return response
        self._trial_active = True
        self._reset_queues("trial begin")
        response.success = True
        response.message = "trial queues reset; runtime remains disarmed"
        return response

    def _on_end_trial(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        self._trial_active = False
        self._safe_stop("trial ended", fault=False)
        response.success = True
        response.message = "trial ended; runtime disarmed and queues reset"
        return response

    def _arming_gate_failure(
        self, now_ns: int, *, require_ready_state: bool = True
    ) -> str:
        if not self._execution_capable or self._command_publishers is None:
            return "process was not launched with mode=armed and allow_command_publication=true"
        if require_ready_state and self._state_machine.state != RuntimeState.READY:
            return f"runtime state is {self._state_machine.state.value}, not READY"
        if self._worker is None or self._runtime is None:
            return "model runtime is not loaded"
        if self._predictions_received < 1:
            return "no successful observe-only warm-up prediction has completed"
        model_commit = getattr(self._model_artifact, "resolved_commit", None)
        if not model_commit or not COMMIT_PATTERN.fullmatch(model_commit):
            return "local model has no verified immutable Hub commit manifest"
        if self._policy_type == "groot":
            if not bool(
                self.get_parameter("execution_readiness_acknowledged").value
            ):
                return (
                    "GR00T execution readiness is not acknowledged; complete and "
                    "review observe-only validation first"
                )
            base_commit = getattr(self._base_model_artifact, "resolved_commit", None)
            if not base_commit or not COMMIT_PATTERN.fullmatch(base_commit):
                return "local GR00T base has no verified immutable Hub commit manifest"
            processor_commit = getattr(
                self._processor_artifact, "resolved_commit", None
            )
            if not processor_commit or not COMMIT_PATTERN.fullmatch(processor_commit):
                return "local Cosmos processor has no verified immutable Hub commit manifest"
        else:
            tokenizer_commit = getattr(self._tokenizer_artifact, "resolved_commit", None)
            if not tokenizer_commit or not COMMIT_PATTERN.fullmatch(tokenizer_commit):
                return "local tokenizer has no verified immutable Hub commit manifest"
        if str(self.get_parameter("task").value) != TASK:
            return "task parameter does not exactly match the training task"
        maximum_duration = float(
            self.get_parameter("maximum_execution_duration_seconds").value
        )
        duration_failure = execution_duration_gate_failure(
            0, now_ns, maximum_duration
        )
        if duration_failure:
            return duration_failure
        if self._latest_snapshot is None:
            return "no validated state/image observation is available"
        observation_age = (now_ns - self._latest_snapshot.created_monotonic_ns) / 1e9
        if observation_age > float(self.get_parameter("maximum_state_age_seconds").value):
            return f"latest validated observation is stale ({observation_age:.3f}s)"
        continuous = self._continuous_gate_failure(now_ns)
        if continuous:
            return continuous
        if bool(self.get_parameter("require_bridge_subscribers").value):
            missing = [
                component
                for component, publisher in self._command_publishers.items()
                if publisher.get_subscription_count() < 1
            ]
            if missing:
                return "bridge command subscribers are missing for: " + ", ".join(missing)
        return ""

    def _continuous_gate_failure(self, now_ns: int) -> str:
        duration_failure = execution_duration_gate_failure(
            self._armed_since_ns,
            now_ns,
            float(self.get_parameter("maximum_execution_duration_seconds").value),
        )
        if duration_failure:
            return duration_failure
        if self._latest_snapshot is None:
            return "no validated state/image observation is available"
        observation_age = (now_ns - self._latest_snapshot.created_monotonic_ns) / 1e9
        maximum_age = float(self.get_parameter("maximum_state_age_seconds").value)
        if observation_age > maximum_age:
            return f"latest validated observation is stale ({observation_age:.3f}s)"
        if bool(self.get_parameter("require_estop_status").value):
            timeout = float(self.get_parameter("estop_status_timeout_seconds").value)
            if self._estop_status is None:
                return "no /dexcontrol/estop_state received"
            age = (now_ns - self._estop_status.received_monotonic_ns) / 1e9
            if age > timeout:
                return f"E-stop status is stale ({age:.3f}s)"
            if self._estop_status.value:
                return "bridge E-stop is active"
        if bool(self.get_parameter("require_teleop_disabled").value):
            timeout = float(self.get_parameter("teleop_status_timeout_seconds").value)
            if self._teleop_status is None:
                return "no Pico teleop disabled status received"
            age = (now_ns - self._teleop_status.received_monotonic_ns) / 1e9
            if age > timeout:
                return f"Pico teleop status is stale ({age:.3f}s)"
            if self._teleop_status.value:
                return "Pico teleop is enabled"
        publisher_failure = self._exclusive_command_publisher_failure()
        if publisher_failure:
            return publisher_failure
        if bool(self.get_parameter("require_bridge_status").value):
            timeout = float(self.get_parameter("bridge_status_timeout_seconds").value)
            for label, stamp in (
                ("applied joint telemetry", self._last_applied_joint_rx_ns),
                ("applied base telemetry", self._last_applied_base_rx_ns),
            ):
                if stamp <= 0:
                    return f"no bridge {label} received"
                age = (now_ns - stamp) / 1e9
                if age > timeout:
                    return f"bridge {label} is stale ({age:.3f}s)"
        return ""

    def _exclusive_command_publisher_failure(self) -> str:
        if not bool(
            self.get_parameter("require_exclusive_command_publishers").value
        ):
            return ""
        if self._command_publishers is None:
            return "command publishers were not constructed"
        topic_counts = {
            **{
                topic: self.count_publishers(topic)
                for topic in COMMAND_TOPICS.values()
            },
            BASE_COMMAND_TOPIC: self.count_publishers(BASE_COMMAND_TOPIC),
        }
        conflicts = unexpected_command_publisher_counts(topic_counts)
        if not conflicts:
            return ""
        details = ", ".join(
            f"{topic}={count}" for topic, count in conflicts.items()
        )
        return "command topics must have exactly the inference publisher; " + details

    def _safe_stop(self, reason: str, *, fault: bool = False, estop: bool = False) -> None:
        with self._lifecycle_lock:
            was_executing = self._state_machine.may_publish
            self._armed_since_ns = 0
            self._reset_queues(reason)
            if self._command_publishers is not None:
                self._publish_zero_base()
            if estop:
                self._state_machine.estop(reason)
            elif fault:
                self._state_machine.fault(reason)
            else:
                self._state_machine.disarm(reason)
            if was_executing or fault or estop:
                self.get_logger().warn(
                    f"Inference stopped: {reason}. Joint command publication ceased; "
                    "the bridge retains its last cached joint targets and received zero cmd_vel."
                )

    def _reset_queues(self, reason: str) -> None:
        with self._prediction_lock:
            self._prediction = None
        self._action_adapter.reset()
        if self._worker is not None:
            self._worker.reset()
        self._last_action_available_ns = 0
        self.get_logger().debug(f"policy/action queues reset: {reason}")

    def _on_parameter_change(self, parameters: list[Any]) -> SetParametersResult:
        # Every parameter contributes to artifact selection, observation
        # validity, action bounds, publication authority, or diagnostics. A
        # running process therefore has one immutable, reviewable safety
        # configuration; changes require a restart and repeat of all gates.
        changed = sorted(parameter.name for parameter in parameters)
        if changed:
            self._safe_stop("immutable runtime parameter change requested", fault=True)
            return SetParametersResult(
                successful=False,
                reason="restart is required to change: " + ", ".join(changed),
            )
        return SetParametersResult(successful=True)

    def _publish_status(self) -> None:
        now_monotonic_ns = time.monotonic_ns()
        worker_stats = self._worker.stats() if self._worker is not None else None
        prediction = None
        with self._prediction_lock:
            if self._prediction is not None:
                prediction = self._prediction.prediction
                queue_size = len(self._prediction.actions)
                queue_age_seconds = (
                    now_monotonic_ns - self._prediction.received_monotonic_ns
                ) / 1e9
            else:
                queue_size = 0
                queue_age_seconds = None
        observation = self._latest_snapshot
        status = {
            "state": self._state_machine.state.value,
            "reason": self._state_machine.reason,
            "mode": self._mode,
            "policy_type": self._policy_type,
            "execution_capable": self._execution_capable,
            "execution_readiness_acknowledged": bool(
                self.get_parameter("execution_readiness_acknowledged").value
            ),
            "trial_active": self._trial_active,
            "task": TASK,
            "model_commit": getattr(self._runtime_info, "model_commit", None),
            "checkpoint_tag": getattr(self._runtime_info, "checkpoint_tag", None),
            "tokenizer_commit": getattr(self._runtime_info, "tokenizer_commit", None),
            "base_model_commit": getattr(
                self._runtime_info, "base_model_commit", None
            ),
            "cosmos_processor_commit": getattr(
                self._runtime_info, "processor_commit", None
            ),
            "runtime_action_chunk_size": getattr(
                self._runtime_info, "action_chunk_size", None
            ),
            "runtime_action_dimension": getattr(
                self._runtime_info, "action_dimension", None
            ),
            "action_queue_size": queue_size,
            "action_queue_age_seconds": queue_age_seconds,
            "observation_to_action_age_seconds": (
                prediction.timings.observation_to_result_seconds + queue_age_seconds
                if prediction is not None and queue_age_seconds is not None
                else None
            ),
            "state_stamp_ns": (
                observation.state_stamp_ns if observation is not None else None
            ),
            "camera_stamp_ns": (
                observation.camera_stamp_ns if observation is not None else None
            ),
            "camera_receive_stamp_ns": (
                observation.receive_stamp_ns if observation is not None else None
            ),
            "observation_created_stamp_ns": (
                observation.created_stamp_ns if observation is not None else None
            ),
            "latest_observation_age_seconds": (
                (now_monotonic_ns - observation.created_monotonic_ns) / 1e9
                if observation is not None
                else None
            ),
            "state_age_at_capture_seconds": (
                observation.state_age_seconds if observation is not None else None
            ),
            "camera_capture_age_at_snapshot_seconds": (
                observation.camera_capture_age_seconds if observation is not None else None
            ),
            "camera_receive_age_at_snapshot_seconds": (
                observation.camera_receive_age_seconds if observation is not None else None
            ),
            "state_camera_skew_seconds": (
                observation.synchronization_skew_seconds
                if observation is not None
                else None
            ),
            "maximum_synchronization_skew_seconds": float(
                self.get_parameter("maximum_synchronization_skew_seconds").value
            ),
            "control_frequency_hz": float(
                self.get_parameter("control_frequency_hz").value
            ),
            "inference_frequency_hz": float(
                self.get_parameter("inference_frequency_hz").value
            ),
            "execution_horizon": int(self.get_parameter("execution_horizon").value),
            "maximum_execution_duration_seconds": float(
                self.get_parameter("maximum_execution_duration_seconds").value
            ),
            "execution_elapsed_seconds": (
                (now_monotonic_ns - self._armed_since_ns) / 1e9
                if self._armed_since_ns
                else None
            ),
            "predictions_received": self._predictions_received,
            "actions_published": self._actions_published,
            "rate_limited_actions": self._rate_limited_actions,
            "hand_clamped_actions": self._hand_clamped_actions,
            "last_hand_clamp": self._last_hand_clamp,
            "joint_clamped_actions": self._joint_clamped_actions,
            "last_joint_clamp": self._last_joint_clamp,
            "base_clamped_actions": self._base_clamped_actions,
            "shadow": {
                "enabled": self._mode == "dry_run",
                "actions_evaluated": self._shadow_actions_evaluated,
                "rate_limited_actions": self._shadow_rate_limited_actions,
                "hand_clamped_actions": self._shadow_hand_clamped_actions,
                "joint_clamped_actions": self._shadow_joint_clamped_actions,
                "base_clamped_actions": self._shadow_base_clamped_actions,
                "queue_starvations": self._shadow_queue_starvations,
                "stale_queue_actions": self._shadow_stale_queue_actions,
                "stale_observation_actions": self._shadow_stale_observation_actions,
                "action_errors": self._shadow_action_errors,
                "last_error": self._last_shadow_error,
                "last_hand_clamp": self._last_shadow_hand_clamp,
                "last_joint_clamp": self._last_shadow_joint_clamp,
            },
            "last_validation_error": self._last_validation_error,
            "worker": vars(worker_stats) if worker_stats is not None else None,
            "timings": vars(prediction.timings) if prediction is not None else None,
            "peak_gpu_allocated_bytes": (
                prediction.peak_gpu_allocated_bytes if prediction is not None else None
            ),
            "peak_gpu_reserved_bytes": (
                prediction.peak_gpu_reserved_bytes if prediction is not None else None
            ),
        }
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self._status_pub.publish(message)

        diagnostic = DiagnosticArray()
        diagnostic.header.stamp = self.get_clock().now().to_msg()
        item = DiagnosticStatus()
        item.name = "dex_vega_lerobot_inference/runtime"
        item.hardware_id = "dexmate_vega_1_pro"
        item.level = (
            DiagnosticStatus.ERROR
            if self._state_machine.state in {RuntimeState.FAULT, RuntimeState.ESTOP}
            else DiagnosticStatus.WARN
            if self._state_machine.state in {RuntimeState.MODEL_LOADING, RuntimeState.ARMED}
            else DiagnosticStatus.OK
        )
        item.message = f"{self._state_machine.state.value}: {self._state_machine.reason}"
        item.values = [self._key_value(key, value) for key, value in status.items()]
        diagnostic.status = [item]
        self._diagnostics_pub.publish(diagnostic)

    def _stamp_or_now(self, message: Any) -> int:
        stamp = message.header.stamp
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        return value if value > 0 else self.get_clock().now().nanoseconds

    def _warn_throttled(self, key: str, message: str, period: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_warning.get(key, float("-inf")) >= period:
            self._last_warning[key] = now
            self.get_logger().warn(message)

    @staticmethod
    def _key_value(key: str, value: Any) -> KeyValue:
        item = KeyValue()
        item.key = str(key)
        item.value = json.dumps(value, sort_keys=True, default=str)
        return item

    def destroy_node(self) -> None:
        try:
            self._safe_stop("node shutdown", fault=False)
            if self._worker is not None and not self._worker.close(timeout_seconds=3.0):
                self.get_logger().warn("inference worker did not stop within 3 seconds")
            if self._camera_source is not None:
                self._camera_source.shutdown()
        finally:
            super().destroy_node()


def command_publication_capable(mode: str, allow_command_publication: bool) -> bool:
    """The sole condition under which command publishers may be constructed."""
    return mode == "armed" and bool(allow_command_publication)


def unexpected_command_publisher_counts(
    topic_counts: dict[str, int],
) -> dict[str, int]:
    """Return command topics that do not have exactly this node's publisher."""
    return {
        topic: int(count)
        for topic, count in topic_counts.items()
        if int(count) != 1
    }


def execution_duration_gate_failure(
    armed_since_ns: int,
    now_ns: int,
    maximum_seconds: float,
) -> str:
    """Fail closed for an invalid limit or an overlong armed interval."""
    if not np.isfinite(maximum_seconds) or maximum_seconds <= 0.0:
        return "maximum execution duration must be finite and positive"
    if armed_since_ns <= 0:
        return ""
    execution_age = max(0.0, (int(now_ns) - int(armed_since_ns)) / 1e9)
    if execution_age <= maximum_seconds:
        return ""
    return (
        f"maximum execution duration exceeded ({execution_age:.3f}s > "
        f"{maximum_seconds:.3f}s)"
    )


def project_local_path_from_server(
    server_path: str | Path,
    server_project_root: str | Path,
    local_project_root: str | Path,
) -> Path:
    """Map a server artifact through the validated workspace bind boundary."""
    server = Path(server_path)
    server_root = Path(server_project_root)
    if not server.is_absolute() or not server_root.is_absolute():
        raise ValueError("policy-server artifact and project-root paths must be absolute")
    try:
        relative = server.relative_to(server_root)
    except ValueError as exc:
        raise ValueError("policy-server artifact is outside its project root") from exc
    if relative == Path(".") or ".." in relative.parts:
        raise ValueError("policy-server artifact path is not project-local")

    local_root = Path(local_project_root).expanduser().resolve()
    local = (local_root / relative).resolve()
    try:
        local.relative_to(local_root)
    except ValueError as exc:
        raise ValueError("mapped policy-server artifact escapes the local project") from exc
    return local


def resolve_external_groot_artifacts(
    runtime_info: Any,
    *,
    local_project_root: str | Path,
    server_project_root: str | Path,
) -> GrootArtifactBundle:
    """Discover and independently verify the exact GR00T server bundle."""
    if runtime_info.policy_type != "groot":
        raise RuntimeError("external runtime is not a GR00T policy server")
    expected = {
        "model_commit": GROOT_MODEL_REVISION,
        "base_model_commit": GROOT_BASE_MODEL_REVISION,
        "processor_commit": GROOT_COSMOS_PROCESSOR_REVISION,
        "checkpoint_tag": GROOT_CHECKPOINT_TAG,
    }
    for name, value in expected.items():
        if getattr(runtime_info, name, None) != value:
            raise RuntimeError(
                f"policy server {name} differs from the exact GR00T deployment pin"
            )
    if runtime_info.tokenizer_path or runtime_info.tokenizer_commit:
        raise RuntimeError("GR00T policy server unexpectedly reported a PI tokenizer")

    local_root = Path(local_project_root).expanduser().resolve()
    model_path = project_local_path_from_server(
        runtime_info.model_path,
        server_project_root,
        local_root,
    )
    base_model_path = project_local_path_from_server(
        runtime_info.base_model_path,
        server_project_root,
        local_root,
    )
    processor_path = project_local_path_from_server(
        runtime_info.processor_path,
        server_project_root,
        local_root,
    )
    artifacts = resolve_groot_artifacts(
        project_root=local_root,
        model_local_path=model_path,
        base_model_local_path=base_model_path,
        cosmos_processor_local_path=processor_path,
        allow_download=False,
        local_files_only=True,
    )
    if artifacts.model.repo_id != GROOT_MODEL_REPO_ID:
        raise RuntimeError("GR00T model manifest identifies an unexpected Hub repo")
    if artifacts.base_model.repo_id != GROOT_BASE_MODEL_REPO_ID:
        raise RuntimeError("GR00T base manifest identifies an unexpected Hub repo")
    if artifacts.cosmos_processor.repo_id != GROOT_COSMOS_PROCESSOR_REPO_ID:
        raise RuntimeError("Cosmos processor manifest identifies an unexpected Hub repo")
    return artifacts


def resolve_external_pi05_artifacts(
    runtime_info: Any,
    *,
    local_project_root: str | Path,
    server_project_root: str | Path,
) -> tuple[ResolvedArtifact, ResolvedArtifact]:
    """Discover and independently verify the PI0.5 server's local artifacts."""
    if runtime_info.policy_type != "pi05":
        raise RuntimeError("external runtime is not a PI0.5 policy server")
    model_commit = str(runtime_info.model_commit or "")
    tokenizer_commit = str(runtime_info.tokenizer_commit or "")
    checkpoint_tag = str(runtime_info.checkpoint_tag or "")
    if not COMMIT_PATTERN.fullmatch(model_commit):
        raise RuntimeError("policy server did not report an immutable model commit")
    if not COMMIT_PATTERN.fullmatch(tokenizer_commit):
        raise RuntimeError("policy server did not report an immutable tokenizer commit")
    if not checkpoint_tag:
        raise RuntimeError("policy server did not report a checkpoint tag")

    local_root = Path(local_project_root).expanduser().resolve()
    model_path = project_local_path_from_server(
        runtime_info.model_path,
        server_project_root,
        local_root,
    )
    tokenizer_path = project_local_path_from_server(
        runtime_info.tokenizer_path,
        server_project_root,
        local_root,
    )
    model = resolve_model_artifact(
        project_root=local_root,
        local_path=model_path,
        repo_id=None,
        revision=model_commit,
        download_directory=None,
        checkpoint_tag=checkpoint_tag,
        allow_download=False,
        local_files_only=True,
    )
    if model.repo_id != MODEL_REPO_ID:
        raise RuntimeError(
            "policy server model manifest does not identify the expected private Hub repo"
        )
    tokenizer = resolve_tokenizer(
        project_root=local_root,
        local_path=tokenizer_path,
        revision=tokenizer_commit,
        download_directory=None,
        allow_download=False,
        local_files_only=True,
    )
    return model, tokenizer


def project_relative_paths_match(
    server_path: str | Path,
    server_project_root: str | Path,
    local_path: str | Path,
    local_project_root: str | Path,
) -> bool:
    """Compare artifacts across the policy-server workspace bind boundary."""
    try:
        mapped = project_local_path_from_server(
            server_path,
            server_project_root,
            local_project_root,
        )
    except (OSError, ValueError):
        return False
    return mapped == Path(local_path).expanduser().resolve()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: InferenceNode | None = None
    try:
        node = InferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Let the finally block complete queue invalidation, worker join, and
        # camera shutdown. `ros2 launch` can deliver the terminal SIGINT and
        # then explicitly signal the child a second time; ignore only those
        # subsequent SIGINTs once shutdown has already begun.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
