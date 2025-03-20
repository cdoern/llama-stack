# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    JobStatus,
    ListPostTrainingJobsResponse,
    LoraFinetuningConfig,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)
from .config import (
    InstructLabKubeFlowPostTrainingConfig,
)
from llama_stack.providers.inline.post_training.torchtune.recipes.lora_finetuning_single_device import (
    LoraFinetuningSingleDevice,
)
from llama_stack.schema_utils import webmethod
from llama_stack.providers.utils.scheduler import Scheduler

_JOB_TYPE_SUPERVISED_FINE_TUNE = "supervised-fine-tune"


class InstructLabPostTrainingImpl:
    def __init__(
        self,
        config: InstructLabKubeFlowPostTrainingConfig,
        datasetio_api: DatasetIO,
        datasets: Datasets,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets
        self._scheduler = Scheduler(backend="kubeflow")

        # TODO: assume sync job, will need jobs API for async scheduling
        self.jobs = {}
        self.checkpoints_dict = {}

    async def shutdown(self):
        pass

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
        model: str,
        checkpoint_dir: Optional[str],
        algorithm_config: Optional[AlgorithmConfig],
    ) -> PostTrainingJob:
        if job_uuid in self.jobs:
            raise ValueError(f"Job {job_uuid} already exists")

        post_training_job = PostTrainingJob(job_uuid=job_uuid)

        job_status_response = PostTrainingJobStatusResponse(
            job_uuid=job_uuid,
            status=JobStatus.scheduled,
            scheduled_at=datetime.now(timezone.utc),
        )
        self.jobs[job_uuid] = job_status_response

        if isinstance(algorithm_config, LoraFinetuningConfig):
            raise NotImplementedError()
        else:
            async def handler():
                import kfp
                import os
                from kfp import dsl
                from kfp.kubernetes import (
                    CreatePVC,
                    DeletePVC,
                    mount_pvc,
                    use_secret_as_env,
                    use_secret_as_volume,
                )

               # KFP_ENDPONT = os.getenv("KFP_ENDPOINT")
               # KUBECONFIG = os.getenv("KUBECONFIG")
                @dsl.component(base_image=RUNTIME_GENERIC_IMAGE, install_kfp_package=False)
                def model_to_pvc_op(model: dsl.Input[dsl.Model], pvc_path: str = "/model"):
                    import os
                    import os.path
                    import shutil

                    # shutil.copytree fails with "Operation Not Permitted" but doing one file at a time works for some reason.
                    for f in os.listdir(model.path):
                        src = os.path.join(model.path, f)
                        dest = os.path.join(pvc_path, f)
                        print(f"Copying {src} to {dest}")
                        if os.path.isdir(src):
                            shutil.copytree(src, dest)
                        else:
                            shutil.copy(src, dest)


                @dsl.container_component
                def ilab_importer_op(repository: str, release: str, base_model: dsl.Output[dsl.Model]):
                    return dsl.ContainerSpec(
                        RHELAI_IMAGE,
                        ["/bin/sh", "-c"],
                        [
                            f"ilab --config=DEFAULT model download --repository {repository} --release {release} --model-dir {base_model.path}"
                        ],
                    )

                @dsl.component(
                    base_image=RHELAI_IMAGE,
                    install_kfp_package=False,
                )
                def data_processing_op(
                    model_path: str = self.config.model_path,
                    sdg_path: str = "/data/sdg",
                    skills_path: str = "/data/skills",
                    knowledge_path: str = "/data/knowledge",
                    max_seq_len: Optional[int] = training_config.max_seq_len,
                    max_batch_len: Optional[int] = training_config.max_batch_len,
                ):
                    import os

                    import instructlab.training.data_process as dp
                    from instructlab.training import (
                        DataProcessArgs,
                        TrainingArgs,
                    )

                    # define training-specific arguments
                    skill_training_args = TrainingArgs(
                        # define data-specific arguments
                        model_path=model_path,
                        data_path=f"{sdg_path}/skills_train_msgs*.jsonl",
                        data_output_dir=skills_path,
                        # define model-trianing parameters
                        max_seq_len=max_seq_len,
                        max_batch_len=max_batch_len,
                        ckpt_output_dir="data/saved_checkpoints",
                        num_epochs=2,
                        effective_batch_size=3840,
                        save_samples=0,
                        learning_rate=2e-6,
                        warmup_steps=800,
                        is_padding_free=True,
                    )

                    knowledge_training_args = TrainingArgs(
                        # define data-specific arguments
                        model_path=model_path,
                        data_path=f"{sdg_path}/knowledge_train_msgs*.jsonl",
                        data_output_dir=knowledge_path,
                        max_seq_len=max_seq_len,
                        max_batch_len=max_batch_len,
                        ckpt_output_dir="data/saved_checkpoints",
                        num_epochs=2,
                        effective_batch_size=3840,
                        save_samples=0,
                        learning_rate=2e-6,
                        warmup_steps=800,
                        is_padding_free=True,
                    )

                    def data_processing(train_args: TrainingArgs) -> None:
                        # early validation logic here
                        if train_args.max_batch_len < train_args.max_seq_len:
                            raise ValueError(
                                f"the 'max_batch_len' cannot be less than 'max_seq_len': {train_args.max_batch_len=} < {train_args.max_seq_len=}"
                            )

                            # process the training data
                        if not os.path.exists(train_args.data_output_dir):
                            os.makedirs(train_args.data_output_dir, exist_ok=True)
                        dp.main(
                            DataProcessArgs(
                                data_output_path=train_args.data_output_dir,
                                model_path=train_args.model_path,
                                data_path=train_args.data_path,
                                max_seq_len=train_args.max_seq_len,
                                chat_tmpl_path=train_args.chat_tmpl_path,
                            )
                        )

                    data_processing(train_args=skill_training_args)
                    data_processing(train_args=knowledge_training_args)


                @dsl.container_component
                def skills_processed_data_to_artifact_op(
                    skills_processed_data: dsl.Output[dsl.Dataset],
                    pvc_path: str = self.config.skills_pvc_path,
                ):
                    return dsl.ContainerSpec(
                        TOOLBOX_IMAGE,
                        ["/bin/sh", "-c"],
                        [f"cp -r {pvc_path} {skills_processed_data.path}"],
                    )


                @dsl.container_component
                def knowledge_processed_data_to_artifact_op(
                    knowledge_processed_data: dsl.Output[dsl.Dataset],
                    pvc_path: str = self.config.knowledge_pvc_path,
                ):
                    return dsl.ContainerSpec(
                        TOOLBOX_IMAGE,
                        ["/bin/sh", "-c"],
                        [f"cp -r {pvc_path} {knowledge_processed_data.path}"],
                    )


                # Change base image to the RHOAI python image with kubeflow_training once available
                @dsl.component(base_image=PYTHON_IMAGE, install_kfp_package=False)
                def pytorch_job_launcher_op(
                    gpu_identifier: str,
                    cpu_per_worker: str,
                    memory_per_worker: str,
                    tolerations: list,
                    node_selectors: dict,
                    pytorchjob_output_yaml: dsl.Output[dsl.Artifact],
                    model_pvc_name: str,
                    input_pvc_name: str,
                    output_pvc_name: str,
                    name_suffix: str,
                    phase_num: int,
                    base_image: str,
                    nproc_per_node: int = 3,
                    nnodes: int = 2,
                    num_epochs: int = 2,
                    effective_batch_size: int = 3840,
                    learning_rate: float = 1e-4,
                    num_warmup_steps: int = 800,
                    save_samples: int = 0,
                    max_batch_len: int = 20000,
                    seed: int = 42,
                    job_timeout: int = 86400,
                    delete_after_done: bool = False,
                ):
                    import logging
                    import os

                    from kubeflow.training import TrainingClient, models
                    from kubeflow.training.constants.constants import ISTIO_SIDECAR_INJECTION
                    from kubeflow.training.utils import utils as kfto_utils

                    def list_phase1_final_model():
                        model_dir = "/output/phase_1/model/hf_format"
                        model_list = os.listdir(model_dir)
                        newest_idx = max(
                            (os.path.getmtime(f"{model_dir}/{model}"), i)
                            for i, model in enumerate(model_list)
                        )[-1]
                        newest_model = model_list[newest_idx]
                        return f"{model_dir}/{newest_model}"

                    if phase_num == 1:
                        path_to_model = "/input_model"
                        path_to_data = "/input_data/knowledge/data.jsonl"
                    elif phase_num == 2:
                        path_to_model = list_phase1_final_model()
                        path_to_data = "/input_data/skills/data.jsonl"
                    else:
                        raise RuntimeError(f"Unsupported value of {phase_num=}")

                    if gpu_identifier == "":
                        raise RuntimeError(f"GPU identifier cannot be empty")
                    resources_per_worker = {
                        "cpu": cpu_per_worker,
                        "memory": memory_per_worker,
                        gpu_identifier: nproc_per_node,
                    }

                    name = f"train-phase-{phase_num}-{name_suffix.rstrip('-sdg')}"
                    command = ["/bin/sh", "-c", "--"]

                    master_args = [
                        f"""
                        echo "Running phase {phase_num}"
                        echo "Using {path_to_model} model for training"
                        echo "Using {path_to_data} data for training"
                        mkdir -p /output/phase_{phase_num}/model;
                        mkdir -p /output/data;
                        torchrun --nnodes {nnodes} \
                            --nproc_per_node {nproc_per_node} \
                            --node_rank \$(RANK) \
                            --rdzv_endpoint \$(MASTER_ADDR):\$(MASTER_PORT) \
                            -m instructlab.training.main_ds \
                            --model_name_or_path={path_to_model} \
                            --data_path={path_to_data} \
                            --output_dir=/output/phase_{phase_num}/model \
                            --num_epochs={num_epochs} \
                            --effective_batch_size={effective_batch_size} \
                            --learning_rate={learning_rate} \
                            --num_warmup_steps={num_warmup_steps} \
                            --save_samples={save_samples} \
                            --log_level=INFO \
                            --max_batch_len={max_batch_len} \
                            --seed={seed} \
                            --cpu_offload_optimizer \
                            --cpu_offload_params_fsdp \
                            --distributed_training_framework fsdp \
                            --checkpoint_at_epoch
                            """
                    ]

                    worker_args = [
                        f"""
                        echo "Running phase {phase_num}"
                        echo "Using {path_to_model} model for training"
                        echo "Using {path_to_data} data for training"
                        mkdir -p /tmp/model;
                        torchrun --nnodes {nnodes} \
                        --nproc_per_node {nproc_per_node} \
                        --node_rank \$(RANK) \
                        --rdzv_endpoint \$(MASTER_ADDR):\$(MASTER_PORT) \
                        -m instructlab.training.main_ds \
                        --model_name_or_path={path_to_model} \
                        --data_path={path_to_data} \
                        --output_dir=/tmp/model \
                        --num_epochs={num_epochs} \
                        --effective_batch_size={effective_batch_size} \
                        --learning_rate={learning_rate} \
                        --num_warmup_steps={num_warmup_steps} \
                        --save_samples={save_samples} \
                        --log_level=INFO \
                        --max_batch_len={max_batch_len} \
                        --seed={seed} \
                        --cpu_offload_optimizer \
                        --cpu_offload_params_fsdp \
                        --distributed_training_framework fsdp \
                        --checkpoint_at_epoch
                        """
                    ]

                    # Set volumes
                    volumes = [
                        models.V1Volume(
                            name="input-data",
                            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                                claim_name=input_pvc_name
                            ),
                        ),
                        models.V1Volume(
                            name="model",
                            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                                claim_name=model_pvc_name
                            ),
                        ),
                        models.V1Volume(
                            name="output",
                            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                                claim_name=output_pvc_name
                            ),
                        ),
                    ]

                    # Set volume mounts
                    volume_mounts_master = [
                        models.V1VolumeMount(
                            mount_path="/input_data", name="input-data", read_only=True
                        ),
                        models.V1VolumeMount(mount_path="/input_model", name="model", read_only=True),
                        models.V1VolumeMount(mount_path="/output", name="output"),
                    ]

                    volume_mounts_worker = [
                        models.V1VolumeMount(
                            mount_path="/input_data", name="input-data", read_only=True
                        ),
                        models.V1VolumeMount(mount_path="/input_model", name="model", read_only=True),
                        models.V1VolumeMount(mount_path="/output", name="output", read_only=True),
                    ]

                    # Set env variables
                    env_vars = [
                        models.V1EnvVar(name="NNODES", value=f"{nnodes}"),
                        models.V1EnvVar(name="NPROC_PER_NODE", value=f"{nproc_per_node}"),
                        models.V1EnvVar(name="XDG_CACHE_HOME", value="/tmp"),
                        models.V1EnvVar(name="TRITON_CACHE_DIR", value="/tmp"),
                        models.V1EnvVar(name="HF_HOME", value="/tmp"),
                        models.V1EnvVar(name="TRANSFORMERS_CACHE", value="/tmp"),
                    ]

                    # Get master and worker container specs
                    master_container_spec = kfto_utils.get_container_spec(
                        base_image=base_image,
                        name="pytorch",
                        resources=resources_per_worker,
                        volume_mounts=volume_mounts_master,
                    )

                    # In the next release of kubeflow-training, the command
                    # and the args will be a part of kfto_utils.get_container_spec function
                    master_container_spec.command = command
                    master_container_spec.args = master_args

                    master_container_spec.env = env_vars

                    worker_container_spec = kfto_utils.get_container_spec(
                        base_image=base_image,
                        name="pytorch",
                        resources=resources_per_worker,
                        volume_mounts=volume_mounts_worker,
                    )
                    worker_container_spec.command = command
                    worker_container_spec.args = worker_args
                    worker_container_spec.env = env_vars

                    # create master pod spec
                    master_pod_template_spec = models.V1PodTemplateSpec(
                        metadata=models.V1ObjectMeta(annotations={ISTIO_SIDECAR_INJECTION: "false"}),
                        spec=models.V1PodSpec(
                            init_containers=None,
                            containers=[master_container_spec],
                            volumes=volumes,
                            tolerations=tolerations,
                            node_selector=node_selectors,
                        ),
                    )

                    # create worker pod spec
                    worker_pod_template_spec = models.V1PodTemplateSpec(
                        metadata=models.V1ObjectMeta(annotations={ISTIO_SIDECAR_INJECTION: "false"}),
                        spec=models.V1PodSpec(
                            init_containers=None,
                            containers=[worker_container_spec],
                            volumes=volumes,
                            tolerations=tolerations,
                            node_selector=node_selectors,
                        ),
                    )

                    logging.getLogger(__name__).setLevel(logging.INFO)
                    logging.info("Generating job template.")
                    logging.info("Creating TrainingClient.")

                    # Initialize training client
                    # This also finds the namespace from /var/run/secrets/kubernetes.io/serviceaccount/namespace
                    # And it also loads the kube config
                    training_client = TrainingClient()
                    namespace = training_client.namespace
                    # Create pytorch job spec
                    job_template = kfto_utils.get_pytorchjob_template(
                        name=name,
                        namespace=namespace,
                        worker_pod_template_spec=worker_pod_template_spec,
                        master_pod_template_spec=master_pod_template_spec,
                        num_workers=nnodes,
                        num_procs_per_worker=nproc_per_node,
                    )
                    # Save the pytorch job yaml for record keeping and debugging
                    with open(pytorchjob_output_yaml.path, "w", encoding="utf-8") as f:
                        f.write(job_template.to_str())

                    # Run the pytorch job
                    logging.info(f"Creating PyTorchJob in namespace: {namespace}")
                    training_client.create_job(job_template, namespace=namespace)

                    expected_conditions = ["Succeeded", "Failed"]
                    logging.info(f"Monitoring job until status is any of {expected_conditions}.")

                    def get_logs(job):
                        _, _ = training_client.get_job_logs(name=job.metadata.name, follow=True)

                    training_client.wait_for_job_conditions(
                        name=name,
                        expected_conditions=set(expected_conditions),
                        wait_timeout=job_timeout,
                        timeout=job_timeout,
                        callback=get_logs,
                    )

                    if delete_after_done:
                        logging.info("Deleting job after completion.")
                        training_client.delete_job(name, namespace)
                    


                @dsl.pipeline(name=job_uuid)
                def ilab_pipeline():
                    model_pvc_task = CreatePVC(
                        pvc_name_suffix="-model-cache",
                        access_modes=["ReadWriteMany"],
                        size=k8s_storage_size,
                        storage_class_name=k8s_storage_class_name,
                    )
                    model_to_pvc_task = model_to_pvc_op(model=model_source_task.output)
                    model_to_pvc_task.set_caching_options(False)
                    mount_pvc(
                        task=model_to_pvc_task, pvc_name=model_pvc_task.output, mount_path="/model"
                    )

                    # Data processing
                    data_processing_task = data_processing_op(max_batch_len=training_config.max_batch_len)
                    mount_pvc(
                        task=data_processing_task,
                        pvc_name=model_pvc_task.output,
                        mount_path="/model",
                    )
                    mount_pvc(
                        task=data_processing_task,
                        pvc_name=sdg_input_pvc_task.output,
                        mount_path="/data",
                    )
                   # data_processing_task.after(model_to_pvc_task, sdg_task)
                    data_processing_task.set_caching_options(False)
                    data_processing_task.set_env_variable("XDG_CACHE_HOME", "/tmp")

                    # Upload "skills_processed_data" and "knowledge_processed_data" artifacts to S3 without blocking the rest of the workflow
                    skills_processed_data_to_artifact_task = skills_processed_data_to_artifact_op()
                    skills_processed_data_to_artifact_task.after(data_processing_task)
                    mount_pvc(
                        task=skills_processed_data_to_artifact_task,
                        pvc_name=sdg_input_pvc_task.output,
                        mount_path="/data",
                    )
                    skills_processed_data_to_artifact_task.set_caching_options(False)
                    knowledge_processed_data_to_artifact_task = (
                        knowledge_processed_data_to_artifact_op()
                    )
                    knowledge_processed_data_to_artifact_task.after(data_processing_task)
                    mount_pvc(
                        task=knowledge_processed_data_to_artifact_task,
                        pvc_name=sdg_input_pvc_task.output,
                        mount_path="/data",
                    )
                    knowledge_processed_data_to_artifact_task.set_caching_options(False)

                    output_pvc_task = CreatePVC(
                        pvc_name_suffix="-output",
                        access_modes=["ReadWriteMany"],
                        size=k8s_storage_size,
                        storage_class_name=k8s_storage_class_name,
                    )
                    output_pvc_task.after(prerequisites_check_task)

                    # Training 1
                    # Using pvc_create_task.output as PyTorchJob name since dsl.PIPELINE_* global variables do not template/work in KFP v2
                    # https://github.com/kubeflow/pipelines/issues/10453
                    training_phase_1 = pytorch_job_launcher_op(
                        gpu_identifier=train_gpu_identifier,
                        cpu_per_worker=train_cpu_per_worker,
                        memory_per_worker=train_memory_per_worker,
                        tolerations=train_tolerations,
                        node_selectors=train_node_selectors,
                        model_pvc_name=model_pvc_task.output,
                        input_pvc_name=sdg_input_pvc_task.output,
                        name_suffix=sdg_input_pvc_task.output,
                        output_pvc_name=output_pvc_task.output,
                        phase_num=1,
                        base_image=RHELAI_IMAGE,
                        nproc_per_node=self.config.nproc_per_node,
                        nnodes=training_config.nnodes,
                        num_epochs=training_config.n_epochs,
                        effective_batch_size=training_config.effective_batch_size, # this might need to be different per-phase
                        learning_rate=training_config.learning_rate, # this might need to be different per-phase
                        num_warmup_steps=training_config.learning_rate,
                        save_samples=self.config.save_samples,
                        max_batch_len=training_config.max_batch_len,
                        seed=self.config.seed,
                    )
                    training_phase_1.after(data_processing_task, model_to_pvc_task)
                    training_phase_1.set_caching_options(False)

                    #### Train 2
                    training_phase_2 = pytorch_job_launcher_op(
                        gpu_identifier=train_gpu_identifier,
                        cpu_per_worker=train_cpu_per_worker,
                        memory_per_worker=train_memory_per_worker,
                        tolerations=train_tolerations,
                        node_selectors=train_node_selectors,
                        model_pvc_name=model_pvc_task.output,
                        input_pvc_name=sdg_input_pvc_task.output,
                        name_suffix=sdg_input_pvc_task.output,
                        output_pvc_name=output_pvc_task.output,
                        phase_num=2,
                        base_image=RHELAI_IMAGE,
                        nproc_per_node=train_gpu_per_worker,
                        nnodes=training_config.nnodes,
                        num_epochs=training_config.n_epochs,
                        effective_batch_size=training_config.effective_batch_size, # this might need to be different per-phase
                        learning_rate=training_config.learning_rate, # this might need to be different per-phase
                        num_warmup_steps=self.config.num_warmup_steps,
                        save_samples=self.config.save_samples,
                        max_batch_len=training_config.max_batch_len,
                        seed=self.config.seed,
                    )

                    training_phase_2.set_caching_options(False)
                    training_phase_2.after(training_phase_1)

                    mount_pvc(
                        task=training_phase_2,
                        pvc_name=output_pvc_task.output,
                        mount_path="/output",
                    )

                    # Connect to the KFP server
                    #kfp.compiler.Compiler().compile(pipeline, "ilab.yaml")
                    # Upload the pipeline (if it doesn’t exist)
                    #pipeline = client.upload_pipeline("ilab.yaml", pipeline_name=f"ilab-pipeline-{job_uuid}")

                    # Create an experiment (if needed)
                    #experiment = client.create_experiment(name=f"ilab-exp-{job_uuid}")

                # Run the pipeline

                dev_arguments = {
                        "k8s_storage_class_name": "nfs-csi",
                        "train_num_epochs_phase_1": 2,
                        "train_num_epochs_phase_2": 2,
                        "train_num_warmup_steps_phase_1": 100,
                        "train_num_warmup_steps_phase_2": 100,
                        "train_learning_rate_phase_1": 1e-4,
                        "train_learning_rate_phase_2": 1e-4,
                    }

                client = get_kfp_client()
                client.create_run_from_pipeline_func(
                    pipeline_func=ilab_pipeline,
                    experiment_name=self.config.experiment_name,
                    run_name=f"ilab-pipeline-{job_uuid}",
                    arguments={**dev_arguments},
                )


            job_uuid = self._scheduler.schedule(_JOB_TYPE_SUPERVISED_FINE_TUNE, job_uuid, handler)


        return PostTrainingJob(job_uuid=job_uuid)

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob: ...

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        return ListPostTrainingJobsResponse(data=[PostTrainingJob(job_uuid=uuid_) for uuid_ in self.jobs])

    @webmethod(route="/post-training/job/status")
    async def get_training_job_status(self, job_uuid: str) -> Optional[PostTrainingJobStatusResponse]:
        return self.jobs.get(job_uuid, None)

    @webmethod(route="/post-training/job/cancel")
    async def cancel_training_job(self, job_uuid: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    @webmethod(route="/post-training/job/artifacts")
    async def get_training_job_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]:
        if job_uuid in self.checkpoints_dict:
            checkpoints = self.checkpoints_dict.get(job_uuid, [])
            return PostTrainingJobArtifactsResponse(job_uuid=job_uuid, checkpoints=checkpoints)
        return None

# type: ignore
import warnings



def get_kfp_client():

    from kfp import Client
    from kubernetes.client import CustomObjectsApi
    from kubernetes.client.configuration import Configuration
    from kubernetes.client.exceptions import ApiException
    from kubernetes.config import list_kube_config_contexts
    from kubernetes.config.config_exception import ConfigException
    from kubernetes.config.kube_config import load_kube_config


    config = Configuration()
    try:
        load_kube_config(client_configuration=config)
        token = config.api_key["authorization"].split(" ")[-1]
    except (KeyError, ConfigException) as e:
        raise ApiException(
            401, "Unauthorized, try running `oc login` command first"
        ) from e
    Configuration.set_default(config)

    _, active_context = list_kube_config_contexts()
    namespace = active_context["context"]["namespace"]

    dspas = CustomObjectsApi().list_namespaced_custom_object(
        "datasciencepipelinesapplications.opendatahub.io",
        "v1alpha1",
        namespace,
        "datasciencepipelinesapplications",
    )

    try:
        dspa = dspas["items"][0]
    except IndexError as e:
        raise ApiException(404, "DataSciencePipelines resource not found") from e

    try:
        if dspa["spec"]["dspVersion"] != "v2":
            raise KeyError
    except KeyError as e:
        raise EnvironmentError(
            "Installed version of Kubeflow Pipelines does not meet minimal version criteria. Use KFPv2 please."
        ) from e

    try:
        host = dspa["status"]["components"]["apiServer"]["externalUrl"]
    except KeyError as e:
        raise ApiException(
            409,
            "DataSciencePipelines resource is not ready. Check for .status.components.apiServer",
        ) from e

    with warnings.catch_warnings(action="ignore"):
        return Client(existing_token=token, host=host)