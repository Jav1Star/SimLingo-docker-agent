import os
from unittest.mock import patch

from simlingo_agents.bench2drive_mcp_server.remote_inference import (
    SplitAgentPipelineClient,
)
from simlingo_agents.common.nats_comm import NatsComm
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType


INSTANCE_ENV = {
    "CLUSTER_ID": "edge-c",
    "NATS_JETSTREAM_DOMAIN": "edge-c",
    "AGENT_ID": "simlingo-encoder",
    "AGENT_INSTANCE_ID": "encoder-uid",
    "SIMLINGO_ENCODER_INSTANCE_ID": "encoder-uid",
    "SIMLINGO_SCHEDULER_INSTANCE_ID": "scheduler-uid",
    "SIMLINGO_LLM_INSTANCE_ID": "llm-uid",
}


def test_agent_stream_is_scoped_to_pod_uid():
    with patch.dict(os.environ, INSTANCE_ENV, clear=False):
        comm = NatsComm()
        config = comm._stream_config()

    assert config.name == "WF_encoder-uid"
    assert config.storage == StorageType.FILE
    assert config.retention == RetentionPolicy.WORK_QUEUE
    assert config.discard == DiscardPolicy.NEW
    assert config.subjects == [
        "workflow.local.edge-c.agent.simlingo-encoder.instance.encoder-uid.>",
        "workflow.global.edge-c.agent.simlingo-encoder.instance.encoder-uid.>",
    ]


def test_pipeline_subjects_target_each_agent_instance():
    with patch.dict(os.environ, INSTANCE_ENV, clear=False):
        client = SplitAgentPipelineClient()
        subjects = client._build_subjects("unused")

    assert subjects["source_input"].endswith(
        ".agent.simlingo-encoder.instance.encoder-uid.encoder_input"
    )
    assert subjects["encoded_output"].endswith(
        ".agent.simlingo-scheduler.instance.scheduler-uid.scheduler_budget_input"
    )
    assert subjects["prefix_input"].endswith(
        ".agent.simlingo-llm.instance.llm-uid.llm_prefix_input"
    )
    assert subjects["final_output"].endswith(
        ".agent.simlingo-llm.instance.llm-uid.llm_final_output"
    )


def test_route_cleanup_does_not_purge_instance_stream():
    with patch.dict(os.environ, INSTANCE_ENV, clear=False):
        client = SplitAgentPipelineClient()
        result = client.cleanup_route("route-009")

    assert result["nats_cleanup"] == "not-required"
