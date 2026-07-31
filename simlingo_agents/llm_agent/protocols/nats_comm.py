try:
    from simlingo_agents.common.nats_comm import NatsComm, NatsMessage
except ModuleNotFoundError:
    from common.nats_comm import NatsComm, NatsMessage

__all__ = ["NatsComm", "NatsMessage"]
