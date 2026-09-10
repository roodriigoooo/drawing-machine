"""Pure runtime support for Direction 4 relation actions.

This package deliberately knows nothing about corpus construction, manifests,
models, or evaluation.  It is the narrow execution boundary shared by a future
predicted decoder and the oracle diagnostic path.
"""

from .candidates import candidate_spans
from .queue import RowByteQueues
from .spec import (
                   COUNT_SUPPORT,
                   D4_SUPPORT,
                   ORACLE_COUNT_SUPPORT,
                   ORACLE_SUPPORT,
                   PREDICTED_SUPPORT,
                   TRANSLATION_SUPPORT,
                   CopyActionKey,
                   RuntimeBounds,
                   SupportPolicy,
)
from .transducer import (
    CopyExecution,
    FaultCode,
    TransducerFault,
    execute_copy,
    prefix_index,
    prefix_progress,
    validate_prefix,
)

__all__ = (
                   "COUNT_SUPPORT",
                   "D4_SUPPORT",
                   "ORACLE_COUNT_SUPPORT",
                   "ORACLE_SUPPORT",
                   "PREDICTED_SUPPORT",
                   "TRANSLATION_SUPPORT",
                   "CopyActionKey",
                   "CopyExecution",
                   "FaultCode",
                   "RowByteQueues",
                   "RuntimeBounds",
                   "SupportPolicy",
                   "TransducerFault",
                   "candidate_spans",
                   "execute_copy",
                   "prefix_index",
                   "prefix_progress",
                   "validate_prefix",
)
