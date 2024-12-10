import itertools
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from typing import List, Dict, Optional

from hassil.util import (
    normalize_whitespace,
    check_excluded_context,
    check_required_context,
)
from hassil.intents import (
    Intents,
    IntentData,
    TextSlotList,
    RangeSlotList,
    SlotList,
    TextSlotValue,
)
from hassil.expression import (
    Expression,
    TextChunk,
    Sequence,
    SequenceType,
    ListReference,
    RuleReference,
    Sentence,
)


def sample_intents(intents: Intents) -> Dict[str, Dict[str, List[str]]]:
    """Sample text strings for sentences from intents."""
    sentences = defaultdict(lambda: defaultdict(list))

    for intent_name, intent in sorted(intents.intents.items(), key=lambda kv: kv[0]):
        for group_idx, intent_data in enumerate(intent.data):
            for intent_sentence in sorted(intent_data.sentences, key=lambda s: s.text):
                sentence_texts = sample_expression(
                    intent_sentence, intent_data, intents
                )
                for sentence_text in sorted(sentence_texts):
                    sentences[intent_name][group_idx].append(sentence_text)

    return sentences


def sample_expression(
    expression: Expression,
    intent_data: IntentData,
    intents: Intents,
) -> Iterable[str]:
    """Sample possible text strings from an expression."""
    if isinstance(expression, TextChunk):
        chunk: TextChunk = expression
        yield chunk.original_text
    elif isinstance(expression, Sequence):
        seq: Sequence = expression
        if seq.is_optional:
            yield ""
        elif seq.type == SequenceType.ALTERNATIVE:
            for item in seq.items:
                yield from sample_expression(item, intent_data, intents)
        elif seq.type == SequenceType.GROUP:
            seq_sentences = map(
                partial(sample_expression, intent_data=intent_data, intents=intents),
                seq.items,
            )
            sentence_texts = itertools.product(*seq_sentences)
            for sentence_words in sentence_texts:
                yield normalize_whitespace("".join(sentence_words))
    elif isinstance(expression, ListReference):
        # {list}
        list_ref: ListReference = expression

        slot_list: Optional[SlotList] = intent_data.slot_lists.get(list_ref.list_name)

        if slot_list is None:
            slot_list = intents.slot_lists.get(list_ref.list_name)

        if isinstance(slot_list, TextSlotList):
            text_list: TextSlotList = slot_list

            # Filter by context
            possible_values: List[TextSlotValue] = []
            if intent_data.requires_context or intent_data.excludes_context:
                for value in text_list.values:
                    if not value.context:
                        possible_values.append(value)
                        continue

                    if intent_data.requires_context and (
                        not check_required_context(
                            intent_data.requires_context,
                            value.context,
                            allow_missing_keys=True,
                        )
                    ):
                        continue

                    if intent_data.excludes_context and (
                        not check_excluded_context(
                            intent_data.excludes_context, value.context
                        )
                    ):
                        continue

                    possible_values.append(value)
            else:
                possible_values = text_list.values

            if possible_values:
                # First and list values
                sample_values = [possible_values[0]]
                if len(possible_values) > 1:
                    sample_values.append(possible_values[-1])

                for value in sample_values:
                    for value_text in sample_expression(
                        value.text_in, intent_data, intents
                    ):
                        yield value_text
                        break
            else:
                yield f"{{{list_ref.list_name}}}"
        elif isinstance(slot_list, RangeSlotList):
            range_list: RangeSlotList = slot_list

            yield str(range_list.start)
            yield str(range_list.stop)
        else:
            yield f"{{{list_ref.list_name}}}"
    elif isinstance(expression, RuleReference):
        # <rule>
        rule_ref: RuleReference = expression

        rule_body: Optional[Sentence] = intent_data.expansion_rules.get(
            rule_ref.rule_name
        )
        if rule_body is None:
            rule_body = intents.expansion_rules.get(rule_ref.rule_name)

        if rule_body is not None:
            yield from sample_expression(rule_body, intent_data, intents)
        else:
            yield f"<{rule_ref.rule_name}>"
    else:
        yield ""
