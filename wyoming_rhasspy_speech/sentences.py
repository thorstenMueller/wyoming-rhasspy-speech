import abc
import itertools
import time
import logging
from collections.abc import Iterable, Sequence as ABCSequence
from typing import Dict, Any, List, Set, Optional, Tuple
from functools import partial

import hassil.parse_expression
import hassil.sample
from hassil.intents import TextSlotList
from hassil.recognize import MissingListError, MissingRuleError
from hassil.util import normalize_whitespace
from hassil.intents import SlotList, TextChunk, TextSlotList, TextSlotValue
from hassil.expression import (
    ListReference,
    RuleReference,
    Sequence,
    SequenceType,
    TextChunk,
    Sentence,
    Expression,
)
from unicode_rbnf import RbnfEngine


_LOGGER = logging.getLogger()


def generate_sentences(sentences_yaml: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    start_time = time.monotonic()

    # sentences:
    #   - same text in and out
    #   - in: text in
    #     out: different text out
    #   - in:
    #       - multiple text
    #       - multiple text in
    #     out: different text out
    # lists:
    #   <name>:
    #     - value 1
    #     - value 2
    # expansion_rules:
    #   <name>: sentence template
    templates = sentences_yaml["sentences"]

    # TODO
    engine = RbnfEngine.for_language("en")

    # Load slot lists
    slot_lists: Dict[str, SlotList] = {}
    for slot_name, slot_info in sentences_yaml.get("lists", {}).items():
        if isinstance(slot_info, ABCSequence):
            slot_info = {"values": slot_info}

        slot_list_values: List[TextSlotValue] = []

        slot_range = slot_info.get("range")
        if slot_range:
            slot_from = int(slot_range["from"])
            slot_to = int(slot_range["to"])
            slot_step = int(slot_range.get("step", 1))
            for i in range(slot_from, slot_to + 1, slot_step):
                slot_list_values.append(
                    TextSlotValue(
                        text_in=TextChunk(engine.format_number(i).replace("-", " ")),
                        value_out=i,
                    )
                )

            slot_lists[slot_name] = TextSlotList(
                name=slot_name, values=slot_list_values
            )
            continue

        slot_values = slot_info.get("values")
        if not slot_values:
            _LOGGER.warning("No values for list %s, skipping", slot_name)
            continue

        for slot_value in slot_values:
            values_in: List[str] = []

            if isinstance(slot_value, str):
                values_in.append(slot_value)
                value_out: str = slot_value
            else:
                # - in: text to say
                #   out: text to output
                value_in = slot_value["in"]
                value_out = slot_value["out"]

                if hassil.intents.is_template(value_in):
                    input_expression = hassil.parse_expression.parse_sentence(value_in)
                    for input_text in hassil.sample.sample_expression(
                        input_expression,
                    ):
                        values_in.append(input_text)
                else:
                    values_in.append(value_in)

            for value_in in values_in:
                slot_list_values.append(
                    TextSlotValue(TextChunk(value_in), value_out=value_out)
                )

        slot_lists[slot_name] = TextSlotList(name=slot_name, values=slot_list_values)

    # Load expansion rules
    expansion_rules: Dict[str, hassil.Sentence] = {}
    for rule_name, rule_text in sentences_yaml.get("expansion_rules", {}).items():
        expansion_rules[rule_name] = hassil.parse_sentence(rule_text)

    # Generate possible sentences
    num_sentences = 0
    for template in templates:
        if isinstance(template, str):
            input_templates: List[str] = [template]
            output_text: Optional[str] = None
        else:
            input_str_or_list = template["in"]
            if isinstance(input_str_or_list, str):
                # One template
                input_templates = [input_str_or_list]
            else:
                # Multiple templates
                input_templates = input_str_or_list

            output_text = template.get("out")

        for input_template in input_templates:
            if hassil.intents.is_template(input_template):
                # Generate possible texts
                input_expression = hassil.parse_expression.parse_sentence(
                    input_template
                )
                for input_text, maybe_output_text in sample_expression_with_output(
                    input_expression,
                    slot_lists=slot_lists,
                    expansion_rules=expansion_rules,
                ):
                    yield (input_text, output_text or maybe_output_text or input_text)
                    num_sentences += 1
            else:
                # Not a template
                yield (input_template, output_text or input_template)
                num_sentences += 1

    end_time = time.monotonic()

    _LOGGER.info(
        "Generated %s sentence(s) with in %0.2f second(s)",
        num_sentences,
        end_time - start_time,
    )


def sample_expression_with_output(
    expression: Expression,
    slot_lists: Optional[Dict[str, SlotList]] = None,
    expansion_rules: Optional[Dict[str, Sentence]] = None,
) -> Iterable[Tuple[str, Optional[str]]]:
    """Sample possible text strings from an expression."""
    if isinstance(expression, TextChunk):
        chunk: TextChunk = expression
        yield (chunk.original_text, chunk.original_text)
    elif isinstance(expression, Sequence):
        seq: Sequence = expression
        if seq.type == SequenceType.ALTERNATIVE:
            for item in seq.items:
                yield from sample_expression_with_output(
                    item,
                    slot_lists,
                    expansion_rules,
                )
        elif seq.type == SequenceType.GROUP:
            seq_sentences = map(
                partial(
                    sample_expression_with_output,
                    slot_lists=slot_lists,
                    expansion_rules=expansion_rules,
                ),
                seq.items,
            )
            sentence_texts = itertools.product(*seq_sentences)
            for sentence_words in sentence_texts:
                yield (
                    normalize_whitespace("".join(w[0] for w in sentence_words)),
                    normalize_whitespace(
                        "".join(w[1] for w in sentence_words if w[1] is not None)
                    ),
                )
        else:
            raise ValueError(f"Unexpected sequence type: {seq}")
    elif isinstance(expression, ListReference):
        # {list}
        list_ref: ListReference = expression
        if (not slot_lists) or (list_ref.list_name not in slot_lists):
            raise MissingListError(f"Missing slot list {{{list_ref.list_name}}}")

        slot_list = slot_lists[list_ref.list_name]
        if isinstance(slot_list, TextSlotList):
            text_list: TextSlotList = slot_list

            if not text_list.values:
                # Not necessarily an error, but may be a surprise
                _LOGGER.warning("No values for list: %s", list_ref.list_name)

            for text_value in text_list.values:
                if text_value.value_out:
                    is_first_text = True
                    for input_text, output_text in sample_expression_with_output(
                        text_value.text_in,
                        slot_lists,
                        expansion_rules,
                    ):
                        if is_first_text:
                            output_text = (
                                str(text_value.value_out)
                                if text_value.value_out is not None
                                else ""
                            )
                            is_first_text = False
                        else:
                            output_text = None

                        yield (input_text, output_text)
                else:
                    yield from sample_expression_with_output(
                        text_value.text_in,
                        slot_lists,
                        expansion_rules,
                    )
        else:
            raise ValueError(f"Unexpected slot list type: {slot_list}")
    elif isinstance(expression, RuleReference):
        # <rule>
        rule_ref: RuleReference = expression
        if (not expansion_rules) or (rule_ref.rule_name not in expansion_rules):
            raise MissingRuleError(f"Missing expansion rule <{rule_ref.rule_name}>")

        rule_body = expansion_rules[rule_ref.rule_name]
        yield from sample_expression_with_output(
            rule_body,
            slot_lists,
            expansion_rules,
        )
    else:
        raise ValueError(f"Unexpected expression: {expression}")
