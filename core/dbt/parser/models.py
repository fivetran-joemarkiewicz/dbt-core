
from copy import deepcopy
from dbt.context.context_config import ContextConfig
from dbt.contracts.graph.parsed import ParsedModelNode
import dbt.flags as flags
from dbt.logger import GLOBAL_LOGGER as logger
from dbt.node_types import NodeType
from dbt.parser.base import SimpleSQLParser
from dbt.parser.search import FileBlock
import dbt.tracking as tracking
from dbt import utils
from dbt_extractor import ExtractionError, py_extract_from_source  # type: ignore
from functools import reduce
from itertools import chain
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class ModelParser(SimpleSQLParser[ParsedModelNode]):
    def parse_from_dict(self, dct, validate=True) -> ParsedModelNode:
        if validate:
            ParsedModelNode.validate(dct)
        return ParsedModelNode.from_dict(dct)

    @property
    def resource_type(self) -> NodeType:
        return NodeType.Model

    @classmethod
    def get_compiled_path(cls, block: FileBlock):
        return block.path.relative_path

    # TODO when this is turned on by default, simplify the nasty if/else tree inside this method.
    def render_update(
        self, node: ParsedModelNode, config: ContextConfig
    ) -> None:
        # TODO go back to 1/100 when this is turned on by default.
        # `True` roughly 1/50 times this function is called
        sample: bool = random.randint(1, 51) == 50

        # top-level declaration of variables
        experimentally_parsed: Optional[Union[str, Dict[str, List[Any]]]] = None
        config_call_dict: Dict[str, Any] = {}
        source_calls: List[List[str]] = []
        result: List[str] = []

        # run the experimental parser if the flag is on or if we're sampling
        if flags.USE_EXPERIMENTAL_PARSER or sample:
            if self._has_banned_macro(node):
                # this log line is used for integration testing. If you change
                # the code at the beginning of the line change the tests in
                # test/integration/072_experimental_parser_tests/test_all_experimental_parser.py
                logger.debug(
                    f"1601: parser fallback to jinja because of macro override for {node.path}"
                )
                experimentally_parsed = "has_banned_macro"
            else:
                # run the experimental parser and return the results
                try:
                    experimentally_parsed = py_extract_from_source(
                        node.raw_sql
                    )
                    logger.debug(f"1699: statically parsed {node.path}")
                # if we want information on what features are barring the experimental
                # parser from reading model files, this is where we would add that
                # since that information is stored in the `ExtractionError`.
                except ExtractionError:
                    experimentally_parsed = "cannot_parse"

        # if the parser succeeded, extract some data in easy-to-compare formats
        if isinstance(experimentally_parsed, dict):
            # create second config format
            for c in experimentally_parsed['configs']:
                ContextConfig._add_config_call(config_call_dict, {c[0]: c[1]})

            # format sources TODO change extractor to match this type
            for s in experimentally_parsed['sources']:
                source_calls.append([s[0], s[1]])
            experimentally_parsed['sources'] = source_calls

        # normal dbt run
        if not flags.USE_EXPERIMENTAL_PARSER:
            # if we're sampling, compare for correctness
            if sample and isinstance(experimentally_parsed, dict):
                # if this will _never_ mutate anything `self` we could avoid these deep copies,
                # but we can't really guarantee that going forward.
                model_parser_copy = self.deepcopy()
                exp_sample_node = deepcopy(node)
                exp_sample_config = deepcopy(config)

                model_parser_copy.populate(
                    exp_sample_node,
                    exp_sample_config,
                    experimentally_parsed['refs'],
                    experimentally_parsed['sources'],
                    dict(experimentally_parsed['configs'])
                )

            # normal rendering
            super().render_update(node, config)

            # now that the sample is populated and the current values are rendered,
            # compare the two and collect the tracking messages
            result += _get_exp_sample_result(
                exp_sample_node,
                exp_sample_config,
                node,
                config,
            )

        # if the --use-experimental-parser flag was set, and the experimental parser succeeded
        elif isinstance(experimentally_parsed, Dict):
            # manually fit configs in
            config._config_call_dict = _get_config_call_dict(experimentally_parsed)

            # update the unrendered config with values from the static parser.
            # values from yaml files are in there already
            self.populate(
                node,
                config,
                experimentally_parsed['refs'],
                experimentally_parsed['sources'],
                dict(experimentally_parsed['configs'])
            )

            self.manifest._parsing_info.static_analysis_parsed_path_count += 1

        # the experimental parser didn't run on this model.
        # fall back to python jinja rendering.
        if isinstance(experimentally_parsed, str):
            if experimentally_parsed == "cannot_parse":
                result += ["01_stable_parser_cannot_parse"]
            elif experimentally_parsed == "has_banned_macro":
                result += ["08_has_banned_macro"]
            # not logging here since the reason should have been logged above
            super().render_update(node, config)
        # the experimental parser ran on this model and failed.
        # fall back to python jinja rendering.
        else:
            logger.debug(
                f"1602: parser fallback to jinja because of extractor failure for {node.path}"
            )
            super().render_update(node, config)

        # fire a tracking event. this fires one event for every sample
        # so that we have data on a per file basis. Not only can we expect
        # no false positives or misses, we can expect the number model
        # files parseable by the experimental parser to match our internal
        # testing.
        if result and tracking.active_user is not None:  # None in some tests
            tracking.track_experimental_parser_sample({
                "project_id": self.root_project.hashed_name(),
                "file_id": utils.get_hash(node),
                "status": result
            })

    # checks for banned macros
    def _has_banned_macro(
        self, node: ParsedModelNode
    ) -> bool:
        # first check if there is a banned macro defined in scope for this model file
        root_project_name = self.root_project.project_name
        project_name = node.package_name
        banned_macros = ['ref', 'source', 'config']

        all_banned_macro_keys: Iterator[str] = chain.from_iterable(
            map(
                lambda name: [
                    f"macro.{project_name}.{name}",
                    f"macro.{root_project_name}.{name}"
                ],
                banned_macros
            )
        )

        return reduce(
            lambda z, key: z or (key in self.manifest.macros),
            all_banned_macro_keys,
            False
        )

    # this method updates the model note rendered and unrendered config as well
    # as the node object. Used to populate these values when circumventing jinja
    # rendering like the static parser.
    def populate(
        self,
        node: ParsedModelNode,
        config: ContextConfig,
        refs: List[List[str]],
        sources: List[List[str]],
        configs: Dict[str, Any]
    ):
        # if there are hooks present this, it WILL render jinja. Will need to change
        # when the experimental parser supports hooks
        self.update_parsed_node_config(node, config)

        # update the unrendered config with values from the file.
        # values from yaml files are in there already
        node.unrendered_config.update(configs)

        # set refs and sources on the node object
        node.refs += refs
        node.sources += sources

        # configs don't need to be merged into the node because they
        # are read from config._config_call_dict

    # for whatever reason this works when `deepcopy(self) does not.`
    def deepcopy(self):
        return ModelParser(
            deepcopy(self.project),
            deepcopy(self.manifest),
            deepcopy(self.root_project)
        )


# pure function. safe to use elsewhere, but unlikely to be useful outside this file.
def _get_config_call_dict(
    static_parser_result: Dict[str, List[Any]]
) -> Dict[str, Any]:
    config_call_dict: Dict[str, Any] = {}

    for c in static_parser_result['configs']:
        ContextConfig._add_config_call(config_call_dict, {c[0]: c[1]})

    return config_call_dict


# returns a list of string codes to be sent as a tracking event
def _get_exp_sample_result(
    sample_node: ParsedModelNode,
    sample_config: ContextConfig,
    node: ParsedModelNode,
    config: ContextConfig
) -> List[str]:
    result: List[Tuple[int, str]] = _get_sample_result(sample_node, sample_config, node, config)

    def process(codemsg):
        code, msg = codemsg
        return f"0{code}_experimental_{msg}"

    return list(map(process, result))


# returns a list of messages and int codes and messages that need a single digit
# prefix to be prepended before being sent as a tracking event
def _get_sample_result(
    sample_node: ParsedModelNode,
    sample_config: ContextConfig,
    node: ParsedModelNode,
    config: ContextConfig
) -> List[Tuple[int, str]]:
    result: List[Tuple[int, str]] = []
    # look for false positive configs
    for k in config._config_call_dict:
        if k not in config._config_call_dict:
            result += [(2, "false_positive_config_value")]
            break

    # look for missed configs
    for k in config._config_call_dict.keys():
        if k not in sample_config._config_call_dict.keys():
            result += [(3, "missed_config_value")]
            break

    # look for false positive sources
    for s in sample_node.sources:
        if s not in node.sources:
            result += [(4, "false_positive_source_value")]
            break

    # look for missed sources
    for s in node.sources:
        if s not in sample_node.sources:
            result += [(5, "missed_source_value")]
            break

    # look for false positive refs
    for r in sample_node.refs:
        if r not in node.refs:
            result += [(6, "false_positive_ref_value")]
            break

    # look for missed refs
    for r in node.refs:
        if r not in sample_node.refs:
            result += [(7, "missed_ref_value")]
            break

    # if there are no errors, return a success value
    if not result:
        result = [(0, "exact_match")]

    return result
