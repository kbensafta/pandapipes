# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models.base_component import Component
from pandapipes.pipeflow_setup import get_lookup, add_table_lookup, get_table_number
import numpy as np

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class NodeElementComponent(Component):
    """

    """

    @classmethod
    def node_element_relevant(cls, net):
        return False

    @classmethod
    def junction_name(cls):
        return 'junction'

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function that creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        raise NotImplementedError

    @classmethod
    def create_node_element_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start,
                                    current_table):
        if cls.node_element_relevant(net):
            table_indices = net[cls.table_name()].index
            table_len = len(table_indices)
            end = current_start + table_len
            ft_lookups[cls.table_name()] = (current_start, end)
            add_table_lookup(table_lookup, cls.table_name(), current_table)
            if not table_len:
                idx_lookups[cls.table_name()] = np.array([], dtype=np.int32)
                idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
            else:
                idx_lookups[cls.table_name()] = -np.ones(table_indices.max() + 1, dtype=np.int32)
                idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
            return end, current_table + 1
        else:
            return current_start, current_table

    @classmethod
    def create_pit_node_element_entries(cls, net, node_element_pit, node_name):
        if cls.node_element_relevant(net):
            ft_lookup = get_lookup(net, "node_element", "from_to")
            node_lookup = get_lookup(net, "node", "index")[node_name]
            node_element_table_nr = get_table_number(get_lookup(net, "node_element", "table"), cls.table_name())
            f, t = ft_lookup[cls.table_name()]

            node_elements = net[cls.table_name()]
            node_element_pit = node_element_pit[f:t, :]
            node_element_pit[:, :] = np.array([node_element_table_nr] + [0] *
                                              (net['_idx_node_element']['node_element_cols'] - 1))
            node_element_pit[:, net['_idx_node']['ELEMENT_IDX']] = node_elements.index.values
            node_element_pit[:, net['_idx_node_element']['JUNCTION']] = node_lookup[node_elements[cls.junction_name()].values]
            node_element_pit[:, net['_idx_node_element']['ACTIVE']] = node_elements.in_service.values
            if len(net._fluid) != 1:
                w_lookup = np.array(get_lookup(net, "node_element", "w"))
                flp, nep = np.where(node_elements.fluid.values == net._fluid[:, np.newaxis])
                node_element_pit[nep, w_lookup[flp]] = 1.
            return node_element_pit


