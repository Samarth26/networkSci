from collections import defaultdict
import random
from typing import Any, Dict, List, Set

from sortedcontainers import SortedList


ATTRIBUTE_CARDINALITIES: Dict[str, int] = {}


class NodeDiversity:
    def __init__(self, node: str, network: Dict[str, List[str]], attributes: Dict[str, Any]) -> None:
        self.node = node
        self.specificDiversity: Dict[str, Dict[int]] = calculateDiversity(node, network, attributes, ATTRIBUTE_CARDINALITIES)
        self.neighbours = network[node]


def createRandNetwork(nodes: List[str], p: float):
    graph = {n:set() for n in nodes}

    for i, n in enumerate(nodes):
        for j in range(i+1, len(nodes)):
            m = nodes[j]
            if (random.uniform(0,1) > p):
                graph[n].add(m)

    return graph


def calculateTotalDiversity(nodeDiversity: NodeDiversity, numNodesInvolved=None):
    totalDiversity = 0

    numNodesInvolved = numNodesInvolved or (len(nodeDiversity.neighbours) + 1)

    for attr, card in ATTRIBUTE_CARDINALITIES.items():
        # calculate diversity as for each attribute between 0 and 1
        max_val = 1 / card

        # each value can contribute up to 1/(no of potential values) for each attribute
        # hence if equal no of all attributes then diversity = 1
        for count in nodeDiversity.specificDiversity[attr].values():
            totalDiversity += min(max_val, count/numNodesInvolved)

    return totalDiversity / len(ATTRIBUTE_CARDINALITIES)


def calculateDiversity(node: str, network: Dict[str, List[str]], attributes: Dict[str, Any]):
    if not attributes:
        return 1.0

    neighbours = network[node]

    specificDiversity = defaultdict(defaultdict(int))

    for attr in ATTRIBUTE_CARDINALITIES.keys():
        counter = defaultdict(int)

        counter[attributes[node][attr]] += 1

        for n in neighbours:
            attrVal = attributes[n][attr]
            counter[attrVal] += 1
            specificDiversity[attr][attrVal] += 1

    return specificDiversity


def removeWorstEdge(node: str, network: Dict[str, List[str]], diversities: Dict[str, NodeDiversity], attributes: Dict[str, Dict[str, Any]]) -> str:
    if not (network[node]):
        return
    nodeDiversity = diversities[node]
    nodeTotalDiversity = calculateTotalDiversity(nodeDiversity)

    worstNode = None, -10

    # scan all neighbours
    for neighbour in network[node]:
        nbDiversity = diversities[neighbour]
        nbTotalDiversity = calculateTotalDiversity(nbDiversity)

        for attr in ATTRIBUTE_CARDINALITIES.keys():
            nbValue = attributes[neighbour][attr]
            nodeValue = attributes[node][attr]

            nodeDiversity.specificDiversity[attr][nbValue] -= 1
            nbDiversity.specificDiversity[attr][nodeValue] -= 1

        # count only neighbours (instead of neighbours + self)
        diversityChange = calculateTotalDiversity(nbDiversity, len(nbDiversity.neighbours)) - nbTotalDiversity \
            + calculateTotalDiversity(nodeDiversity, len(nodeDiversity.neighbours)) - nodeTotalDiversity

        if diversityChange > worstNode[1]:
            worstNode = neighbour, diversityChange

        for attr in ATTRIBUTE_CARDINALITIES.keys():
            nbValue = attributes[neighbour][attr]
            nodeValue = attributes[node][attr]

            nodeDiversity.specificDiversity[attr][nbValue] += 1
            nbDiversity.specificDiversity[attr][nodeValue] += 1


    # remove worst neighbour and adjust diversity objects
    worstNodeDiversity = diversities[worstNode[0]]

    for attr in ATTRIBUTE_CARDINALITIES.keys():
        nbValue = attributes[worstNode[0]][attr]
        nodeValue = attributes[node][attr]

        nodeDiversity.specificDiversity[attr][nbValue] -= 1
        worstNodeDiversity.specificDiversity[attr][nodeValue] -= 1
    
    network[node].remove(worstNode[0])
    network[worstNode[0]].remove(node)


def addBestEdge(nodePQ: SortedList[str], network: Dict[str, Set[str]], nodeDiversities: Dict[str, NodeDiversity], attributes: Dict[str, Dict[str, Any]]):
    # get the highest priority node
    node1 = nodePQ(0)
    node1Diversity = nodeDiversities[node1]
    node1TotalDiversity = calculateTotalDiversity(node1Diversity)

    output = 0, -1

    for node2 in nodePQ:
        if node2 in network[node1]:
            continue

        node2Diversity = nodeDiversities[node2]
        node2TotalDiversity = calculateTotalDiversity(node2Diversity)

        for attr in ATTRIBUTE_CARDINALITIES.keys():
            node1Value = attributes[node1][attr]
            node2Value = attributes[node2][attr]

            # update the diversities of each node and measure the change in diversity score
            node1Diversity[attr][node2Value] += 1
            node2Diversity[attr][node1Value] += 1

        diversityChange = calculateTotalDiversity(node2Diversity, len(network[node2]) + 2) - node2TotalDiversity \
        + calculateTotalDiversity(node1Diversity, len(network[node1]) + 2) - node1TotalDiversity

        # update output if necessary
        if diversityChange > output[1]:
            output = node2, diversityChange

        for attr in ATTRIBUTE_CARDINALITIES.keys():
            node1Value = attributes[node1][attr]
            node2Value = attributes[node2][attr]

            # update the diversities of each node and measure the change in diversity score
            node1Diversity[attr][node2Value] -= 1
            node2Diversity[attr][node1Value] -= 1

    if output[0] == node1:
        return

    # update each node's info and add a link
    node2 = output[0]
    for attr in ATTRIBUTE_CARDINALITIES.keys():
        node1Value = attributes[node1][attr]
        node2Value = attributes[node2][attr]

        # update the diversities of each node and measure the change in diversity score
        node1Diversity[attr][node2Value] += 1
        node2Diversity[attr][node1Value] += 1
    
    network[node1].add(node2)
    network[node2].add(node1)


def qn4(network: Dict[str, Set[str]], attributes: Dict[str, Dict[str, Any]], kmax: int):
    """
        1. Find hubs that exceed kmax
        2. measure diversity factors and remove edges increase diversity for both nodes when removed
            - if no such edge remove with minimal net diversity decrease across both nodes
        3. add nodes by following requirements
            - find lowest degree node
            - find next lowest degree unconnected node
            - add edge if increases diversity else find next 
        4. stop adding edges once no of edges added equals number removed

        measuring diversity
        1. for each attribute find the cardinality within the network
            e.g: for country if 10 countries are mentioned then cardinality is 10
        2. for each node measure diversity for one attribute
            - di = min(count of each value i seen, no of neighbours/cardinality)
            - diversity = (sum of di for each value i) / no of neighours
        3. node's diversity = mean of diversity for all attributes
    """
    edgesRemoved = 0

    cardinalities = defaultdict(set)

    for _, nodeAttributes in attributes.values():
        for attribute, value in nodeAttributes.items():
            cardinalities[attribute].add(value)

    for k, v in cardinalities.items():
        ATTRIBUTE_CARDINALITIES[k] = len(v)

    nodeDiversities = {
        node: NodeDiversity(n, network, attributes) for n in network
    }

    for node in network.keys():
        neighbours = network[node]
        while len(neighbours) > kmax:
            removeWorstEdge(node, network, nodeDiversities, attributes)
            edgesRemoved += 1

    nodePQ = SortedList(
        (i for i, j in network.items() if len(j) < kmax), 
        key=lambda i: len(network[i])
    )

    while edgesRemoved:
        addBestEdge(nodePQ, network, nodeDiversities, attributes)