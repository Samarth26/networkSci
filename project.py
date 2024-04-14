import argparse
import time
import grequests

from bs4 import BeautifulSoup

from collections import defaultdict
import networkx as nx
import json
from thefuzz import fuzz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from typing import Any, Dict, List, Set

from sortedcontainers import SortedList


ATTRIBUTE_CARDINALITIES: Dict[str, int] = {}



##############################################################################################
######## UTILS FOR READING DATA
##############################################################################################

def read_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="xlrd")
    df = df.drop_duplicates("dblp")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    json_data = []

    for batch in range(len(df) // 20 + 1):
        start = batch*20
        print(f"scraped {start} / {len(df)}", end="\r")
        done = False
        
        while not done:
            urls = df["dblp"].iloc[start: start + 20]
            results = grequests.map((grequests.get(u, headers=headers) for u in urls))
            done = all(r.status_code != 429 for r in results)
            if not done:
                sleeptime = max(int(r.headers.get("Retry-After", 0)) for r in results)
                print("retrying after: ", sleeptime)
                time.sleep(sleeptime)
            else:
                time.sleep(1)  # space out the requests otherwise dblp will demand a 5 min time out

        for r, row in df.iloc[start: start + len(urls)].reset_index(inplace=False).iterrows():
            response = results[r]
            soup = BeautifulSoup(response.content, 'lxml', from_encoding="utf8")
            main = soup.find_all('cite', class_='data tts-content')

            dblp_name = soup.find("span", {"class": "name primary", "itemprop":"name"})
            dblp_name = dblp_name.text.lower() if dblp_name else row["name"].lower()

            combined_info = []
            for i in main:
                coauthors = i.find_all('span', itemprop='author')
                published = i.find_all('span', itemprop='datePublished')
                authors = [x.find('span', itemprop='name').text for x in coauthors]
                date = [x.text for x in published]
                
                combined = {
                    'co-authors': authors,
                    'publish-date':date
                }
                combined_info.append(combined)

            json_data.append({
                'name': dblp_name,
                'country':row['country'],
                'institution': row['institution'],
                'dblp': row['dblp'],
                'expertise': row['expertise'],
                'publish-info': combined_info
            })

    file_path = 'network.json'
    with open(file_path, 'w') as json_file:
        json.dump(json_data, json_file)

    return df


def build_graph(networkList, df):
    graph = nx.Graph()
    for author_info in networkList:
        name = author_info['name'].lower()
        graph.add_node(name, institution=author_info["institution"], country=author_info["country"])

    name_lookup = {row["dblp_name"]: i for i, row in df.iterrows()}

    for i, network in enumerate(networkList):
        print(f"{i}/{len(networkList)}", end="\r")
        name = network['name'].lower()
        for publicationList in network['publish-info']:
            for co_author_name in publicationList['co-authors']:
                co_author_name = co_author_name.lower()

                if co_author_name not in name_lookup:
                    continue

                graph.add_edge(name, co_author_name)

    return graph


##############################################################################################
######## Code for Question 1 (network stats)
##############################################################################################

def get_network_stats(graph: nx.Graph):
    giant_component: nx.Graph = graph.subgraph(max(nx.connected_components(graph), key=len))

    stats = {
        "nodeCount": graph.number_of_nodes(),
        "edgeCount": graph.size(),
        "meanDegree": np.mean(list(dict(graph.degree()).values())),
        "meanClusteringCoeff": nx.average_clustering(graph),
        "diameter": nx.approximation.diameter(giant_component),
        "GCSize": giant_component.number_of_nodes()
    }

    return stats


def plot_degree_dist(input_graph, ylim=(1e-5, 1)):
    axs = plt.subplots(1, 1)
    ax = axs[0]

    degrees = list(dict(input_graph.degree()).values())

    hist, edges = np.histogram(
        list(dict(input_graph.degree()).values()),
        bins=500,
        density=False)

    # plot the graph's degree distribution
    deg = ax.axes[0].scatter((edges[:-1] + edges[1:])/2, hist/input_graph.number_of_nodes())
    ax.axes[0].set_xscale("log", base=10)
    ax.axes[0].set_yscale("log", base=10)

    # plot a possible poisson distribution to compare
    k = np.mean(degrees)
    t = np.arange(0, max(edges), 1)
    d = np.exp(-k)*np.power(k, t)/factorial(t)
    poisson = ax.axes[0].plot(t, d, "r")

    ax.axes[0].set_ylim(*ylim)

    plt.legend([deg, poisson], labels=["Degree Distribution", "Poisson pdf based on <k>"])
    plt.xlabel('Degree (bin)')
    plt.ylabel('Degree (bin) Probability')
    plt.title('Degree pdf')

    plt.show()


def plot_assortative(input_graph):
    degree = dict(input_graph.degree())
    avg_neigh_degree = nx.average_neighbor_degree(input_graph)
    x = []
    y = []
    for k in degree.keys():
        x.append(degree[k])
        y.append(avg_neigh_degree[k])
    plt.scatter(x, y)
    # plot trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    plt.xlabel('Degree')
    plt.ylabel('Average Neighbor Degree')
    plt.title('Average Neighbor Degree vs Degree')
    plt.show()


##############################################################################################
######## Code for Question 3 (Random Network Generation)
##############################################################################################

def createRandNetwork(graph: nx.Graph, p: float):
    new_graph = nx.Graph()
    new_graph.add_nodes_from(graph)

    nodes = [*new_graph.nodes]
    num_nodes = len(nodes)
    rng = np.random.rand(num_nodes * num_nodes)

    for i, n in enumerate(nodes):
        for j in range(i+1, num_nodes):
            m = nodes[j]
            id = i * num_nodes + j - (j >= i)
            if i != j and rng[id] < p:
                new_graph.add_edge(n, m)

    return new_graph


##############################################################################################
######## Code for Question 4 (Reducing GC size and maintaining diversity)
##############################################################################################

ATTRIBUTE_CARDINALITIES: Dict[str, int] = {}

class NodeDiversity:
    def __init__(self, node: str, network: nx.Graph) -> None:
        self.node = node
        self.specificDiversity: Dict[str, Dict[int]] = calculateDiversity(node, network)
        self.neighbours = network[node]


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


def calculateDiversity(node: str, graph: nx.Graph):
    neighbours = graph[node]

    specificDiversity = defaultdict(lambda :defaultdict(int))

    node = graph.nodes[node]

    for attr in ATTRIBUTE_CARDINALITIES.keys():
        if attr in node:
            specificDiversity[attr][node[attr]] += 1

        for n in neighbours:
            neighbour = graph.nodes[n]
            attrVal = neighbour[attr]
            specificDiversity[attr][attrVal] += 1

    return specificDiversity


def removeWorstBridge(graph: nx.Graph, diversities: Dict[str, NodeDiversity], tolerance=0.2):
    num_removed = 0

    giant_component: nx.Graph = graph.subgraph(max(nx.connected_components(graph), key=len))
    bridges = [*nx.bridges(graph.subgraph(giant_component))]

    seen = set()

    i = 0
    while i < len(bridges):
        (node1, node2) = bridges[i]
        seen.add((node1, node2))
        n1Diversity = diversities[node1]
        n1TotalDiversity = calculateTotalDiversity(n1Diversity)

        n2Diversity = diversities[node2]
        n2TotalDiversity = calculateTotalDiversity(n2Diversity)

        for attr in ATTRIBUTE_CARDINALITIES.keys():
            n1Value = graph.nodes[node1][attr]
            n2Value = graph.nodes[node2][attr]

            n1Diversity.specificDiversity[attr][n2Value] -= 1
            n2Diversity.specificDiversity[attr][n1Value] -= 1

        diversityChange = calculateTotalDiversity(n1Diversity, len(n1Diversity.neighbours)) - n1TotalDiversity \
            + calculateTotalDiversity(n2Diversity, len(n2Diversity.neighbours)) - n2TotalDiversity

        if -diversityChange > tolerance:
            for attr in ATTRIBUTE_CARDINALITIES.keys():
                n1Value = graph.nodes[node1][attr]
                n2Value = graph.nodes[node2][attr]

                n1Diversity.specificDiversity[attr][n2Value] += 1
                n2Diversity.specificDiversity[attr][n1Value] += 1
            i += 1
        else:
            graph.remove_edge(node1, node2)
            num_removed += 1
            i = 0
            bridges = [b for b in nx.bridges(graph.subgraph(giant_component)) if b not in seen]

    return num_removed


def removeWorstEdge(node_name: str, graph: nx.Graph, diversities: Dict[str, NodeDiversity]) -> str:
    nodeDiversity = diversities[node_name]
    nodeTotalDiversity = calculateTotalDiversity(nodeDiversity)

    worstNode = None, -10

    node = graph.nodes[node_name]

    # scan all neighbours
    for neighbour in graph[node_name]:
        nbDiversity = diversities[neighbour]
        nbTotalDiversity = calculateTotalDiversity(nbDiversity)

        neighbour_node = graph.nodes[neighbour]

        for attr in ATTRIBUTE_CARDINALITIES.keys():
            nbValue = neighbour_node[attr]
            nodeValue = node[attr]

            nodeDiversity.specificDiversity[attr][nbValue] -= 1
            nbDiversity.specificDiversity[attr][nodeValue] -= 1

        # count only neighbours (instead of neighbours + self)
        diversityChange = calculateTotalDiversity(nbDiversity, len(nbDiversity.neighbours)) - nbTotalDiversity \
            + calculateTotalDiversity(nodeDiversity, len(nodeDiversity.neighbours)) - nodeTotalDiversity

        if diversityChange > worstNode[1]:
            worstNode = neighbour, diversityChange

        for attr in ATTRIBUTE_CARDINALITIES.keys():
            nbValue = neighbour_node[attr]
            nodeValue = node[attr]

            nodeDiversity.specificDiversity[attr][nbValue] += 1
            nbDiversity.specificDiversity[attr][nodeValue] += 1


    # remove worst neighbour and adjust diversity objects
    worstNodeDiversity = diversities[worstNode[0]]

    for attr in ATTRIBUTE_CARDINALITIES.keys():
        nbValue = graph.nodes[worstNode[0]][attr]
        nodeValue = node[attr]

        nodeDiversity.specificDiversity[attr][nbValue] -= 1
        worstNodeDiversity.specificDiversity[attr][nodeValue] -= 1
    
    graph.remove_edge(worstNode[0], node_name)


def addBestEdge(nodePQ: SortedList, graph: nx.Graph, nodeDiversities: Dict[str, NodeDiversity], kmax: int):
    # get the highest priority node
    node1 = nodePQ[0]
    node1Diversity = nodeDiversities[node1]
    node1TotalDiversity = calculateTotalDiversity(node1Diversity)

    output = node1, 0, 0

    for i, node2 in enumerate(nodePQ):
        if node2 in graph[node1] or node2 == node1 or len(graph[node2]) >= kmax:
            continue

        node2Diversity = nodeDiversities[node2]
        node2TotalDiversity = calculateTotalDiversity(node2Diversity)

        for attr in ATTRIBUTE_CARDINALITIES.keys():
            node1Value = graph.nodes[node1][attr]
            node2Value = graph.nodes[node2][attr]

            # update the diversities of each node and measure the change in diversity score
            node1Diversity.specificDiversity[attr][node2Value] += 1
            node2Diversity.specificDiversity[attr][node1Value] += 1

        diversityChange = calculateTotalDiversity(node2Diversity, len(graph[node2]) + 2) - node2TotalDiversity \
        + calculateTotalDiversity(node1Diversity, len(graph[node1]) + 2) - node1TotalDiversity

        # update output if necessary
        if diversityChange > output[1]:
            output = node2, diversityChange, i

        for attr in ATTRIBUTE_CARDINALITIES.keys():
            node1Value = graph.nodes[node1][attr]
            node2Value = graph.nodes[node2][attr]

            # update the diversities of each node and measure the change in diversity score
            node1Diversity.specificDiversity[attr][node2Value] -= 1
            node2Diversity.specificDiversity[attr][node1Value] -= 1

    if output[0] == node1:
        nodePQ.remove(node1)
        return 0

    # update each node's info and add a link
    node2 = output[0]
    for attr in ATTRIBUTE_CARDINALITIES.keys():
        node1Value = graph.nodes[node1][attr]
        node2Value = graph.nodes[node2][attr]

        # update the diversities of each node and measure the change in diversity score
        node1Diversity.specificDiversity[attr][node2Value] += 1
        node2Diversity.specificDiversity[attr][node1Value] += 1
    
    graph.add_edge(node1, node2)

    # update the priority queue
    nodePQ.pop(output[2])  # pop 2nd one first other wise the order becomes affected
    nodePQ.pop(0)
    if len(graph[node1]) < kmax:
        nodePQ.add(node1)
    if len(graph[node2]) < kmax:
        nodePQ.add(node2)

    return 1


def qn4(graph: nx.Graph, kmax: int):
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

    for node in graph.nodes:
        for attribute, value in graph.nodes[node].items():
            cardinalities[attribute].add(value)

    for k, v in cardinalities.items():
        ATTRIBUTE_CARDINALITIES[k] = len(v)

    nodeDiversities = {
        n: NodeDiversity(n, graph) for n in graph.nodes
    }

    print(f"Initial Diversity: {sum(calculateTotalDiversity(n) for n in nodeDiversities.values())}")

    # remove bridges that decrease diversity
    print("Removing bridge edges")
    edgesRemoved = removeWorstBridge(graph, nodeDiversities, 0.1)

    print("Removing non-bridge edges")
    for node in graph.nodes.keys():
        neighbours = graph[node]
        while len(neighbours) > kmax:
            removeWorstEdge(node, graph, nodeDiversities)
            edgesRemoved += 1

    # decreasing order priority queue to allow more isolates
    giant_component: nx.Graph = graph.subgraph(max(nx.connected_components(graph), key=len))
    nodePQ = SortedList(
        (i for i in graph if len(graph[i]) < kmax and i not in giant_component), 
        key=lambda i: -len(graph[i])
    )

    while edgesRemoved and len(nodePQ):
        print(f"{edgesRemoved} edges left to add back", end="\r")
        edgesRemoved -= addBestEdge(nodePQ, graph, nodeDiversities, kmax)

    print(f"Final Diversity: {sum(calculateTotalDiversity(n) for n in nodeDiversities.values())}")

    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filename', type=str, 
        help="file path of data source. Must be .xls file same as was provided"
    )
    parser.add_argument(
        "-s", "--scrape", action='store_true'
    )

    args = parser.parse_args()

    if args.scrape:
        df = read_data(args.filename)
    else:
        df = pd.read_excel(args.filename, engine="xlrd")

    networkList = json.load(open('network.json'))

    df = df.drop_duplicates(subset='dblp')
    df = df.reset_index(drop=True)

    # Make an array with the names in the networkList
    names = []
    for i in range(len(networkList)):
        names.append(networkList[i]['name'])

    # Put the names in the df
    df['dblp_name'] = names

    original_graph = build_graph(networkList, df)

    # Question 1
    print("===========================Question 1===========================")
    plot_degree_dist(original_graph)
    plot_assortative(original_graph)
    print(get_network_stats(original_graph))
    print("================================================================\n")


    # Question 2
    print("===========================Question 2===========================")
    print("================================================================\n")

    # Question 3
    # higher values of p lead the poisson calculation to become nan foor higher degrees
    print("===========================Question 3===========================")
    rand_graph = createRandNetwork(original_graph, 0.1)
    plot_degree_dist(rand_graph, ylim=(0, 1))
    plot_assortative(rand_graph)
    print(get_network_stats(rand_graph))
    print("================================================================\n")

    # Question 4
    print("===========================Question 4===========================")
    diversified_graph = build_graph(networkList, df)
    diversified_graph = qn4(diversified_graph, 50)
    plot_degree_dist(diversified_graph, ylim=(1e-5, 1))
    plot_assortative(diversified_graph)
    print(get_network_stats(diversified_graph))
    print("================================================================\n")


if __name__ == "__main__":
    main()