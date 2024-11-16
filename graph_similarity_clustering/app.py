from neo4j import GraphDatabase
import networkx as nx
from netrd.distance import LaplacianSpectral as Dist
# from netrd.distance import DegreeDivergence as Dist
# from netrd.distance import IpsenMikhailov as Dist
# from netrd.distance import PortraitDivergence as Dist
# from netrd.distance import NetLSD as Dist # Undirected
# from netrd.distance import NetSimile as Dist # Undirected
# from netrd.distance import OnionDivergence as Dist # Undirected
# from netrd.distance import dkSeries as Dist # Undirected

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import numpy as np


def get_job_ids(driver):
    records, summary, keys = driver.execute_query(
        "MATCH (s:Stage) RETURN DISTINCT(s.jobId) as jobId",
        database_="neo4j",
    )

    return [record.values('jobId')[0] for record in records]

def build_graph(driver, job_id):
    records, summary, keys = driver.execute_query(
        "MATCH (s1:Stage {jobId: $job_id})-[l:LINKED_TO]->(s2:Stage {jobId: $job_id}) RETURN s1.stageId AS l_id, s2.stageId AS r_id;",
        job_id=job_id,
        database_="neo4j",
    )

    G = nx.DiGraph()
    for edge in records:
        G.add_edge(edge.get('l_id'), edge.get('r_id'))

    return G


def main(URI, AUTH):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()

        jobs = enumerate(get_job_ids(driver))

        graphs = {i: build_graph(driver, job_id) for i, job_id in jobs}

        def dist(j1, j2):
            return Dist().dist(graphs[j1[0]], graphs[j2[0]])

        dist_matrix = pairwise_distances(np.array(list(graphs.keys())).reshape(-1, 1), metric=dist)

        c = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='single', distance_threshold=0.1)
        print(c.fit_predict(dist_matrix))

if __name__ == '__main__':
    import tomli as toml
    from pathlib import Path

    conf = toml.load(open(Path(Path(__name__).parent, 'resources', 'config.toml'), 'rb'))

    URI = conf['uri']
    AUTH = (conf['user'], conf['pass'])

    main(URI, AUTH)