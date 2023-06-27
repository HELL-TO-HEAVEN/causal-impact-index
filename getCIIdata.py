import gc
import time
import pandas as pd
import pickle
from operator import itemgetter
import os
from collections import deque
import random

import dask.dataframe as dd
import torch
from rank_bm25 import *
from transformers import AutoModel, AutoTokenizer

citationGraph_path = "fullCitationDataset.parquet.gzip"
fullPapersDataset_path = (
    "/ei/iscluster/ikagrawal/newFullData/fullpapersdata_New4_title.parquet.gzip"
)
yearWiseCSVsPath = "/ei/iscluster/ikagrawal/newFullData/years5/"
outputPath = "./outputs_27june/FALSE/"
bm25Path = "/ei/iscluster/ikagrawal/newFullData/years4/"
numPartitions_citationGraph = 70  # dask level partitions for the CitationGraph
threshold = 0.93  # Threshold for top10 papers
batchSize = 20  # batch size for sentence bert model
DEBUG = False  # set to True incase of debugging, gives useful print statements.
yearPrev = -1

start = time.time()


def calculate_max_depth(tuple_list):
    max_depth = float("-inf")  # Initialize max_depth with negative infinity
    for _, depth in tuple_list:
        if depth > max_depth:
            max_depth = depth
    return max_depth


def bfs_edge_list(edge_list, start_node):
    # Create an empty set to track visited nodes
    visited = set()
    # Create a deque for BFS traversal
    queue = deque([(start_node, 0)])  # (node, level)
    while queue:
        node, level = queue.popleft()
        # Skip already visited nodes
        if node in visited:
            continue
        # Mark current node as visited
        visited.add(node)
        # Yield the current node and its level
        # print(level)
        yield node, level
        # Get the outgoing edges from the current node
        outgoing_edges = edge_list[edge_list["to"] == node]
        # Enqueue the neighboring nodes for the next level
        for _, row in outgoing_edges.iterrows():
            neighbor = row["from"]
            queue.append((neighbor, level + 1))


def getDescendants(df, start_node):
    start = time.time()
    result = df.map_partitions(bfs_edge_list, start_node).compute(
        scheduler="threads", num_workers=30
    )
    res_list = []
    # Getting the dask generator objects into actual values.
    for generator in result:
        for val in generator:
            res_list.append(val)
    del result
    res_list = sorted(res_list, key=itemgetter(1))
    print("Done in :", time.time() - start)
    return res_list


def childrenFromDescendants(descendants):
    ids_with_distance_one = []
    for id, distance in descendants:
        if distance == 1:
            ids_with_distance_one.append(id)
    return ids_with_distance_one


def listFromDescendants(descendants):
    ids_with_distance_one = []
    for id, distance in descendants:
        ids_with_distance_one.append(id)
    return ids_with_distance_one


def countDescendants(descendants):
    unique_values = set(descendants)
    return len(unique_values)


# Citation Graph is read into Dask for parallelisation
df = dd.read_parquet(citationGraph_path)
df = df.repartition(npartitions=numPartitions_citationGraph)


def getBM25TopK(query, subsetToSearch, k):
    tokenized_query = query.split(" ")
    docs = bm25.get_batch_scores(tokenized_query, subsetToSearch)
    top_n = np.argsort(docs)[::-1][:k]
    return top_n


# Cosine Similarity Model
cosineSim = torch.nn.CosineSimilarity(dim=1)
model = AutoModel.from_pretrained("allenai/specter2").cuda()
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2")
tokenizer.model_max_length = 512

df_common = pd.read_parquet(fullPapersDataset_path)

if not os.path.exists():
    os.mkdir(outputPath)

# Add the list of paperAs you need, -1 for paperB if you need the code to find most cited paperB
paperAset = []  # [[219708452, 235732009],[22006400, -1] ...]

for paperNum, paperSet in enumerate(paperAset):
    outputRow = {
        "Role": [],
        "Abstract": [],
        "Citations": [],
        "InfCit": [],
        "PaperId": [],
        "Sim": [],
    }

    paperA = int(paperSet[0])

    # print("T:  Loop Started: ", time.time() - start)
    descendants = getDescendants(df, paperA)
    childrenOfA = childrenFromDescendants(descendants)
    lisDescendants = listFromDescendants(descendants)

    numChildrenOfA = len(childrenOfA)
    if numChildrenOfA == 0:
        if DEBUG:
            print("Skipping since Num Children = 0")
        continue
    if DEBUG:
        print("Paper A", paperA, "Number of its Children", numChildrenOfA)

    dataA = df_common[df_common["id"] == paperA].head(1)

    outputRow["Role"].append("PaperA")
    outputRow["Abstract"].append(dataA["title"].item())
    outputRow["Citations"].append(dataA["citationcount"].item())
    outputRow["InfCit"].append(dataA["influentialcitationcount"].item())
    outputRow["PaperId"].append(paperA)
    outputRow["Sim"].append(dataA["year"].item())
    if DEBUG:
        print(outputRow)

    if paperSet[1] == -1:
        bestPaperB = ""
        bestPaperB_citation = -1
        bestPaperB_year = -1
        for _paperB in childrenOfA:
            try:
                citCount = (
                    df_common[df_common["id"] == _paperB]
                    .head(1)["citationcount"]
                    .item()
                )
            except:
                citCount = -2

            if citCount >= bestPaperB_citation:
                bestPaperB_citation = citCount
                bestPaperB = _paperB

    else:
        bestPaperB = int(paperSet[1])

    countPaperB = 0
    if not os.path.exists(outputPath + str(paperA)):
        os.mkdir(outputPath + str(paperA))

    for _paperB in [bestPaperB]:
        paperB = int(_paperB)
        if len(df_common[df_common["id"] == _paperB]) == 0:
            if DEBUG:
                print("  Child not found in Dataset  ")
            continue
        paperByear = df_common[df_common["id"] == _paperB].head(1)["year"].item()
        try:
            if os.path.exists(f"{yearWiseCSVsPath}{paperByear}.csvnewnew"):
                dtypes = {
                    "rowId": "Int32",
                    "id": "Int32",
                    "abstract": "string",
                    "year": "Int32",
                }
                colNames = ["rowId", "id", "abstract", "year"]
                CandidatePool = pd.read_csv(
                    f"{yearWiseCSVsPath}{paperByear}.csvnewnew",
                    dtype=dtypes,
                    names=colNames,
                    usecols=["id", "abstract"],
                )
        except:
            if DEBUG:
                print("Candidate Pool Faulty")
            continue

        if paperB in CandidatePool["id"].values:
            paperB_Abstract = (
                CandidatePool[CandidatePool["id"] == paperB].head(1)["abstract"].item()
            )
        else:
            if DEBUG:
                print("Abstract Not Found, relying on Title")
            paperB_Abstract = (
                df_common[df_common["id"] == paperB].head(1)["title"].item()
            )
        dataB = df_common[df_common["id"] == paperB].head(1)

        outputRow["Role"].append("PaperB")
        outputRow["Abstract"].append(paperB_Abstract)
        outputRow["Citations"].append(dataB["citationcount"].item())
        outputRow["InfCit"].append(dataB["influentialcitationcount"].item())
        outputRow["PaperId"].append(paperB)
        outputRow["Sim"].append(dataB["year"].item())
        if DEBUG:
            print(outputRow)

        start1 = time.time()
        file_path = f"{bm25Path}{paperByear}.csvBM25"
        if paperByear != yearPrev:
            yearPrev = paperByear
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    bm25 = pickle.load(file)
                    # bm25_sparse_matrix = csr_matrix(bm25.matrix)
            else:
                lis = []
                if DEBUG:
                    print("BM25 Doesnt Exist Creating")
                listOfAbstracts = CandidatePool["abstract"].tolist()
                if DEBUG:
                    print("Abstracts for this Candidate Pool", len(listOfAbstracts))
                for doc in listOfAbstracts:
                    try:
                        lis.append(doc.split())
                    except:
                        lis.append([])
                del listOfAbstracts
                bm25 = BM25Okapi(lis)
                del lis
        else:
            if DEBUG:
                print("Reusing BM25 Old")
        indexCandidatePool = pd.Index(range(len(CandidatePool)))
        try:
            top50set_indexes_ofCP = getBM25TopK(paperB_Abstract, indexCandidatePool, 60)
        except:
            if DEBUG:
                print("BM25 Issue")
            continue

        encoded_PaperB = model(
            tokenizer.encode(paperB_Abstract, return_tensors="pt", truncation=True).cuda()
        ).pooler_output

        finalCandidatePool = []
        count_goodPapers = 0
        countCandidatePoolDescendantsA = 0
        list_potentialPaper_id = []
        list_citationCount = []
        list_abstract = []
        list_infCit = []
        for paper in top50set_indexes_ofCP:
            if paper in lisDescendants:
                countCandidatePoolDescendantsA += 1
                if DEBUG:
                    print("descendent of paper A")
                continue
            if paper == paperB:
                if DEBUG:
                    print("Same as B")
                continue

            potentialPaper = int(CandidatePool.iloc[paper]["id"])
            abstract = CandidatePool.iloc[paper]["abstract"]
            paperCand = df_common[df_common["id"] == potentialPaper].head(1)
            potentialPaper_id = paperCand["id"].item()
            citationCount = paperCand["citationcount"].item()
            infCit = paperCand["influentialcitationcount"].item()

            list_potentialPaper_id.append(potentialPaper_id)
            list_citationCount.append(citationCount)
            list_abstract.append(abstract)
            list_infCit.append(infCit)

            if abstract == paperB_Abstract:
                if DEBUG:
                    print("Candidate Selected is same as PaperB skipping")
                continue

        full_output = []
        for batch in range(0, len(list_abstract), batchSize):
            encoded_input = tokenizer(
                list_abstract[batch : batch + batchSize],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded_input["token_type_ids"] = encoded_input["token_type_ids"].cuda()
            encoded_input["input_ids"] = encoded_input["input_ids"].cuda()
            encoded_input["attention_mask"] = encoded_input["attention_mask"].cuda()
            candidate_paper = model(**encoded_input).pooler_output
            output1 = cosineSim(encoded_PaperB, candidate_paper).tolist()
            full_output.extend(output1)

        output = full_output
        count_goodPapers = 0
        for i, _output in enumerate(output):
            if _output > threshold:
                count_goodPapers += 1
                lis_temp = [
                    _output,
                    list_potentialPaper_id[i],
                    list_citationCount[i],
                    list_abstract[i],
                    list_infCit[i],
                ]
                if lis_temp not in finalCandidatePool:
                    finalCandidatePool.append(lis_temp)
        finalCandidatePool_sorted = sorted(
            finalCandidatePool, key=itemgetter(0), reverse=True
        )
        finalCandidatePool_sorted = finalCandidatePool_sorted[:10]

        for i, finalPaper in enumerate(finalCandidatePool_sorted):
            outputRow["Role"].append("Paper " + str(i))
            outputRow["Abstract"].append(finalPaper[3])
            outputRow["Citations"].append(finalPaper[2])
            outputRow["InfCit"].append(finalPaper[4])
            outputRow["PaperId"].append(finalPaper[1])
            outputRow["Sim"].append(finalPaper[0])
            if DEBUG:
                print(outputRow)

        countChildrenofB_descendantsOfA_wrong = 0
        countChildrenofB_descendantsOfA = 0

        descA = descendants
        descB = getDescendants(df, paperB)
        successorsB = childrenFromDescendants(descB)
        successorsA = childrenFromDescendants(descA)

        for children_of_B in successorsB:
            if children_of_B in successorsA:
                countChildrenofB_descendantsOfA += 1
        averageDescendantDepth = 0
        countDesc = 0

        outputRow["Role"].append("Count of PaperA's Descendants in Candidate Pool ")
        outputRow["Abstract"].append(countCandidatePoolDescendantsA)

        outputRow["Role"].append("Count of PaperB's children built along A-B link ")
        outputRow["Abstract"].append(countChildrenofB_descendantsOfA)

        outputRow["Role"].append("Descendants of paper A")
        outputRow["Abstract"].append(f"{countDescendants(descA)}")

        outputRow["Role"].append("Descendants of paper B")
        outputRow["Abstract"].append(f"{countDescendants(descB)}")

        outputRow["Role"].append("Depth of paper A")
        outputRow["Abstract"].append(f"{calculate_max_depth(descA)}")

        outputRow["Role"].append("Depth of paper B")
        outputRow["Abstract"].append(f"{calculate_max_depth(descB)}")

        dat = pd.DataFrame(
            {
                "Role": pd.Series(outputRow["Role"]),
                "Abstract": pd.Series(outputRow["Abstract"]),
                "Citations": pd.Series(outputRow["Citations"]),
                "InfCit": pd.Series(outputRow["InfCit"]),
                "PaperId": pd.Series(outputRow["PaperId"]),
                "Sim": pd.Series(outputRow["Sim"]),
            }
        )
        dat.to_csv(outputPath + paperSet[2] + "/" + str(paperA) + "/output_" + str(paperB) + "_" + str(time.time()) + "_.csv", index=False,)
        countPaperB += 1
        if DEBUG:
            print(f"  Done {countPaperB}/{countDescendants(descA)}")
