from langgraph.graph import END, START, StateGraph

from backend.agent.nodes import WorkflowNodes
from backend.agent.state import AgentState
from backend.ingestion.embedder import embed_texts


def build_graph(vectors, graph, history, embed=embed_texts):
	nodes = WorkflowNodes(vectors, graph, history, embed)
	workflow = StateGraph(AgentState)
	stages = ["load_history", "rewrite", "retrieve", "expand", "answer", "verify", "save_history"]
	for stage in stages:
		workflow.add_node(stage, getattr(nodes, stage))
	workflow.add_edge(START, stages[0])
	for source, target in zip(stages, stages[1:]):
		workflow.add_edge(source, target)
	workflow.add_edge(stages[-1], END)
	return workflow.compile()