const graphContainer = document.getElementById("graphCanvas");
const graphMeta = document.getElementById("graphMeta");
const detailHint = document.getElementById("detailHint");
const detailNodeId = document.getElementById("detailNodeId");
const detailOpcode = document.getElementById("detailOpcode");
const detailShortLabel = document.getElementById("detailShortLabel");
const detailInDegree = document.getElementById("detailInDegree");
const detailOutDegree = document.getElementById("detailOutDegree");
const fitButton = document.getElementById("fitButton");
const stabilizeButton = document.getElementById("stabilizeButton");

const EDGE_COLORS = {
  0: "#00AA00",
  1: "#FF0000",
  2: "#0000FF",
  3: "#000000",
};

let network = null;

function getJobId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("jobId");
}

function buildAdjacencyMaps(nodeIds, edges) {
  const incoming = new Map();
  const outgoing = new Map();

  nodeIds.forEach((nodeId) => {
    incoming.set(String(nodeId), []);
    outgoing.set(String(nodeId), []);
  });

  edges.forEach(([source, target]) => {
    outgoing.get(String(source))?.push(String(target));
    incoming.get(String(target))?.push(String(source));
  });

  return { incoming, outgoing };
}

function updateDetailPanel(nodeId, nodeName, inDegree, outDegree) {
  detailHint.textContent = "已选中节点，可继续点击其他节点查看详情。";
  detailNodeId.textContent = nodeId;
  detailOpcode.textContent = nodeName;
  detailShortLabel.textContent = String(nodeName).slice(0, 8);
  detailInDegree.textContent = String(inDegree);
  detailOutDegree.textContent = String(outDegree);
}

function clearDetailPanel() {
  detailHint.textContent = "点击任意节点后，这里会显示 opcode、节点编号和关联信息。";
  detailNodeId.textContent = "-";
  detailOpcode.textContent = "-";
  detailShortLabel.textContent = "-";
  detailInDegree.textContent = "-";
  detailOutDegree.textContent = "-";
}

function buildNetworkData(graph) {
  const nodeIds = Object.keys(graph.nodes);
  const { incoming, outgoing } = buildAdjacencyMaps(nodeIds, graph.edges);

  const nodes = nodeIds.map((nodeId) => {
    const nodeName = graph.nodes[nodeId];
    return {
      id: Number(nodeId),
      label: nodeName,
      title: `节点编号: ${nodeId}\nOpcode: ${nodeName}`,
      shape: "dot",
      size: 20,
      color: {
        background: "#97c2fc",
        border: "#5b8fc7",
        highlight: {
          background: "#f6bd60",
          border: "#d97706",
        },
      },
      font: {
        color: "#111827",
        size: 16,
        face: "Microsoft YaHei",
      },
      metadata: {
        nodeId,
        nodeName,
        inDegree: (incoming.get(String(nodeId)) || []).length,
        outDegree: (outgoing.get(String(nodeId)) || []).length,
      },
    };
  });

  const edges = graph.edges.map(([source, target, edgeType]) => ({
    from: Number(source),
    to: Number(target),
    label: String(edgeType),
    arrows: "to",
    smooth: false,
    width: 2,
    color: EDGE_COLORS[edgeType] || EDGE_COLORS[3],
    font: {
      size: 16,
      align: "middle",
      strokeWidth: 0,
    },
  }));

  return { nodes, edges };
}

function drawGraph(graph) {
  const data = buildNetworkData(graph);
  const options = {
    autoResize: true,
    interaction: {
      dragNodes: true,
      dragView: true,
      zoomView: true,
      hover: true,
      navigationButtons: true,
      keyboard: true,
    },
    nodes: {
      borderWidth: 1.5,
      shape: "dot",
    },
    edges: {
      arrows: {
        to: {
          enabled: true,
          scaleFactor: 0.8,
        },
      },
      smooth: false,
    },
    physics: {
      enabled: true,
      stabilization: {
        enabled: true,
        iterations: 400,
        updateInterval: 25,
      },
      barnesHut: {
        gravitationalConstant: -5000,
        centralGravity: 0.3,
        springLength: 200,
        springConstant: 0.03,
        damping: 0.22,
      },
      minVelocity: 0.75,
    },
    layout: {
      improvedLayout: true,
    },
  };

  network = new vis.Network(graphContainer, data, options);

  network.on("click", (params) => {
    if (!params.nodes || params.nodes.length === 0) {
      clearDetailPanel();
      return;
    }
    const selectedNode = data.nodes.find((node) => node.id === params.nodes[0]);
    if (!selectedNode) {
      clearDetailPanel();
      return;
    }
    updateDetailPanel(
      selectedNode.metadata.nodeId,
      selectedNode.metadata.nodeName,
      selectedNode.metadata.inDegree,
      selectedNode.metadata.outDegree
    );
  });
}

async function main() {
  const jobId = getJobId();
  if (!jobId) {
    graphMeta.textContent = "缺少 jobId，无法加载图谱。";
    return;
  }

  graphMeta.textContent = `任务编号：${jobId}，正在加载 SRG 图数据...`;
  try {
    const response = await fetch(`/api/jobs/${jobId}/srg-json`);
    if (!response.ok) {
      throw new Error("图数据加载失败");
    }
    const graph = await response.json();
    graphMeta.textContent = `任务编号：${jobId}；节点数：${graph.node_count}；边数：${graph.edge_count}。`;
    drawGraph(graph);

    fitButton.addEventListener("click", () => {
      network?.fit({ animation: { duration: 400, easingFunction: "easeInOutQuad" } });
    });
    stabilizeButton.addEventListener("click", () => {
      network?.stabilize(200);
    });
  } catch (error) {
    graphMeta.textContent = `图数据加载失败：${error.message}`;
  }
}

main();
