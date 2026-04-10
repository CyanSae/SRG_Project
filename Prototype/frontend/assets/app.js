const detectModeElement = document.getElementById("detectMode");
const sourceTypeElement = document.getElementById("sourceType");
const granularityField = document.getElementById("granularityField");
const addressField = document.getElementById("addressField");
const batchAddressField = document.getElementById("batchAddressField");
const bytecodeField = document.getElementById("bytecodeField");
const batchBytecodeField = document.getElementById("batchBytecodeField");
const bytecodeFileElement = document.getElementById("bytecodeFile");
const bytecodeElement = document.getElementById("bytecode");
const batchBytecodeElement = document.getElementById("batchBytecode");
const batchAddressElement = document.getElementById("batchAddress");
const statusBox = document.getElementById("statusBox");
const resultPanel = document.getElementById("resultPanel");
const batchResultPanel = document.getElementById("batchResultPanel");

function setStatus(message) {
  statusBox.textContent = message;
}

function setFieldVisibility(element, visible) {
  element.hidden = !visible;
  element.style.display = visible ? "" : "none";
  element.querySelectorAll("input, textarea, select").forEach((control) => {
    control.disabled = !visible;
  });
}

function toggleSourceField() {
  const isBatchMode = detectModeElement.value === "batch";
  const isAddressMode = sourceTypeElement.value === "address";

  setFieldVisibility(granularityField, isBatchMode);

  if (isBatchMode) {
    setFieldVisibility(addressField, false);
    setFieldVisibility(bytecodeField, false);
    setFieldVisibility(batchAddressField, isAddressMode);
    setFieldVisibility(batchBytecodeField, !isAddressMode);
    return;
  }

  setFieldVisibility(batchAddressField, false);
  setFieldVisibility(batchBytecodeField, false);
  setFieldVisibility(addressField, isAddressMode);
  setFieldVisibility(bytecodeField, !isAddressMode);
}

function renderRanking(elementId, rows) {
  const element = document.getElementById(elementId);
  element.innerHTML = "";
  rows.forEach((row) => {
    const item = document.createElement("li");
    item.textContent = `${row.label_zh}（${(row.confidence * 100).toFixed(2)}%）`;
    element.appendChild(item);
  });
}

function renderResult(result) {
  const data = result.data;
  resultPanel.hidden = false;
  batchResultPanel.hidden = true;
  document.getElementById("resultTitle").textContent = data.displayResult.label;
  document.getElementById("confidenceBadge").textContent = `置信度 ${(data.displayResult.confidence * 100).toFixed(2)}%`;
  document.getElementById("binarySummary").textContent = data.prediction.binary.summary;
  document.getElementById("categorySummary").textContent = data.prediction.category.summary;
  document.getElementById("subtypeSummary").textContent = data.prediction.subtype.summary;
  document.getElementById("graphStats").textContent =
    `节点数：${data.prediction.graphStats.nodeCount}，边数：${data.prediction.graphStats.edgeCount}`;
  document.getElementById("inputStats").textContent =
    `${data.input.sourceDescription}${data.input.contractAddress ? `；地址：${data.input.contractAddress}` : ""}${data.input.creationTx ? `；创建交易：${data.input.creationTx}` : ""}`;
  document.getElementById("jobStats").textContent =
    `任务编号：${data.jobId}；当前展示粒度：${data.granularityLabel}`;
  document.getElementById("downloadLink").href = data.downloads.srgJson;
  document.getElementById("visualizeLink").href = data.downloads.srgVisualizer;
  renderRanking("binaryRanking", data.prediction.binary.ranking);
  renderRanking("categoryRanking", data.prediction.category.ranking);
  renderRanking("subtypeRanking", data.prediction.subtype.ranking);
}

function renderBatchResult(result) {
  const data = result.data;
  resultPanel.hidden = true;
  batchResultPanel.hidden = false;
  document.getElementById("batchSummary").textContent =
    `批量任务编号：${data.batchJobId}；总数：${data.summary.total}；成功：${data.summary.success}。`;
  document.getElementById("batchMaliciousSummary").textContent =
    `恶意：${data.summary.malicious}；正常：${data.summary.benign}。`;
  document.getElementById("batchFailedSummary").textContent =
    `失败：${data.summary.failed}。`;

  const wrapper = document.getElementById("batchTableWrapper");
  const rows = [];
  rows.push("<table><thead><tr><th>序号</th><th>任务编号</th><th>输入来源</th><th>结果</th><th>置信度</th><th>图谱</th><th>下载</th></tr></thead><tbody>");
  data.results.forEach((item) => {
    rows.push(
      `<tr><td>${item.index}</td><td>${item.jobId}</td><td>${item.input.sourceDescription}</td><td>${item.displayResult.label}</td><td>${(item.displayResult.confidence * 100).toFixed(2)}%</td><td><a href="${item.downloads.srgVisualizer}" target="_blank" rel="noopener noreferrer">查看图谱</a></td><td><a href="${item.downloads.srgJson}">下载 SRG</a></td></tr>`
    );
  });
  if (data.failedItems.length > 0) {
    data.failedItems.forEach((item) => {
      rows.push(
        `<tr><td>${item.index}</td><td>-</td><td>失败项</td><td colspan="4">${item.error}</td></tr>`
      );
    });
  }
  rows.push("</tbody></table>");
  wrapper.innerHTML = rows.join("");
}

async function readUploadedBytecode() {
  const file = bytecodeFileElement.files?.[0];
  if (!file) {
    return "";
  }
  return await file.text();
}

function parseBatchCsv(fileContent) {
  const lines = fileContent
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return [];
  }

  const header = lines[0].split(",").map((item) => item.trim().toLowerCase());
  const addressIndex = header.indexOf("contract_address");
  const bytecodeIndex = header.indexOf("bytecode");

  if (addressIndex === -1 && bytecodeIndex === -1) {
    return [];
  }

  return lines.slice(1).map((line) => {
    const columns = line.split(",").map((item) => item.trim());
    return {
      address: addressIndex >= 0 ? (columns[addressIndex] || "") : "",
      bytecode: bytecodeIndex >= 0 ? (columns[bytecodeIndex] || "") : "",
    };
  }).filter((item) => item.address || item.bytecode);
}

async function readUploadedItems() {
  const fileContent = await readUploadedBytecode();
  const csvItems = parseBatchCsv(fileContent);
  if (csvItems.length > 0) {
    return csvItems;
  }
  return fileContent
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

document.getElementById("healthButton").addEventListener("click", async () => {
  setStatus("正在检查服务状态...");
  const response = await fetch("/api/health");
  const data = await response.json();
  setStatus(`服务状态正常。\n模型路径：${data.modelPath}`);
});

detectModeElement.addEventListener("change", toggleSourceField);
sourceTypeElement.addEventListener("change", toggleSourceField);
toggleSourceField();

document.getElementById("analyzeForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  resultPanel.hidden = true;
  batchResultPanel.hidden = true;

  const detectMode = detectModeElement.value;
  const sourceType = sourceTypeElement.value;
  const granularity = detectMode === "batch"
    ? document.getElementById("granularity").value
    : "subtype";
  let apiPath = "/api/analyze";
  let payload = { sourceType, granularity };

  if (detectMode === "batch") {
    apiPath = "/api/analyze/batch";
    let rawItems = [];
    if (sourceType === "address") {
      rawItems = batchAddressElement.value
        .split(/\r?\n/)
        .map((item) => item.trim())
        .filter(Boolean);
    } else {
      rawItems = batchBytecodeElement.value
        .split(/\r?\n/)
        .map((item) => item.trim())
        .filter(Boolean);
    }
    if (rawItems.length === 0) {
      rawItems = await readUploadedItems();
    }
    payload.items = rawItems.map((item) => {
      if (typeof item === "object" && item !== null) {
        return {
          address: item.address || "",
          bytecode: item.bytecode || "",
        };
      }
      return sourceType === "address" ? { address: item } : { bytecode: item };
    }).filter((item) => sourceType === "address" ? item.address : item.bytecode);
  } else {
    payload = {
      sourceType,
      granularity,
      address: document.getElementById("address").value.trim(),
      bytecode: bytecodeElement.value.trim(),
    };
    if (sourceType === "bytecode" && !payload.bytecode) {
      payload.bytecode = (await readUploadedBytecode()).trim();
    }
  }

  setStatus(
    detectMode === "batch"
      ? "批量任务已提交，系统正在逐条执行：输入校验 -> 字节码获取 -> SRG 构建 -> RGCN 检测。"
      : "任务已提交，系统正在依次执行：输入校验 -> 字节码获取 -> SRG 构建 -> RGCN 检测。"
  );

  try {
    const response = await fetch(apiPath, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
      throw new Error(result.error || "检测失败");
    }
    if (detectMode === "batch") {
      renderBatchResult(result);
      setStatus(
        `批量检测完成。\n批量任务编号：${result.data.batchJobId}\n成功：${result.data.summary.success}，失败：${result.data.summary.failed}。`
      );
    } else {
      renderResult(result);
      setStatus(`检测完成。\n任务编号：${result.data.jobId}\n当前结果：${result.data.displayResult.message}`);
    }
  } catch (error) {
    setStatus(`检测失败：${error.message}`);
  }
});
