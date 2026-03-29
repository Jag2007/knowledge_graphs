const uploadForm = document.getElementById("upload-form");
const askForm = document.getElementById("ask-form");
const uploadStatus = document.getElementById("upload-status");
const askStatus = document.getElementById("ask-status");
const uploadSummary = document.getElementById("upload-summary");
const answerPanel = document.getElementById("answer-panel");
const uploadJsonPanel = document.getElementById("upload-json-panel");
const askJsonPanel = document.getElementById("ask-json-panel");
const uploadJsonInput = document.getElementById("upload-json-input");
const uploadJsonContent = document.getElementById("upload-json-content");
const askJsonInput = document.getElementById("ask-json-input");
const askJsonContent = document.getElementById("ask-json-content");
const uploadMessage = document.getElementById("upload-message");
const uploadChunks = document.getElementById("upload-chunks");
const uploadTriples = document.getElementById("upload-triples");
const answerText = document.getElementById("answer-text");

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function togglePanel(panel) {
  panel.classList.toggle("hidden");
}

document.getElementById("upload-json-toggle").addEventListener("click", () => {
  togglePanel(uploadJsonPanel);
});

document.getElementById("ask-json-toggle").addEventListener("click", () => {
  togglePanel(askJsonPanel);
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("pdf-file");

  if (!fileInput.files.length) {
    uploadStatus.textContent = "Please choose a PDF file first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  uploadJsonInput.textContent = prettyJson({
    file_name: fileInput.files[0].name,
    content_type: fileInput.files[0].type || "application/pdf",
  });

  uploadStatus.textContent = "Uploading the PDF and building the graph...";
  uploadSummary.classList.add("hidden");
  uploadJsonContent.textContent = prettyJson({
    status: "processing",
    message: "The PDF is being uploaded and processed. Large documents can take a little longer.",
  });

  try {
    const response = await fetch("/upload_pdf", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    uploadJsonContent.textContent = prettyJson(data);

    if (!response.ok || data.error) {
      uploadStatus.textContent = data.error || "The PDF could not be processed.";
      return;
    }

    uploadStatus.textContent = "The PDF was uploaded and the graph was created successfully.";
    uploadMessage.textContent = data.message || "Graph created successfully.";
    uploadChunks.textContent = data.chunks_processed ?? 0;
    uploadTriples.textContent = data.triples_added ?? 0;
    uploadSummary.classList.remove("hidden");
  } catch (error) {
    uploadStatus.textContent = "The upload request failed. Please try again.";
    uploadJsonContent.textContent = prettyJson({ error: String(error) });
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.getElementById("question-input").value.trim();

  if (!question) {
    askStatus.textContent = "Please enter a question.";
    return;
  }

  askStatus.textContent = "Searching the knowledge graph...";
  answerPanel.classList.add("hidden");

  try {
    const requestBody = { question };
    askJsonInput.textContent = prettyJson(requestBody);
    askJsonContent.textContent = prettyJson({
      status: "processing",
      message: "The question is being searched in the knowledge graph.",
    });
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const data = await response.json();
    askJsonContent.textContent = prettyJson(data);

    if (!response.ok) {
      askStatus.textContent = data.detail || "The question could not be processed.";
      answerText.textContent = "";
      return;
    }

    askStatus.textContent = "Answer ready.";
    answerText.textContent =
      data.answer || "It is not in the uploaded document. Please check the text.";
    answerPanel.classList.remove("hidden");
  } catch (error) {
    askStatus.textContent = "The question request failed. Please try again.";
    askJsonContent.textContent = prettyJson({ error: String(error) });
  }
});
